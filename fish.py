#!/usr/bin/env python3
"""French river water height console tool using Hub'Eau API."""

import argparse
import json
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx
import plotext as plt

CACHE_PATH = Path.home() / ".cache" / "fish" / "hist_avg.json"

BASE = "https://hubeau.eaufrance.fr/api/v2/hydrometrie"
GEOCODE_URL = "https://api-adresse.data.gouv.fr/search/"
TIMEOUT = 30


def geocode(location: str) -> tuple[float, float]:
    """Geocode a location name using the French address API. Returns (lat, lon)."""
    resp = httpx.get(
        GEOCODE_URL,
        params={"q": location, "limit": 1},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    features = resp.json().get("features", [])
    if not features:
        print(f"Could not geocode '{location}'.", file=sys.stderr)
        sys.exit(1)
    lon, lat = features[0]["geometry"]["coordinates"]
    return lat, lon


def search_stations_nearby(lat: float, lon: float, radius_km: float) -> list[dict]:
    """Find hydrometric stations within radius_km of a point."""
    resp = httpx.get(
        f"{BASE}/referentiel/stations",
        params={
            "latitude": lat,
            "longitude": lon,
            "distance": radius_km,
            "en_service": "true",
            "format": "json",
            "size": 20,
        },
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json().get("data", [])


def search_stations(query: str) -> None:
    """Search stations by name or river and print results."""
    resp = httpx.get(
        f"{BASE}/referentiel/stations",
        params={"libelle_station": query, "size": 20, "format": "json"},
        timeout=TIMEOUT,
    )
    # Also search by river name
    resp2 = httpx.get(
        f"{BASE}/referentiel/stations",
        params={"libelle_cours_eau": query, "size": 20, "format": "json"},
        timeout=TIMEOUT,
    )
    seen = set()
    results = []
    for r in [resp, resp2]:
        r.raise_for_status()
        for s in r.json().get("data", []):
            code = s["code_station"]
            if code not in seen:
                seen.add(code)
                results.append(s)

    if not results:
        print(f"No stations found for '{query}'.")
        return

    print(f"{'Code':<15} {'Station':<40} {'River'}")
    print("─" * 80)
    for s in results:
        name = s.get("libelle_station", "?")
        river = s.get("libelle_cours_eau", "?")
        print(f"{s['code_station']:<15} {name:<40} {river}")


def get_station_info(code: str) -> dict:
    """Fetch station metadata. Exits on failure."""
    resp = httpx.get(
        f"{BASE}/referentiel/stations",
        params={"code_station": code, "format": "json"},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json().get("data", [])
    if not data:
        print(f"Station '{code}' not found.", file=sys.stderr)
        sys.exit(1)
    return data[0]


def fetch_obs_elab(
    code: str, date_min: str, date_max: str, grandeur: str | None = None
) -> list[dict]:
    """Fetch elaborated observations for a date range."""
    results = []
    cursor = None
    while True:
        params = {
            "code_entite": code,
            "date_debut_obs_elab": date_min,
            "date_fin_obs_elab": date_max,
            "size": 1000,
            "format": "json",
        }
        if grandeur:
            params["grandeur_hydro_elab"] = grandeur
        if cursor:
            params["cursor"] = cursor
        resp = httpx.get(f"{BASE}/obs_elab", params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        body = resp.json()
        results.extend(body.get("data", []))
        cursor = body.get("next")
        if not cursor:
            break
    return results


# Preferred height grandeur codes in order: daily mean, daily min, daily max
HEIGHT_GRANDEURS = ["HmnJ", "HINnJ", "HIXnJ"]


def pick_height_grandeur(obs: list[dict]) -> str | None:
    """Pick the best available height grandeur from observations."""
    available = {o["grandeur_hydro_elab"] for o in obs}
    for g in HEIGHT_GRANDEURS:
        if g in available:
            return g
    # Fallback: any code starting with H
    for g in sorted(available):
        if g.startswith("H"):
            return g
    return None


def fetch_recent_3months(code: str) -> tuple[list[str], list[float], str]:
    """Fetch last 3 months of daily water height. Returns (dates, values, grandeur_used)."""
    today = date.today()
    date_min = (today - timedelta(days=90)).isoformat()
    date_max = today.isoformat()

    obs = fetch_obs_elab(code, date_min, date_max)
    grandeur = pick_height_grandeur(obs)
    if not grandeur:
        return [], [], ""

    height_obs = [
        o
        for o in obs
        if o.get("grandeur_hydro_elab") == grandeur
        and o.get("resultat_obs_elab") is not None
    ]
    height_obs.sort(key=lambda o: o["date_obs_elab"])

    dates = [o["date_obs_elab"] for o in height_obs]
    values = [o["resultat_obs_elab"] for o in height_obs]
    return dates, values, grandeur


def fetch_historical_average(code: str, grandeur: str) -> tuple[float | None, int]:
    """Fetch today's date across the past 10 years, return average height and count."""
    today = date.today()
    values = []
    # Fetch year by year (API doesn't support multi-year ranges well)
    for year_offset in range(1, 11):
        try:
            target = today.replace(year=today.year - year_offset)
        except ValueError:
            continue  # Feb 29 in non-leap year
        d = target.isoformat()
        obs = fetch_obs_elab(code, d, d, grandeur=grandeur)
        for o in obs:
            if o.get("resultat_obs_elab") is not None:
                values.append(o["resultat_obs_elab"])
    if not values:
        return None, 0
    return sum(values) / len(values), len(values)


def load_cache() -> dict:
    """Load historical average cache from disk."""
    try:
        raw = CACHE_PATH.read_text()
        cache = json.loads(raw)
        if cache.get("year") != date.today().year:
            return {"year": date.today().year, "data": {}}
        return cache
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return {"year": date.today().year, "data": {}}


def save_cache(cache: dict) -> None:
    """Save historical average cache to disk."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache))


def prepopulate_cache(code: str, grandeur: str, cache: dict) -> None:
    """Fetch 3 months of historical averages (today+90d) across 10 years, store in cache."""
    today = date.today()
    end = today + timedelta(days=90)
    by_day: dict[str, list[float]] = defaultdict(list)

    for year_offset in range(1, 11):
        try:
            d_min = today.replace(year=today.year - year_offset)
            d_max = end.replace(year=today.year - year_offset)
        except ValueError:
            continue
        year = today.year - year_offset
        print(
            f"  Caching {code} {grandeur} [{year_offset}/10] {year}...",
            end="\r",
            file=sys.stderr,
        )
        try:
            obs = fetch_obs_elab(code, d_min.isoformat(), d_max.isoformat(), grandeur)
        except (httpx.HTTPStatusError, httpx.TimeoutException):
            continue
        for o in obs:
            val = o.get("resultat_obs_elab")
            if val is not None:
                md = o["date_obs_elab"][5:10]  # MM-DD
                by_day[md].append(val)

    # Clear progress line
    print(" " * 60, end="\r", file=sys.stderr)

    for md, vals in by_day.items():
        key = f"{code}:{md}:{grandeur}"
        cache["data"][key] = [sum(vals) / len(vals), len(vals)]


def get_historical_average(
    code: str, grandeur: str, cache: dict
) -> tuple[float | None, int]:
    """Get cached historical average, prepopulating on miss."""
    md = date.today().strftime("%m-%d")
    key = f"{code}:{md}:{grandeur}"
    if key not in cache["data"]:
        prepopulate_cache(code, grandeur, cache)
    entry = cache["data"].get(key)
    if entry:
        return entry[0], entry[1]
    return None, 0


def display_table(
    rows: list[tuple[str, str, str, float | None, float | None, int]],
) -> None:
    """Print a table of station data."""
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    headers = ("River", "Station", "Code", "Today", "10y avg")
    ra = (False, False, False, True, True)
    # Compute column widths
    fmt_rows = []
    for river, name, code, today_val, avg_val, avg_count in rows:
        today_s = f"{today_val:.0f} mm" if today_val is not None else "— mm"
        if avg_val is not None:
            avg_s = f"{avg_val:.0f} mm ({avg_count}y)"
        else:
            avg_s = "—"
        fmt_rows.append((river, name, code, today_s, avg_s))

    col_w = [len(h) for h in headers]
    for r in fmt_rows:
        for i, cell in enumerate(r):
            col_w[i] = max(col_w[i], len(cell))

    def row_str(cells: tuple[str, ...], right_align: tuple[bool, ...] = ra) -> str:
        parts = []
        for cell, w, r in zip(cells, col_w, right_align):
            parts.append(cell.rjust(w) if r else cell.ljust(w))
        return "  ".join(parts)

    print(f"  {BOLD}{row_str(headers)}{RESET}")
    print(f"  {'─' * (sum(col_w) + 2 * (len(col_w) - 1))}")
    for river, name, code, today_s, avg_s in fmt_rows:
        r_str = row_str((river, name, f"{DIM}{code}{RESET}", today_s, avg_s))
        print(f"  {r_str}")
    print()


def fetch_rain_forecast(lat: float, lon: float) -> list[tuple[str, float]]:
    """Fetch 8-hour rain forecast from Open-Meteo. Returns list of (hour_label, mm)."""
    try:
        resp = httpx.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": "precipitation",
                "forecast_hours": 8,
            },
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json().get("hourly", {})
        times = data.get("time", [])
        precip = data.get("precipitation", [])
        return [(t.split("T")[1], p) for t, p in zip(times, precip)]
    except Exception:
        return []


def fetch_sunlight(lat: float, lon: float) -> dict | None:
    """Fetch sunrise/sunset from Open-Meteo. Returns dict with times or None."""
    try:
        resp = httpx.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "sunrise,sunset",
                "timezone": "auto",
                "forecast_days": 1,
            },
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        daily = resp.json().get("daily", {})
        sunrise = daily["sunrise"][0]
        sunset = daily["sunset"][0]
        sr = datetime.fromisoformat(sunrise)
        ss = datetime.fromisoformat(sunset)
        noon = sr + (ss - sr) / 2
        return {
            "sunrise": sr.strftime("%H:%M"),
            "sunset": ss.strftime("%H:%M"),
            "peak_start": (noon - timedelta(hours=2)).strftime("%H:%M"),
            "peak_end": (noon + timedelta(hours=2)).strftime("%H:%M"),
        }
    except Exception:
        return None


def display(
    station: dict,
    dates: list[str],
    values: list[float],
    avg: float | None,
    avg_count: int,
) -> None:
    """Render the graph and summary."""
    name = station.get("libelle_station", "?")
    river = station.get("libelle_cours_eau", "?")
    code = station["code_station"]

    # ANSI escape codes
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    RESET = "\033[0m"

    # Warez-style header box
    title = f"{river} à {name}"
    code_line = f"[{code}]"
    box_width = max(len(title), len(code_line)) + 4
    print(f"{CYAN}┌{'─' * box_width}┐{RESET}")
    print(
        f"{CYAN}│{RESET}  {BOLD}{CYAN}{title}{RESET}{' ' * (box_width - len(title) - 2)}{CYAN}│{RESET}"
    )
    print(
        f"{CYAN}│{RESET}  {GREEN}{code_line}{RESET}{' ' * (box_width - len(code_line) - 2)}{CYAN}│{RESET}"
    )
    print(f"{CYAN}└{'─' * box_width}┘{RESET}")

    if dates and values:
        plt.clear_figure()
        plt.plot_size(80, 20)
        plt.theme("dark")
        plt.canvas_color("black")
        plt.axes_color("black")
        plt.ticks_color("cyan")
        plt.ticks_style("bold")
        plt.grid(False)
        plt.title("Water Height — Last 3 Months (mm)")
        # Use short date labels
        labels = [d[5:] for d in dates]  # MM-DD
        plt.plot(list(range(len(values))), values, color="green+", marker="braille")
        # Show ~6 tick labels
        step = max(1, len(labels) // 6)
        xticks = list(range(0, len(labels), step))
        xlabels = [labels[i] for i in xticks]
        plt.xticks(xticks, xlabels)
        plt.ylabel("mm")
        plt.show()
    else:
        print(f"  {GREEN}No recent data available for this station.{RESET}")

    print()
    today = date.today()
    if avg is not None:
        print(
            f"  {GREEN}>>{RESET} Today's average ({today.strftime('%b %d')}, {avg_count}-year): {BOLD}{GREEN}{avg:.0f} mm{RESET}"
        )
    else:
        print(
            f"  {GREEN}>>{RESET} No historical data available for {today.strftime('%b %d')}."
        )


def fetch_today_level(code: str) -> float | None:
    """Fetch the latest real-time water height for a station."""
    resp = httpx.get(
        f"{BASE}/observations_tr",
        params={
            "code_entite": code,
            "grandeur_hydro": "H",
            "size": 1,
            "sort": "desc",
            "format": "json",
        },
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json().get("data", [])
    if data and data[0].get("resultat_obs") is not None:
        return data[0]["resultat_obs"]
    return None


def plot_station(station: dict) -> None:
    """Fetch data and display plot for a single station."""
    code = station["code_station"]
    try:
        dates, values, grandeur = fetch_recent_3months(code)
        avg, avg_count = (
            fetch_historical_average(code, grandeur) if grandeur else (None, 0)
        )
    except httpx.HTTPStatusError as e:
        name = station.get("libelle_station", code)
        print(
            f"  Skipping {name} ({code}): API error {e.response.status_code}",
            file=sys.stderr,
        )
        return
    display(station, dates, values, avg, avg_count)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="French river water height console tool"
    )
    parser.add_argument("location", nargs="?", help="Location name (e.g. Paris, Lyon)")
    parser.add_argument(
        "--station", metavar="CODE", help="Plot a specific station by code"
    )
    parser.add_argument(
        "--station-list", metavar="QUERY", help="Search stations by name or river"
    )
    args = parser.parse_args()

    if args.station_list:
        search_stations(args.station_list)
        return

    if args.station:
        station = get_station_info(args.station)
        plot_station(station)
        return

    if not args.location:
        parser.print_help()
        sys.exit(1)

    lat, lon = geocode(args.location)
    stations = search_stations_nearby(lat, lon, 25)
    if not stations:
        print(f"No stations found within 25 km of '{args.location}'.")
        sys.exit(1)
    print(f"Found {len(stations)} station(s) near {args.location}\n")

    DIM = "\033[2m"
    RESET = "\033[0m"

    cache = load_cache()
    rows = []
    for station in stations:
        code = station["code_station"]
        name = station.get("libelle_station") or "?"
        river = station.get("libelle_cours_eau") or "?"
        try:
            today_val = fetch_today_level(code)
        except (httpx.HTTPStatusError, httpx.TimeoutException):
            print(
                f"  {DIM}{river} à {name} ({code}): unavailable{RESET}",
                file=sys.stderr,
            )
            continue
        # Pick grandeur from recent data for historical average
        try:
            _, _, grandeur = fetch_recent_3months(code)
        except (httpx.HTTPStatusError, httpx.TimeoutException):
            grandeur = ""
        if grandeur:
            avg_val, avg_count = get_historical_average(code, grandeur, cache)
        else:
            avg_val, avg_count = None, 0
        rows.append((river, name, code, today_val, avg_val, avg_count))

    display_table(rows)
    save_cache(cache)

    # Weather and sunlight for the searched location
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    RESET = "\033[0m"

    forecast = fetch_rain_forecast(lat, lon)
    if forecast:
        print(f"  {BOLD}Rain forecast (next 8h):{RESET}")
        max_mm = max(mm for _, mm in forecast)
        for hour, mm in forecast:
            bar = "▇" * round(mm / max_mm * 10) if max_mm > 0 and mm > 0 else ""
            print(f"  {hour}  {mm:4.1f} mm  {CYAN}{bar}{RESET}")
        print()

    sun = fetch_sunlight(lat, lon)
    if sun:
        print(f"  {BOLD}Sunlight:{RESET}")
        print(f"  {YELLOW}☀{RESET}  Sunrise       {sun['sunrise']}")
        print(f"  {YELLOW}☀{RESET}  Sunset        {sun['sunset']}")
        print(
            f"  {YELLOW}☀{RESET}  Peak sunlight {sun['peak_start']} – {sun['peak_end']}"
        )


if __name__ == "__main__":
    main()
