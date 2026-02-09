#!/usr/bin/env python3
"""French river water height console tool using Hub'Eau API."""

import argparse
import sys
from datetime import date, timedelta

import httpx
import plotext as plt

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


def display_station_info(station: dict) -> None:
    """Display station metadata in a box."""
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    RESET = "\033[0m"

    name = station.get("libelle_station", "?")
    river = station.get("libelle_cours_eau", "?")
    code = station["code_station"]

    fields = [
        ("Station", name),
        ("Code", code),
        ("River", river),
        (
            "Coordinates",
            f"{station.get('latitude_station', '?')}, {station.get('longitude_station', '?')}",
        ),
        (
            "Altitude",
            f"{station.get('altitude_ref_alti_station', '?')} m"
            if station.get("altitude_ref_alti_station") is not None
            else "?",
        ),
        ("Status", "Active" if station.get("en_service") else "Inactive"),
        ("Département", station.get("code_departement", "?")),
    ]

    max_label = max(len(f[0]) for f in fields)
    max_value = max(len(str(f[1])) for f in fields)
    box_width = max_label + max_value + 7

    print(f"{CYAN}┌{'─' * box_width}┐{RESET}")
    title = f"{river} à {name}"
    print(
        f"{CYAN}│{RESET}  {BOLD}{CYAN}{title}{RESET}{' ' * (box_width - len(title) - 2)}{CYAN}│{RESET}"
    )
    print(f"{CYAN}├{'─' * box_width}┤{RESET}")
    for label, value in fields:
        line = f"{GREEN}{label:<{max_label}}{RESET}  {value}"
        visible_len = max_label + 2 + len(str(value))
        print(
            f"{CYAN}│{RESET}  {line}{' ' * (box_width - visible_len - 2)}{CYAN}│{RESET}"
        )
    print(f"{CYAN}└{'─' * box_width}┘{RESET}")

    forecast = fetch_rain_forecast(
        station.get("latitude_station", 0), station.get("longitude_station", 0)
    )
    if forecast:
        print(f"\n  {BOLD}Rain forecast (next 8h):{RESET}")
        max_mm = max(mm for _, mm in forecast)
        for hour, mm in forecast:
            bar = "▇" * round(mm / max_mm * 10) if max_mm > 0 and mm > 0 else ""
            print(f"  {hour}  {mm:4.1f} mm  {CYAN}{bar}{RESET}")


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


def plot_station(station: dict) -> None:
    """Fetch data and display plot for a single station."""
    code = station["code_station"]
    dates, values, grandeur = fetch_recent_3months(code)
    avg, avg_count = fetch_historical_average(code, grandeur) if grandeur else (None, 0)
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
    parser.add_argument(
        "--station-info", action="store_true", help="Show station details only"
    )
    args = parser.parse_args()

    if args.station_list:
        search_stations(args.station_list)
        return

    if args.station_info:
        if not args.station:
            parser.error("--station-info requires --station CODE")
        station = get_station_info(args.station)
        display_station_info(station)
        return

    if args.station:
        station = get_station_info(args.station)
        plot_station(station)
        return

    if not args.location:
        parser.print_help()
        sys.exit(1)

    lat, lon = geocode(args.location)
    stations = search_stations_nearby(lat, lon, 100)
    if not stations:
        print(f"No stations found within 100 km of '{args.location}'.")
        sys.exit(1)
    print(f"Found {len(stations)} station(s) near {args.location}\n")
    for station in stations:
        plot_station(station)


if __name__ == "__main__":
    main()
