#!/usr/bin/env python3
"""French river water height console tool using Hub'Eau API."""

import argparse
import json
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import httpx
import plotext as plt

CACHE_PATH = Path.home() / ".cache" / "fish" / "hist_avg.json"

BASE = "https://hubeau.eaufrance.fr/api/v2/hydrometrie"
GEOCODE_URL = "https://api-adresse.data.gouv.fr/search/"
TIMEOUT = 30

# ANSI escape codes
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"


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
    params: dict = {
        "latitude": lat,
        "longitude": lon,
        "distance": radius_km,
        "en_service": "true",
        "format": "json",
        "size": 20,
    }
    stations: list[dict] = []
    url: str | None = None
    while True:
        if url:
            resp = httpx.get(url, timeout=TIMEOUT)
        else:
            resp = httpx.get(
                f"{BASE}/referentiel/stations", params=params, timeout=TIMEOUT
            )
        resp.raise_for_status()
        body = resp.json()
        stations.extend(body.get("data", []))
        url = body.get("next")
        if not url:
            break
    return stations


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
    url = None
    while True:
        if url:
            resp = httpx.get(url, timeout=TIMEOUT)
        else:
            params = {
                "code_entite": code,
                "date_debut_obs_elab": date_min,
                "date_fin_obs_elab": date_max,
                "size": 1000,
                "format": "json",
            }
            if grandeur:
                params["grandeur_hydro_elab"] = grandeur
            resp = httpx.get(f"{BASE}/obs_elab", params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        body = resp.json()
        results.extend(body.get("data", []))
        url = body.get("next")
        if not url:
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


def fetch_date_level(code: str, target_date: date) -> float | None:
    """Fetch the elaborated daily water height for a specific date."""
    d = target_date.isoformat()
    obs = fetch_obs_elab(code, d, d)
    grandeur = pick_height_grandeur(obs)
    if not grandeur:
        return None
    for o in obs:
        if (
            o.get("grandeur_hydro_elab") == grandeur
            and o.get("resultat_obs_elab") is not None
        ):
            return o["resultat_obs_elab"]
    return None


def fetch_recent_3months(
    code: str, target_date: date | None = None
) -> tuple[list[str], list[float], str]:
    """Fetch last 3 months of daily water height. Returns (dates, values, grandeur_used)."""
    today = date.today()
    end = min(target_date or today, today)
    date_min = (end - timedelta(days=90)).isoformat()
    date_max = end.isoformat()

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


def fetch_historical_average(
    code: str, grandeur: str, target_date: date | None = None
) -> tuple[float | None, int]:
    """Fetch target date across the past 10 years, return average height and count."""
    ref = target_date or date.today()
    values = []
    # Fetch year by year (API doesn't support multi-year ranges well)
    for year_offset in range(1, 11):
        try:
            target = ref.replace(year=ref.year - year_offset)
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


def prepopulate_cache(
    code: str, grandeur: str, cache: dict, target_date: date | None = None
) -> None:
    """Fetch 3 months of historical averages starting at target_date across 10 years, store in cache."""
    ref = target_date or date.today()
    by_day: dict[str, list[float]] = defaultdict(list)

    for year_offset in range(1, 11):
        try:
            d_min = ref.replace(year=ref.year - year_offset)
        except ValueError:
            continue
        d_max = d_min + timedelta(days=90)
        year = ref.year - year_offset
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
    code: str, grandeur: str, cache: dict, target_date: date | None = None
) -> tuple[float | None, int]:
    """Get cached historical average, prepopulating on miss."""
    ref = target_date or date.today()
    md = ref.strftime("%m-%d")
    key = f"{code}:{md}:{grandeur}"
    if key not in cache["data"]:
        prepopulate_cache(code, grandeur, cache, target_date)
    entry = cache["data"].get(key)
    if entry:
        return entry[0], entry[1]
    return None, 0


def display_table(
    rows: list[tuple[str, str, str, float | None, float | None, int]],
    target_date: date | None = None,
) -> None:
    """Print a table of station data."""
    ref = target_date or date.today()
    date_label = "Today" if ref >= date.today() else ref.strftime("%b %d")
    headers = ("River", "Station", "Code", date_label, "10y avg")
    ra = (False, False, False, True, True)
    # Filter and format rows
    fmt_rows = []
    for river, name, code, today_val, avg_val, avg_count in rows:
        if not code or river == "?" or today_val is None:
            continue
        today_s = f"{today_val:.0f} mm"
        if avg_val is not None:
            avg_s = f"{avg_val:.0f} mm ({avg_count}y)"
        else:
            avg_s = "—"
        fmt_rows.append((river, name, code, today_s, avg_s))

    col_w = [len(h) for h in headers]
    for r in fmt_rows:
        for i, cell in enumerate(r):
            col_w[i] = max(col_w[i], len(cell))

    # Shrink River (0) and Station (1) to fit within 79 chars (2 indent + content)
    max_content = 79 - 2  # 2-char left indent
    separators = 2 * (len(col_w) - 1)
    while sum(col_w) + separators > max_content:
        # Shrink the wider of River/Station first
        shrink = 0 if col_w[0] >= col_w[1] else 1
        if col_w[shrink] <= len(headers[shrink]):
            shrink = 1 - shrink
        if col_w[shrink] <= len(headers[shrink]):
            break
        col_w[shrink] -= 1

    def truncate(text: str, width: int) -> str:
        return text[:width] if len(text) > width else text

    def row_str(cells: tuple[str, ...], right_align: tuple[bool, ...] = ra) -> str:
        parts = []
        for cell, w, r in zip(cells, col_w, right_align):
            cell = truncate(cell, w)
            parts.append(cell.rjust(w) if r else cell.ljust(w))
        return "  ".join(parts)

    row_width = sum(col_w) + separators
    print(f"  {BOLD}{row_str(headers)}{RESET}")
    print(f"  {'─' * row_width}")
    for river, name, code, today_s, avg_s in fmt_rows:
        r_str = row_str((river, name, code, today_s, avg_s))
        # Apply dim styling to the code column after padding
        r_str = r_str.replace(code, f"{DIM}{code}{RESET}", 1)
        print(f"  {r_str}")
    print()


COMPASS_LABELS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def degrees_to_compass(degrees: float) -> str:
    """Convert wind direction in degrees to compass label."""
    return COMPASS_LABELS[round(degrees / 45) % 8]


def _hhmm_to_minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def is_fishable_hour(hour: str, sunrise: str, sunset: str) -> bool:
    """Check if an hour falls within the fishable window (sunrise-30min to sunset+30min)."""
    t = _hhmm_to_minutes(hour)
    start = _hhmm_to_minutes(sunrise) - 30
    end = _hhmm_to_minutes(sunset) + 30
    return start <= t <= end


def wind_color(speed: float, gust: float) -> str:
    """Return wind condition label from a fly fishing perspective."""
    if speed > 25 or gust > 30:
        return "red"
    if speed >= 15 or gust >= 20:
        return "yellow"
    return "green"


# Technique-specific thresholds.
# Wind/gust in km/h. Precipitation in mm/h (universal, not per-technique).
TECHNIQUES = (
    "Mouche sèche",
    "Mouche nymphe",
    "Lancer UL",
    "Leurre 7g+",
    "Toc",
    "Silure au posé",
)

_TECHNIQUE_THRESHOLDS = {
    "Mouche sèche": {
        "difficult": {"wind": 15, "gust": 25},
        "no-go": {"wind": 25, "gust": 40},
    },
    "Mouche nymphe": {
        "difficult": {"wind": 20, "gust": 30},
        "no-go": {"wind": 30, "gust": 45},
    },
    "Lancer UL": {
        "difficult": {"wind": 20, "gust": 35},
        "no-go": {"wind": 30, "gust": 50},
    },
    "Leurre 7g+": {
        "difficult": {"wind": 30, "gust": 45},
        "no-go": {"wind": 45, "gust": 60},
    },
    "Toc": {
        "difficult": {"wind": 25, "gust": 40},
        "no-go": {"wind": 40, "gust": 60},
    },
    "Silure au posé": {
        "difficult": {"wind": 35, "gust": 50},
        "no-go": {"wind": 50, "gust": 70},
    },
}
_HARD_STOP_GUST = 70  # km/h
_PRECIP_DIFFICULT = 4  # mm/h
_PRECIP_NOGO = 8  # mm/h
_THUNDERSTORM_CODES = {95, 96, 99}

_VERDICT_SEVERITY = {"go": 0, "difficult": 1, "no-go": 2}


def _worst_verdict(a: str, b: str) -> str:
    return a if _VERDICT_SEVERITY[a] >= _VERDICT_SEVERITY[b] else b


def technique_verdicts(hours: list[dict], weathercode: int) -> dict[str, str]:
    """Compute per-technique go/difficult/no-go from hourly weather data."""
    if weathercode in _THUNDERSTORM_CODES:
        return {t: "no-go" for t in TECHNIQUES}

    fishable = [h for h in hours if h.get("fishable", True)]
    if not fishable:
        return {t: "go" for t in TECHNIQUES}

    # Check hard stops across all fishable hours
    for h in fishable:
        if h["wind_gust_kmh"] > _HARD_STOP_GUST or h["precipitation"] >= _PRECIP_NOGO:
            return {t: "no-go" for t in TECHNIQUES}

    # Precipitation verdict (universal)
    precip_verdict = "go"
    for h in fishable:
        if h["precipitation"] >= _PRECIP_DIFFICULT:
            precip_verdict = "difficult"
            break

    # Per-technique wind verdict
    result = {}
    for tech in TECHNIQUES:
        thresholds = _TECHNIQUE_THRESHOLDS[tech]
        wind_verdict = "go"
        for h in fishable:
            speed = h["wind_kmh"]
            gust = h["wind_gust_kmh"]
            if (
                speed >= thresholds["no-go"]["wind"]
                or gust >= thresholds["no-go"]["gust"]
            ):
                wind_verdict = "no-go"
                break
            if (
                speed >= thresholds["difficult"]["wind"]
                or gust >= thresholds["difficult"]["gust"]
            ):
                wind_verdict = _worst_verdict(wind_verdict, "difficult")
        result[tech] = _worst_verdict(wind_verdict, precip_verdict)

    return result


def fetch_daily_forecast(
    lat: float,
    lon: float,
    days: int,
    start_date: date | None = None,
) -> list[dict]:
    """Fetch daily + hourly weather from Open-Meteo.

    Each day includes wind_verdict, peak_start/peak_end, and each hourly
    entry gets fishable flag, wind direction, and condition label.
    Uses archive API for past dates, forecast API otherwise.
    """
    daily_params = (
        "temperature_2m_max,temperature_2m_min,"
        "precipitation_sum,windspeed_10m_max,"
        "winddirection_10m_dominant,weathercode,"
        "sunrise,sunset"
    )
    hourly_params = (
        "temperature_2m,precipitation,"
        "windspeed_10m,wind_gusts_10m,wind_direction_10m,"
        "cloudcover"
    )
    try:
        use_archive = start_date is not None and start_date < date.today()
        if use_archive:
            end = start_date + timedelta(days=days - 1)
            url = "https://archive-api.open-meteo.com/v1/archive"
            params: dict = {
                "latitude": lat,
                "longitude": lon,
                "daily": daily_params,
                "hourly": hourly_params,
                "timezone": "auto",
                "start_date": start_date.isoformat(),
                "end_date": end.isoformat(),
            }
        else:
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat,
                "longitude": lon,
                "daily": daily_params,
                "hourly": hourly_params,
                "timezone": "auto",
                "forecast_days": days,
            }
        resp = httpx.get(url, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        body = resp.json()
        daily = body.get("daily", {})
        hourly = body.get("hourly", {})

        # Index hourly data by date
        hourly_by_date: dict[str, list[dict]] = defaultdict(list)
        h_times = hourly.get("time", [])
        h_temps = hourly.get("temperature_2m", [])
        h_precip = hourly.get("precipitation", [])
        h_wind = hourly.get("windspeed_10m", [])
        h_gusts = hourly.get("wind_gusts_10m", [])
        h_dirs = hourly.get("wind_direction_10m", [])
        h_cloud = hourly.get("cloudcover", [])
        for i, t in enumerate(h_times):
            d, hm = t.split("T")
            dir_deg = h_dirs[i] if i < len(h_dirs) else 0
            hourly_by_date[d].append(
                {
                    "hour": hm,
                    "temp": h_temps[i],
                    "precipitation": h_precip[i],
                    "wind_kmh": h_wind[i],
                    "wind_gust_kmh": h_gusts[i],
                    "direction_deg": dir_deg,
                    "direction_compass": degrees_to_compass(dir_deg),
                    "cloudcover": h_cloud[i],
                }
            )

        result = []
        d_times = daily.get("time", [])
        for i, d in enumerate(d_times):
            sr = daily["sunrise"][i].split("T")[1]
            ss = daily["sunset"][i].split("T")[1]
            sr_min = _hhmm_to_minutes(sr)
            ss_min = _hhmm_to_minutes(ss)
            noon_min = sr_min + (ss_min - sr_min) // 2
            hours = hourly_by_date.get(d, [])
            # Mark fishable + compute worst wind during fishable hours
            worst = "green"
            for h in hours:
                h["fishable"] = is_fishable_hour(h["hour"], sr, ss)
                if h["fishable"]:
                    c = wind_color(h["wind_kmh"], h["wind_gust_kmh"])
                    if _WIND_SEVERITY[c] > _WIND_SEVERITY[worst]:
                        worst = c
            wcode = daily["weathercode"][i]
            verdicts = technique_verdicts(hours, wcode)
            result.append(
                {
                    "date": d,
                    "temp_max": daily["temperature_2m_max"][i],
                    "temp_min": daily["temperature_2m_min"][i],
                    "precipitation_sum": daily["precipitation_sum"][i],
                    "wind_max_kmh": daily["windspeed_10m_max"][i],
                    "wind_direction_dominant": (
                        degrees_to_compass(daily["winddirection_10m_dominant"][i])
                    ),
                    "weathercode": wcode,
                    "sunrise": sr,
                    "sunset": ss,
                    "peak_start": f"{(noon_min - 120) // 60:02d}:{(noon_min - 120) % 60:02d}",
                    "peak_end": f"{(noon_min + 120) // 60:02d}:{(noon_min + 120) % 60:02d}",
                    "wind_verdict": _WIND_VERDICTS[worst],
                    "technique_verdicts": verdicts,
                    "thresholds": {
                        "wind_unit": "km/h",
                        "gust_unit": "km/h",
                        "precip_unit": "mm/h",
                        "precip_difficult": _PRECIP_DIFFICULT,
                        "precip_nogo": _PRECIP_NOGO,
                        "techniques": {
                            name: _TECHNIQUE_THRESHOLDS[name] for name in TECHNIQUES
                        },
                    },
                    "hourly": hours,
                }
            )
        return result
    except (httpx.HTTPError, KeyError, IndexError):
        return []


def fetch_station_data(code: str, target_date: date | None, cache: dict) -> dict:
    """Fetch 3-month history and historical average for a station."""
    dates, values, grandeur = fetch_recent_3months(code, target_date)
    if grandeur:
        avg, avg_count = get_historical_average(code, grandeur, cache, target_date)
    else:
        avg, avg_count = None, 0
    return {
        "dates": dates,
        "values": values,
        "grandeur": grandeur,
        "avg": avg,
        "avg_count": avg_count,
    }


def display(
    station: dict,
    dates: list[str],
    values: list[float],
    avg: float | None,
    avg_count: int,
    target_date: date | None = None,
) -> None:
    """Render the graph and summary."""
    name = station.get("libelle_station", "?")
    code = station["code_station"]

    # Warez-style header box
    title = name
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
    ref = target_date or date.today()
    is_today = ref == date.today()
    label = ref.strftime("%b %d")
    if avg is not None:
        avg_label = (
            f"Today's average ({label},"
            if is_today
            else f"Historical average ({label},"
        )
        print(
            f"  {GREEN}>>{RESET} {avg_label} {avg_count}-year): {BOLD}{GREEN}{avg:.0f} mm{RESET}"
        )
    else:
        print(f"  {GREEN}>>{RESET} No historical data available for {label}.")


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


def print_rain_section(day: dict, is_today: bool) -> None:
    """Print rain section from a forecast day dict."""
    hours = day.get("hourly", [])
    if not hours:
        return
    rain_label = "Rain forecast" if is_today else "Rain"
    print(f"  {BOLD}{rain_label}:{RESET}")
    max_mm = max((h["precipitation"] for h in hours), default=0)
    for h in hours:
        mm = h["precipitation"]
        bar_len = round(mm / max_mm * 10) if max_mm > 0 and mm > 0 else 0
        color = CYAN if h.get("fishable", True) else DIM
        bar = f"{color}{'▇' * bar_len}{RESET}" if bar_len else ""
        print(f"  {h['hour']}  {mm:4.1f} mm  {bar}")
    print()


_WIND_VERDICTS = {
    "green": "Great for casting",
    "yellow": "Challenging conditions",
    "red": "Too windy for fly fishing",
}
_WIND_SEVERITY = {"green": 0, "yellow": 1, "red": 2}
_WIND_ANSI = {"green": GREEN, "yellow": YELLOW, "red": RED}


def print_wind_section(day: dict, is_today: bool) -> None:
    """Print wind section from a forecast day dict."""
    hours = day.get("hourly", [])
    if not hours:
        return
    label = "Wind forecast" if is_today else "Wind"
    print(f"  {BOLD}{label}:{RESET}")
    max_speed = max((h["wind_kmh"] for h in hours), default=1)
    for h in hours:
        speed = h["wind_kmh"]
        gust = h["wind_gust_kmh"]
        compass = h.get("direction_compass", "")
        bar_len = round(speed / max_speed * 20) if max_speed > 0 else 0
        fishable = h.get("fishable", True)
        cond = wind_color(speed, gust)
        ansi = _WIND_ANSI[cond] if fishable else DIM
        bar = f"{ansi}{'▇' * bar_len}{RESET}"
        print(
            f"  {h['hour']}  {speed:4.1f} km/h  {compass:2s}  gusts {gust:4.1f}  {bar}"
        )
    verdict = day.get("wind_verdict", _WIND_VERDICTS["green"])
    # Find the color for the verdict
    verdict_color = "green"
    for k, v in _WIND_VERDICTS.items():
        if v == verdict:
            verdict_color = k
            break
    print(f"  {BOLD}Overall:{RESET} {_WIND_ANSI[verdict_color]}{verdict}{RESET}")

    # Technique verdicts
    verdicts = day.get("technique_verdicts", {})
    if verdicts:
        _VERDICT_ANSI = {"go": GREEN, "difficult": YELLOW, "no-go": RED}
        _VERDICT_LABEL = {"go": "Go", "difficult": "Difficult", "no-go": "No-go"}
        print(f"  {BOLD}Techniques:{RESET}")
        for tech, v in verdicts.items():
            color = _VERDICT_ANSI[v]
            label = _VERDICT_LABEL[v]
            print(f"    {color}●{RESET} {tech}: {color}{label}{RESET}")
    print()


def print_summary(
    rows: list[tuple[str, str, str, float | None, float | None, int]],
    day: dict | None = None,
) -> None:
    """Print a one-line summary aggregating levels, rain, and wind."""
    parts = []

    # Levels vs 10y average
    ratios = []
    for _, _, _, today_val, avg_val, _ in rows:
        if today_val is not None and avg_val is not None and avg_val > 0:
            ratios.append(today_val / avg_val)
    if ratios:
        pct = round((sum(ratios) / len(ratios) - 1) * 100)
        if pct > 0:
            parts.append(f"Levels +{pct}% above avg")
        elif pct < 0:
            parts.append(f"Levels {pct}% below avg")
        else:
            parts.append("Levels at avg")

    if day:
        # Rain total (fishable hours only)
        total_mm = sum(
            h["precipitation"] for h in day.get("hourly", []) if h.get("fishable", True)
        )
        if total_mm > 0:
            parts.append(f"Rain: {total_mm:.1f} mm")
        else:
            parts.append("Rain: dry")

        # Wind verdict
        verdict = day.get("wind_verdict", _WIND_VERDICTS["green"])
        verdict_color = "green"
        for k, v in _WIND_VERDICTS.items():
            if v == verdict:
                verdict_color = k
                break
        parts.append(f"{_WIND_ANSI[verdict_color]}{verdict}{RESET}")

        verdicts = day.get("technique_verdicts", {})
        if verdicts:
            nogo = [t for t, v in verdicts.items() if v == "no-go"]
            difficult = [t for t, v in verdicts.items() if v == "difficult"]
            if len(nogo) == len(verdicts):
                parts.append(f"{RED}No technique practicable{RESET}")
            elif nogo or difficult:
                avoid = nogo + difficult
                parts.append(f"{YELLOW}Avoid: {', '.join(avoid)}{RESET}")

    print()
    print(f"  {GREEN}>>{RESET} {' | '.join(parts)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="French river water height console tool"
    )
    parser.add_argument("location", nargs="?", help="Location name (e.g. Paris, Lyon)")
    parser.add_argument(
        "--station", metavar="CODE", help="Plot a specific station by code"
    )
    date_group = parser.add_mutually_exclusive_group()
    date_group.add_argument(
        "--date",
        metavar="YYYY-MM-DD",
        type=date.fromisoformat,
        default=None,
        help="Date to display data for (default: today)",
    )
    date_group.add_argument(
        "--tomorrow",
        action="store_true",
        help="Show data for tomorrow (equivalent to --date with tomorrow's date)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        dest="json_output",
        help="Output machine-readable JSON instead of human display",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        metavar="N",
        choices=range(1, 17),
        help="Number of days of weather forecast (1-16, default: 1)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.days is not None and args.station:
        print("--days is only valid with a location", file=sys.stderr)
        sys.exit(1)

    target_date = date.today() + timedelta(days=1) if args.tomorrow else args.date

    if args.station:
        station = get_station_info(args.station)
        cache = load_cache()
        data = fetch_station_data(station["code_station"], target_date, cache)
        save_cache(cache)
        if args.json_output:
            print(
                json.dumps(
                    {
                        "station": {
                            "code": station["code_station"],
                            "name": station.get("libelle_station", "?"),
                            "river": station.get("libelle_cours_eau", "?"),
                        },
                        "history": {
                            "dates": data["dates"],
                            "values": data["values"],
                            "grandeur": data["grandeur"],
                        },
                        "historical_average": {
                            "value": data["avg"],
                            "year_count": data["avg_count"],
                        },
                    },
                    ensure_ascii=False,
                )
            )
            return
        display(
            station,
            data["dates"],
            data["values"],
            data["avg"],
            data["avg_count"],
            target_date,
        )
        return

    if not args.location:
        parser.print_help()
        sys.exit(1)

    lat, lon = geocode(args.location)
    stations = search_stations_nearby(lat, lon, 25)
    if not stations:
        print(f"No stations found within 25 km of '{args.location}'.")
        sys.exit(1)

    cache = load_cache()
    rows = []
    for station in stations:
        code = station["code_station"]
        name = station.get("libelle_station") or "?"
        river = station.get("libelle_cours_eau") or "?"
        try:
            if target_date is None or target_date >= date.today():
                today_val = fetch_today_level(code)
            else:
                today_val = fetch_date_level(code, target_date)
        except (httpx.HTTPStatusError, httpx.TimeoutException):
            if not args.json_output:
                print(
                    f"  {DIM}{river} à {name} ({code}): unavailable{RESET}",
                    file=sys.stderr,
                )
            continue
        try:
            data = fetch_station_data(code, target_date, cache)
        except (httpx.HTTPStatusError, httpx.TimeoutException):
            data = {"avg": None, "avg_count": 0}
        rows.append((river, name, code, today_val, data["avg"], data["avg_count"]))

    save_cache(cache)

    if args.json_output:
        days = args.days or 1
        json_stations = []
        for river, name, code, tv, av, ac in rows:
            if not code or river == "?" or tv is None:
                continue
            pct = None
            if av is not None and av > 0:
                pct = round((tv / av - 1) * 100)
            json_stations.append(
                {
                    "river": river,
                    "name": name,
                    "code": code,
                    "level_mm": tv,
                    "avg_mm": av,
                    "avg_year_count": ac,
                    "level_vs_avg_pct": pct,
                }
            )
        print(
            json.dumps(
                {
                    "location": args.location,
                    "coordinates": {"lat": lat, "lon": lon},
                    "stations": json_stations,
                    "forecast": fetch_daily_forecast(lat, lon, days),
                },
                ensure_ascii=False,
            )
        )
        return

    # Human-readable output
    days = args.days or 1
    print(f"Found {len(stations)} station(s) near {args.location}\n")
    display_table(rows, target_date)

    is_today = target_date is None or target_date == date.today()
    weather = fetch_daily_forecast(lat, lon, days, start_date=target_date)

    for i, day in enumerate(weather):
        if days > 1:
            print(f"  {BOLD}{day['date']}{RESET}")
        print_rain_section(day, is_today and i == 0)
        print_wind_section(day, is_today and i == 0)
        print(f"  {BOLD}Sunlight:{RESET}")
        print(f"  {YELLOW}☀{RESET}  Sunrise       {day['sunrise']}")
        print(f"  {YELLOW}☀{RESET}  Sunset        {day['sunset']}")
        print(
            f"  {YELLOW}☀{RESET}  Peak sunlight {day['peak_start']} – {day['peak_end']}"
        )

    first_day = weather[0] if weather else None
    print_summary(rows, first_day)


if __name__ == "__main__":
    main()
