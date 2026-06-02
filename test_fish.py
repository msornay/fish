"""Tests for fish.py."""

import json
from unittest.mock import patch, MagicMock
from datetime import date, timedelta
from io import StringIO

import httpx
import pytest

import fish


# --- pick_height_grandeur ---


def test_pick_height_grandeur_prefers_hmn():
    obs = [{"grandeur_hydro_elab": "HmnJ"}, {"grandeur_hydro_elab": "HIXnJ"}]
    assert fish.pick_height_grandeur(obs) == "HmnJ"


def test_pick_height_grandeur_prefers_hin_over_hix():
    obs = [{"grandeur_hydro_elab": "HINnJ"}, {"grandeur_hydro_elab": "HIXnJ"}]
    assert fish.pick_height_grandeur(obs) == "HINnJ"


def test_pick_height_grandeur_falls_back_to_h_prefix():
    obs = [{"grandeur_hydro_elab": "HXYzJ"}]
    assert fish.pick_height_grandeur(obs) == "HXYzJ"


def test_pick_height_grandeur_returns_none_when_no_height():
    obs = [{"grandeur_hydro_elab": "QmnJ"}]
    assert fish.pick_height_grandeur(obs) is None


def test_pick_height_grandeur_empty():
    assert fish.pick_height_grandeur([]) is None


# --- fetch_obs_elab ---


def _mock_response(data, cursor=None):
    resp = MagicMock()
    resp.json.return_value = {"data": data, "next": cursor}
    resp.raise_for_status.return_value = None
    return resp


@patch("fish.httpx.get")
def test_fetch_obs_elab_single_page(mock_get):
    mock_get.return_value = _mock_response([{"resultat_obs_elab": 100}])
    result = fish.fetch_obs_elab("X", "2025-01-01", "2025-01-02")
    assert len(result) == 1
    assert result[0]["resultat_obs_elab"] == 100


@patch("fish.httpx.get")
def test_fetch_obs_elab_pagination(mock_get):
    cursor_url = "https://hubeau.eaufrance.fr/api/v2/hydrometrie/obs_elab?cursor=xyz789"
    mock_get.side_effect = [
        _mock_response([{"v": 1}], cursor=cursor_url),
        _mock_response([{"v": 2}]),
    ]
    result = fish.fetch_obs_elab("X", "2025-01-01", "2025-03-01")
    assert len(result) == 2
    assert mock_get.call_count == 2
    # Second call should use the cursor URL directly, not merge with params
    second_call = mock_get.call_args_list[1]
    assert second_call[0][0] == cursor_url
    assert "params" not in second_call[1]


@patch("fish.httpx.get")
def test_fetch_obs_elab_passes_grandeur(mock_get):
    mock_get.return_value = _mock_response([])
    fish.fetch_obs_elab("X", "2025-01-01", "2025-01-02", grandeur="HIXnJ")
    params = mock_get.call_args[1]["params"]
    assert params["grandeur_hydro_elab"] == "HIXnJ"


@patch("fish.httpx.get")
def test_fetch_obs_elab_omits_grandeur_when_none(mock_get):
    mock_get.return_value = _mock_response([])
    fish.fetch_obs_elab("X", "2025-01-01", "2025-01-02")
    params = mock_get.call_args[1]["params"]
    assert "grandeur_hydro_elab" not in params


# --- get_station_info ---


@patch("fish.httpx.get")
def test_get_station_info_returns_first(mock_get):
    station = {"code_station": "X1", "libelle_station": "Test"}
    resp = MagicMock()
    resp.json.return_value = {"data": [station]}
    resp.raise_for_status.return_value = None
    mock_get.return_value = resp

    assert fish.get_station_info("X1")["code_station"] == "X1"


@patch("fish.httpx.get")
def test_get_station_info_exits_when_not_found(mock_get):
    resp = MagicMock()
    resp.json.return_value = {"data": []}
    resp.raise_for_status.return_value = None
    mock_get.return_value = resp

    with pytest.raises(SystemExit):
        fish.get_station_info("NOPE")


# --- fetch_recent_3months ---


@patch("fish.fetch_obs_elab")
def test_fetch_recent_3months_sorted(mock_fetch):
    mock_fetch.return_value = [
        {
            "grandeur_hydro_elab": "HmnJ",
            "date_obs_elab": "2025-03-02",
            "resultat_obs_elab": 200,
        },
        {
            "grandeur_hydro_elab": "HmnJ",
            "date_obs_elab": "2025-03-01",
            "resultat_obs_elab": 100,
        },
    ]
    dates, values, grandeur = fish.fetch_recent_3months("X")
    assert dates == ["2025-03-01", "2025-03-02"]
    assert values == [100, 200]
    assert grandeur == "HmnJ"


@patch("fish.fetch_obs_elab")
def test_fetch_recent_3months_no_data(mock_fetch):
    mock_fetch.return_value = []
    dates, values, grandeur = fish.fetch_recent_3months("X")
    assert dates == []
    assert values == []
    assert grandeur == ""


# --- fetch_historical_average ---


@patch("fish.fetch_obs_elab")
def test_fetch_historical_average_computes(mock_fetch):
    mock_fetch.return_value = [{"resultat_obs_elab": 100}]
    avg, count = fish.fetch_historical_average("X", "HmnJ")
    assert avg == 100.0
    assert count == 10


@patch("fish.fetch_obs_elab")
def test_fetch_historical_average_no_data(mock_fetch):
    mock_fetch.return_value = []
    avg, count = fish.fetch_historical_average("X", "HmnJ")
    assert avg is None
    assert count == 0


# --- geocode ---


def _mock_geocode_response(results):
    resp = MagicMock()
    resp.json.return_value = results
    resp.raise_for_status.return_value = None
    return resp


@patch("fish.httpx.get")
def test_geocode_returns_lat_lon(mock_get):
    mock_get.return_value = _mock_geocode_response([{"lat": "48.85", "lon": "2.35"}])
    lat, lon = fish.geocode("Paris")
    assert lat == 48.85
    assert lon == 2.35


@patch("fish.httpx.get")
def test_geocode_exits_when_not_found(mock_get):
    mock_get.return_value = _mock_geocode_response([])
    with pytest.raises(SystemExit):
        fish.geocode("nonexistent")


# --- search_stations_nearby ---


@patch("fish.httpx.get")
def test_search_stations_nearby_returns_data(mock_get):
    stations = [{"code_station": "S1"}, {"code_station": "S2"}]
    mock_get.return_value = _mock_response(stations)
    result = fish.search_stations_nearby(48.85, 2.35, 25)
    assert len(result) == 2
    params = mock_get.call_args[1]["params"]
    assert params["distance"] == 25


@patch("fish.httpx.get")
def test_search_stations_nearby_paginates(mock_get):
    cursor_url = "https://hubeau.eaufrance.fr/api/v2/hydrometrie/referentiel/stations?cursor=abc123"
    mock_get.side_effect = [
        _mock_response([{"code_station": "S1"}], cursor=cursor_url),
        _mock_response([{"code_station": "S2"}]),
    ]
    result = fish.search_stations_nearby(48.85, 2.35, 25)
    assert len(result) == 2
    assert mock_get.call_count == 2
    # Second call should use the cursor URL directly, not merge with params
    second_call = mock_get.call_args_list[1]
    assert second_call[0][0] == cursor_url
    assert "params" not in second_call[1]


@patch("fish.httpx.get")
def test_search_stations_nearby_empty(mock_get):
    mock_get.return_value = _mock_response([])
    result = fish.search_stations_nearby(48.85, 2.35, 25)
    assert result == []


# --- fetch_station_data ---


@patch("fish.get_historical_average", return_value=(150.0, 5))
@patch(
    "fish.fetch_recent_3months",
    return_value=(["2025-03-01"], [100], "HmnJ"),
)
def test_fetch_station_data_returns_history_and_avg(mock_recent, mock_avg):
    cache = {"year": 2026, "data": {}}
    result = fish.fetch_station_data("X1", None, cache)
    assert result["dates"] == ["2025-03-01"]
    assert result["values"] == [100]
    assert result["grandeur"] == "HmnJ"
    assert result["avg"] == 150.0
    assert result["avg_count"] == 5
    mock_recent.assert_called_once_with("X1", None)
    mock_avg.assert_called_once_with("X1", "HmnJ", cache, None)


@patch("fish.fetch_recent_3months", return_value=([], [], ""))
def test_fetch_station_data_no_grandeur_skips_avg(mock_recent):
    cache = {"year": 2026, "data": {}}
    result = fish.fetch_station_data("X1", None, cache)
    assert result["dates"] == []
    assert result["values"] == []
    assert result["grandeur"] == ""
    assert result["avg"] is None
    assert result["avg_count"] == 0


@patch("fish.get_historical_average", return_value=(200.0, 8))
@patch(
    "fish.fetch_recent_3months",
    return_value=(["2025-06-01"], [300], "HmnJ"),
)
def test_fetch_station_data_passes_target_date(mock_recent, mock_avg):
    cache = {"year": 2026, "data": {}}
    td = date(2026, 6, 1)
    result = fish.fetch_station_data("X1", td, cache)
    mock_recent.assert_called_once_with("X1", td)
    mock_avg.assert_called_once_with("X1", "HmnJ", cache, td)
    assert result["avg"] == 200.0


# --- cache ---


def test_load_cache_missing_file(tmp_path):
    with patch.object(fish, "CACHE_PATH", tmp_path / "nope.json"):
        cache = fish.load_cache()
    assert cache["year"] == date.today().year
    assert cache["data"] == {}


def test_load_cache_wrong_year(tmp_path):
    p = tmp_path / "hist_avg.json"
    p.write_text(json.dumps({"year": 1999, "data": {"k": [1, 2]}}))
    with patch.object(fish, "CACHE_PATH", p):
        cache = fish.load_cache()
    assert cache["data"] == {}


def test_load_cache_valid(tmp_path):
    p = tmp_path / "hist_avg.json"
    data = {"year": date.today().year, "data": {"X:01-01:HmnJ": [100.0, 5]}}
    p.write_text(json.dumps(data))
    with patch.object(fish, "CACHE_PATH", p):
        cache = fish.load_cache()
    assert cache["data"]["X:01-01:HmnJ"] == [100.0, 5]


def test_save_cache_creates_dirs(tmp_path):
    p = tmp_path / "sub" / "dir" / "hist_avg.json"
    with patch.object(fish, "CACHE_PATH", p):
        fish.save_cache({"year": 2026, "data": {}})
    assert p.exists()
    assert json.loads(p.read_text())["year"] == 2026


# --- get_historical_average ---


def test_get_historical_average_cache_hit():
    today_md = date.today().strftime("%m-%d")
    cache = {"year": 2026, "data": {f"X:{today_md}:HmnJ": [200.0, 7]}}
    avg, count = fish.get_historical_average("X", "HmnJ", cache)
    assert avg == 200.0
    assert count == 7


@patch("fish.fetch_obs_elab")
@patch("fish.date")
def test_get_historical_average_cache_miss_triggers_prepopulate(mock_date, mock_fetch):
    fixed = date(2025, 3, 1)
    mock_date.today.return_value = fixed
    mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
    mock_fetch.return_value = [
        {"date_obs_elab": fixed.isoformat(), "resultat_obs_elab": 300.0}
    ]
    cache = {"year": 2025, "data": {}}
    avg, count = fish.get_historical_average("X", "HmnJ", cache)
    assert avg == 300.0
    assert count == 10  # 10 years, same value each
    assert mock_fetch.call_count == 10


@patch("fish.fetch_obs_elab")
def test_prepopulate_cache_handles_api_errors(mock_fetch):
    import httpx

    mock_fetch.side_effect = httpx.TimeoutException("timeout")
    cache = {"year": 2026, "data": {}}
    fish.prepopulate_cache("X", "HmnJ", cache)
    assert cache["data"] == {}


@patch("fish.fetch_obs_elab")
def test_prepopulate_cache_year_boundary(mock_fetch):
    """When target_date is in Oct-Dec, the 90-day window crosses into the next year.
    Ensure the historical date range doesn't invert (d_min > d_max)."""
    mock_fetch.return_value = [
        {"date_obs_elab": "2024-11-15", "resultat_obs_elab": 200.0},
        {"date_obs_elab": "2025-01-10", "resultat_obs_elab": 300.0},
    ]
    cache = {"year": 2026, "data": {}}
    # Nov 15 + 90 days = mid-Feb next year — crosses year boundary
    fish.prepopulate_cache("X", "HmnJ", cache, target_date=date(2026, 11, 15))
    assert mock_fetch.call_count == 10
    # Verify each call has d_min <= d_max
    for call in mock_fetch.call_args_list:
        d_min_str, d_max_str = call[0][1], call[0][2]
        assert d_min_str <= d_max_str, f"Inverted range: {d_min_str} > {d_max_str}"
    # Verify cache was populated with data from both sides of the year boundary
    assert "X:11-15:HmnJ" in cache["data"]
    assert "X:01-10:HmnJ" in cache["data"]


# --- display ---


@patch("fish.plt")
def test_display_header_no_duplicate_river(mock_plt):
    """Header should show station name directly, not prepend river name."""
    station = {
        "code_station": "F700000103",
        "libelle_station": "La Seine à Paris - Austerlitz [>2006]",
        "libelle_cours_eau": "La Seine",
    }
    with patch("sys.stdout", new_callable=StringIO) as out:
        fish.display(station, [], [], None, 0)
        output = out.getvalue()
    # Station name should appear once, not duplicated with river prefix
    assert "La Seine à La Seine" not in output
    assert "La Seine à Paris - Austerlitz [>2006]" in output


# --- display_table ---


def test_display_table_output():
    rows = [
        ("La Loue", "Station A", "X001", 854.0, 862.0, 10),
        ("Le Doubs", "Station B", "X002", 120.0, None, 0),
    ]
    with patch("sys.stdout", new_callable=StringIO) as out:
        fish.display_table(rows)
        output = out.getvalue()
    assert "La Loue" in output
    assert "X001" in output
    assert "854 mm" in output
    assert "862 mm" in output


def test_display_table_skips_missing_level():
    rows = [
        ("La Loue", "Station A", "X001", 854.0, 862.0, 10),
        ("Le Doubs", "Station B", "X002", None, None, 0),
    ]
    with patch("sys.stdout", new_callable=StringIO) as out:
        fish.display_table(rows)
        output = out.getvalue()
    assert "X001" in output
    assert "X002" not in output


def test_display_table_skips_missing_river():
    rows = [
        ("?", "Station A", "X001", 854.0, 862.0, 10),
    ]
    with patch("sys.stdout", new_callable=StringIO) as out:
        fish.display_table(rows)
        output = out.getvalue()
    assert "X001" not in output


def test_display_table_skips_missing_code():
    rows = [
        ("La Loue", "Station A", "", 854.0, 862.0, 10),
    ]
    with patch("sys.stdout", new_callable=StringIO) as out:
        fish.display_table(rows)
        output = out.getvalue()
    assert "Station A" not in output


def test_display_table_lines_max_79_chars():
    rows = [
        (
            "Le Doubs",
            "La Seine à Paris - Austerlitz [>2006]",
            "F700000103",
            854.0,
            862.0,
            10,
        ),
    ]
    with patch("sys.stdout", new_callable=StringIO) as out:
        fish.display_table(rows)
        output = out.getvalue()
    for line in output.splitlines():
        # Strip ANSI escape codes before measuring
        clean = line
        for code in [
            fish.BOLD,
            fish.DIM,
            fish.CYAN,
            fish.GREEN,
            fish.YELLOW,
            fish.RED,
            fish.RESET,
        ]:
            clean = clean.replace(code, "")
        assert len(clean) <= 79, f"Line is {len(clean)} chars: {clean!r}"


# --- fetch_today_level ---


@patch("fish.httpx.get")
def test_fetch_today_level_returns_value(mock_get):
    mock_get.return_value = _mock_response([{"resultat_obs": 1234.0}])
    assert fish.fetch_today_level("X1") == 1234.0


@patch("fish.httpx.get")
def test_fetch_today_level_returns_none_when_empty(mock_get):
    mock_get.return_value = _mock_response([])
    assert fish.fetch_today_level("X1") is None


@patch("fish.httpx.get")
def test_fetch_today_level_returns_none_when_no_result(mock_get):
    mock_get.return_value = _mock_response([{"resultat_obs": None}])
    assert fish.fetch_today_level("X1") is None


def _mock_tr_response(values, cursor=None):
    resp = MagicMock()
    resp.json.return_value = {
        "data": [{"resultat_obs": v} for v in values],
        "next": cursor,
    }
    resp.raise_for_status.return_value = None
    return resp


@patch("fish.httpx.get")
def test_fetch_today_level_aggregate_hixnj_returns_max(mock_get):
    mock_get.return_value = _mock_tr_response([800.0, 813.0, 805.0])
    assert fish.fetch_today_level("X1", grandeur="HIXnJ") == 813.0


@patch("fish.httpx.get")
def test_fetch_today_level_aggregate_hinnj_returns_min(mock_get):
    mock_get.return_value = _mock_tr_response([800.0, 813.0, 805.0])
    assert fish.fetch_today_level("X1", grandeur="HINnJ") == 800.0


@patch("fish.httpx.get")
def test_fetch_today_level_aggregate_hmnj_returns_mean(mock_get):
    mock_get.return_value = _mock_tr_response([800.0, 810.0, 820.0])
    assert fish.fetch_today_level("X1", grandeur="HmnJ") == 810.0


@patch("fish.httpx.get")
def test_fetch_today_level_aggregate_unknown_returns_latest(mock_get):
    mock_get.return_value = _mock_tr_response([800.0, 813.0, 805.0])
    assert fish.fetch_today_level("X1", grandeur="Hwhatever") == 805.0


@patch("fish.httpx.get")
def test_fetch_today_level_aggregate_empty_falls_back_to_instant(mock_get):
    # Aggregate window empty (e.g. early morning before any obs uploaded), then
    # the fallback instant call returns a stale-but-present latest reading.
    mock_get.side_effect = [
        _mock_tr_response([]),
        _mock_response([{"resultat_obs": 813.0}]),
    ]
    assert fish.fetch_today_level("X1", grandeur="HIXnJ") == 813.0
    assert mock_get.call_count == 2


@patch("fish.httpx.get")
def test_fetch_today_level_aggregate_paginates(mock_get):
    cursor_url = (
        "https://hubeau.eaufrance.fr/api/v2/hydrometrie/observations_tr?cursor=abc"
    )
    mock_get.side_effect = [
        _mock_tr_response([700.0, 720.0], cursor=cursor_url),
        _mock_tr_response([800.0]),
    ]
    assert fish.fetch_today_level("X1", grandeur="HIXnJ") == 800.0
    assert mock_get.call_count == 2
    second_call = mock_get.call_args_list[1]
    assert second_call[0][0] == cursor_url
    assert "params" not in second_call[1]


# --- --date argument ---


def test_date_argument_parsing():
    parser = fish.argparse.ArgumentParser()
    parser.add_argument("location", nargs="?")
    parser.add_argument("--date", type=date.fromisoformat, default=None)
    args = parser.parse_args(["Paris", "--date", "2025-06-15"])
    assert args.date == date(2025, 6, 15)


def test_date_argument_default_is_none():
    parser = fish.argparse.ArgumentParser()
    parser.add_argument("location", nargs="?")
    parser.add_argument("--date", type=date.fromisoformat, default=None)
    args = parser.parse_args(["Paris"])
    assert args.date is None


# --- fetch_date_level ---


@patch("fish.fetch_obs_elab")
def test_fetch_date_level_returns_value(mock_fetch):
    mock_fetch.return_value = [
        {"grandeur_hydro_elab": "HmnJ", "resultat_obs_elab": 500.0},
    ]
    result = fish.fetch_date_level("X1", date(2025, 6, 15))
    mock_fetch.assert_called_once_with("X1", "2025-06-15", "2025-06-15", grandeur=None)
    assert result == 500.0


@patch("fish.fetch_obs_elab")
def test_fetch_date_level_passes_grandeur(mock_fetch):
    mock_fetch.return_value = [
        {"grandeur_hydro_elab": "HIXnJ", "resultat_obs_elab": 800.0},
    ]
    result = fish.fetch_date_level("X1", date(2025, 6, 15), grandeur="HIXnJ")
    mock_fetch.assert_called_once_with(
        "X1", "2025-06-15", "2025-06-15", grandeur="HIXnJ"
    )
    assert result == 800.0


@patch("fish.fetch_obs_elab")
def test_fetch_date_level_returns_none_when_no_data(mock_fetch):
    mock_fetch.return_value = []
    result = fish.fetch_date_level("X1", date(2025, 6, 15))
    assert result is None


@patch("fish.fetch_obs_elab")
def test_fetch_date_level_returns_none_when_no_height_grandeur(mock_fetch):
    mock_fetch.return_value = [
        {"grandeur_hydro_elab": "QmnJ", "resultat_obs_elab": 100.0},
    ]
    result = fish.fetch_date_level("X1", date(2025, 6, 15))
    assert result is None


# --- fetch_recent_3months with target_date ---


@patch("fish.fetch_obs_elab")
def test_fetch_recent_3months_with_target_date(mock_fetch):
    mock_fetch.return_value = [
        {
            "grandeur_hydro_elab": "HmnJ",
            "date_obs_elab": "2025-06-15",
            "resultat_obs_elab": 300,
        },
    ]
    target = date(2025, 6, 15)
    dates, values, grandeur = fish.fetch_recent_3months("X", target)
    call_args = mock_fetch.call_args
    assert call_args[0][1] == (target - timedelta(days=90)).isoformat()
    assert call_args[0][2] == target.isoformat()


# --- display_table with target_date ---


def test_display_table_with_past_date():
    rows = [("La Loue", "Station A", "X001", 854.0, 862.0, 10)]
    with patch("sys.stdout", new_callable=StringIO) as out:
        fish.display_table(rows, date(2025, 6, 15))
        output = out.getvalue()
    assert "Jun 15" in output
    assert "Today" not in output


# --- future date handling ---


@patch("fish.fetch_obs_elab")
@patch("fish.date")
def test_fetch_recent_3months_caps_end_to_today_for_future(mock_date, mock_fetch):
    fixed_today = date(2026, 2, 9)
    mock_date.today.return_value = fixed_today
    mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
    mock_fetch.return_value = []
    future = date(2026, 8, 10)
    fish.fetch_recent_3months("X", future)
    call_args = mock_fetch.call_args[0]
    # End date should be capped to today, not the future date
    assert call_args[2] == fixed_today.isoformat()


def test_display_table_shows_today_for_future_date():
    rows = [("La Loue", "Station A", "X001", 854.0, 862.0, 10)]
    future = date.today() + timedelta(days=180)
    with patch("sys.stdout", new_callable=StringIO) as out:
        fish.display_table(rows, future)
        output = out.getvalue()
    assert "Today" in output


# --- --tomorrow argument ---


def test_tomorrow_flag_sets_target_date():
    """--tomorrow should set target_date to tomorrow's date."""
    parser = fish.build_parser()
    args = parser.parse_args(["Paris", "--tomorrow"])
    assert args.tomorrow is True
    # When --tomorrow is used, no explicit date should be provided.
    assert args.date is None


def test_tomorrow_flag_default_is_false():
    parser = fish.build_parser()
    args = parser.parse_args(["Paris"])
    assert args.tomorrow is False


def test_tomorrow_and_date_are_mutually_exclusive():
    parser = fish.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["Paris", "--tomorrow", "--date", "2025-01-01"])


def test_rain_section_empty_day():
    day = {"hourly": []}
    with patch("sys.stdout", new_callable=StringIO) as out:
        fish.print_rain_section(day, is_today=False)
        output = out.getvalue()
    assert output == ""


def test_rain_section_today_label():
    day = {
        "hourly": [
            {"hour": "14:00", "precipitation": 1.2, "fishable": True},
            {"hour": "15:00", "precipitation": 0.5, "fishable": True},
        ],
    }
    with patch("sys.stdout", new_callable=StringIO) as out:
        fish.print_rain_section(day, is_today=True)
        output = out.getvalue()
    assert "Rain forecast:" in output
    assert "1.2 mm" in output
    assert "0.5 mm" in output


def test_rain_section_dims_non_fishable_hours():
    day = {
        "hourly": [
            {"hour": "05:00", "precipitation": 1.0, "fishable": False},
            {"hour": "10:00", "precipitation": 2.0, "fishable": True},
        ],
    }
    with patch("sys.stdout", new_callable=StringIO) as out:
        fish.print_rain_section(day, is_today=True)
        output = out.getvalue()
    assert fish.DIM in output
    assert fish.CYAN in output


def test_rain_section_past_date_label():
    day = {
        "hourly": [
            {"hour": "10:00", "precipitation": 2.0, "fishable": True},
        ],
    }
    with patch("sys.stdout", new_callable=StringIO) as out:
        fish.print_rain_section(day, is_today=False)
        output = out.getvalue()
    assert "Rain:" in output
    assert "Rain forecast" not in output
    assert "2.0 mm" in output


def test_rain_section_renders_pressure_inline():
    day = {
        "hourly": [
            {
                "hour": "10:00",
                "precipitation": 0.0,
                "fishable": True,
                "pressure_hpa": 1015.4,
                "pressure_trend": "rising",
            },
            {
                "hour": "11:00",
                "precipitation": 0.2,
                "fishable": True,
                "pressure_hpa": 1014.1,
                "pressure_trend": "falling",
            },
        ],
    }
    with patch("sys.stdout", new_callable=StringIO) as out:
        fish.print_rain_section(day, is_today=True)
        output = out.getvalue()
    assert "1015.4 hPa" in output
    assert "↑" in output
    assert "1014.1 hPa" in output
    assert "↓" in output


def test_rain_section_handles_missing_pressure():
    day = {
        "hourly": [
            {
                "hour": "10:00",
                "precipitation": 0.0,
                "fishable": True,
                "pressure_hpa": None,
                "pressure_trend": None,
            },
        ],
    }
    with patch("sys.stdout", new_callable=StringIO) as out:
        fish.print_rain_section(day, is_today=False)
        output = out.getvalue()
    assert "hPa" not in output
    assert "0.0 mm" in output


# --- annotate_pressure_trend ---


def test_annotate_pressure_trend_rising():
    hours = [{"pressure_hpa": 1000.0 + i} for i in range(6)]
    fish.annotate_pressure_trend(hours)
    assert hours[0]["pressure_trend"] is None
    assert hours[1]["pressure_trend"] is None
    assert hours[2]["pressure_trend"] is None
    assert hours[3]["pressure_trend"] == "rising"
    assert hours[5]["pressure_trend"] == "rising"


def test_annotate_pressure_trend_falling():
    hours = [{"pressure_hpa": 1020.0 - i * 2} for i in range(6)]
    fish.annotate_pressure_trend(hours)
    assert hours[3]["pressure_trend"] == "falling"
    assert hours[5]["pressure_trend"] == "falling"


def test_annotate_pressure_trend_stable():
    hours = [{"pressure_hpa": 1013.0 + (i % 2) * 0.3} for i in range(6)]
    fish.annotate_pressure_trend(hours)
    assert hours[3]["pressure_trend"] == "stable"
    assert hours[5]["pressure_trend"] == "stable"


def test_annotate_pressure_trend_handles_none():
    hours = [{"pressure_hpa": None} for _ in range(6)]
    fish.annotate_pressure_trend(hours)
    assert all(h["pressure_trend"] is None for h in hours)


# --- degrees_to_compass ---


def test_degrees_to_compass_north():
    assert fish.degrees_to_compass(0) == "N"
    assert fish.degrees_to_compass(360) == "N"


def test_degrees_to_compass_cardinal():
    assert fish.degrees_to_compass(90) == "E"
    assert fish.degrees_to_compass(180) == "S"
    assert fish.degrees_to_compass(270) == "W"


def test_degrees_to_compass_intercardinal():
    assert fish.degrees_to_compass(45) == "NE"
    assert fish.degrees_to_compass(135) == "SE"
    assert fish.degrees_to_compass(225) == "SW"
    assert fish.degrees_to_compass(315) == "NW"


def test_degrees_to_compass_boundary():
    assert fish.degrees_to_compass(22) == "N"
    assert fish.degrees_to_compass(23) == "NE"


# --- wind_color ---


def test_wind_color_green():
    assert fish.wind_color(10, 15) == "green"


def test_wind_color_yellow_by_speed():
    assert fish.wind_color(20, 15) == "yellow"


def test_wind_color_yellow_by_gust():
    assert fish.wind_color(10, 25) == "yellow"


def test_wind_color_red_by_speed():
    assert fish.wind_color(30, 15) == "red"


def test_wind_color_red_by_gust():
    assert fish.wind_color(10, 35) == "red"


def test_wind_color_boundary_green():
    assert fish.wind_color(14.9, 19.9) == "green"


def test_wind_color_boundary_yellow():
    assert fish.wind_color(15, 20) == "yellow"


def test_wind_color_boundary_red():
    assert fish.wind_color(25.1, 20) == "red"


# --- is_fishable_hour ---


def test_is_fishable_hour_within_window():
    assert fish.is_fishable_hour("10:00", "07:00", "20:00") is True


def test_is_fishable_hour_before_sunrise():
    assert fish.is_fishable_hour("05:00", "07:00", "20:00") is False


def test_is_fishable_hour_after_sunset():
    assert fish.is_fishable_hour("22:00", "07:00", "20:00") is False


def test_is_fishable_hour_within_30min_before_sunrise():
    assert fish.is_fishable_hour("06:30", "07:00", "20:00") is True


def test_is_fishable_hour_within_30min_after_sunset():
    assert fish.is_fishable_hour("20:30", "07:00", "20:00") is True


def test_is_fishable_hour_just_outside_margin():
    assert fish.is_fishable_hour("06:29", "07:00", "20:00") is False
    assert fish.is_fishable_hour("20:31", "07:00", "20:00") is False


# --- technique_verdicts ---


def _make_hours(wind=5.0, gust=8.0, precip=0.0, fishable=True, weathercode=1):
    """Build a single-hour list for technique_verdicts tests."""
    return [
        {
            "hour": "10:00",
            "wind_kmh": wind,
            "wind_gust_kmh": gust,
            "precipitation": precip,
            "fishable": fishable,
            "weathercode": weathercode,
        }
    ]


def test_technique_verdicts_all_go():
    result = fish.technique_verdicts(_make_hours(5, 8, 0.0))
    assert set(result.values()) == {"go"}
    assert len(result) == len(fish.TECHNIQUES)


def test_technique_verdicts_thunderstorm_during_fishable_hours():
    result = fish.technique_verdicts(_make_hours(5, 8, 0.0, weathercode=95))
    assert all(v == "no-go" for v in result.values())


def test_technique_verdicts_thunderstorm_outside_fishable_hours():
    # A storm overnight (non-fishable) must not blanket the fishable window.
    hours = _make_hours(5, 8, 0.0, fishable=False, weathercode=95) + _make_hours(
        5, 8, 0.0, fishable=True, weathercode=1
    )
    result = fish.technique_verdicts(hours)
    assert all(v == "go" for v in result.values())


def test_technique_verdicts_hard_stop_gust():
    result = fish.technique_verdicts(_make_hours(40, 75, 0.0))
    assert all(v == "no-go" for v in result.values())


def test_technique_verdicts_hard_stop_precip():
    result = fish.technique_verdicts(_make_hours(5, 8, 9.0))
    assert all(v == "no-go" for v in result.values())


def test_technique_verdicts_precip_difficult():
    result = fish.technique_verdicts(_make_hours(5, 8, 5.0))
    assert all(v in ("difficult", "no-go") for v in result.values())
    assert result["Toc"] == "difficult"


def test_technique_verdicts_precip_below_cap():
    result = fish.technique_verdicts(_make_hours(5, 8, 3.9))
    assert all(v == "go" for v in result.values())


def test_technique_verdicts_mouche_seche_nogo():
    result = fish.technique_verdicts(_make_hours(26, 20, 0.0))
    assert result["Mouche sèche"] == "no-go"
    assert result["Mouche nymphe"] == "difficult"
    assert result["Toc"] == "difficult"
    assert result["Silure au posé"] == "go"


def test_technique_verdicts_non_fishable_ignored():
    result = fish.technique_verdicts(_make_hours(50, 80, 20.0, fishable=False))
    assert all(v == "go" for v in result.values())


def test_technique_verdicts_graduated():
    hours = [
        {
            "hour": "10:00",
            "wind_kmh": 32,
            "wind_gust_kmh": 48,
            "precipitation": 0.0,
            "fishable": True,
            "weathercode": 1,
        }
    ]
    result = fish.technique_verdicts(hours)
    assert result["Mouche sèche"] == "no-go"
    assert result["Mouche nymphe"] == "no-go"
    assert result["Lancer UL"] == "no-go"
    assert result["Leurre 7g+"] == "difficult"
    assert result["Toc"] == "difficult"
    assert result["Silure au posé"] == "go"


def test_technique_verdicts_silure_resilient():
    result = fish.technique_verdicts(_make_hours(45, 55, 0.0))
    assert result["Silure au posé"] == "difficult"
    assert result["Silure au posé"] != "no-go"


# --- print_wind_section ---


def _make_wind_day(hours, verdict="Great for casting", technique_verdicts=None):
    """Helper: build a forecast day dict for wind section tests."""
    return {
        "wind_verdict": verdict,
        "technique_verdicts": technique_verdicts or {},
        "hourly": [
            {
                "hour": h[0],
                "wind_kmh": h[1],
                "wind_gust_kmh": h[2],
                "direction_compass": h[3],
                "fishable": h[4] if len(h) > 4 else True,
                "precipitation": 0.0,
            }
            for h in hours
        ],
    }


def test_wind_section_today_label():
    day = _make_wind_day([("14:00", 10.0, 15.0, "NW")])
    with patch("sys.stdout", new_callable=StringIO) as out:
        fish.print_wind_section(day, is_today=True)
        output = out.getvalue()
    assert "Wind forecast:" in output


def test_wind_section_past_label():
    day = _make_wind_day([("14:00", 10.0, 15.0, "NW")])
    with patch("sys.stdout", new_callable=StringIO) as out:
        fish.print_wind_section(day, is_today=False)
        output = out.getvalue()
    assert "Wind:" in output
    assert "Wind forecast" not in output


def test_wind_section_empty_day():
    day = {"hourly": [], "wind_verdict": "Great for casting"}
    with patch("sys.stdout", new_callable=StringIO) as out:
        fish.print_wind_section(day, is_today=False)
        output = out.getvalue()
    assert output == ""


def test_wind_section_shows_compass():
    day = _make_wind_day([("10:00", 10.0, 15.0, "NW")])
    with patch("sys.stdout", new_callable=StringIO) as out:
        fish.print_wind_section(day, is_today=True)
        output = out.getvalue()
    assert "NW" in output


def test_wind_section_shows_speed_and_gust():
    day = _make_wind_day([("10:00", 12.5, 18.0, "E")])
    with patch("sys.stdout", new_callable=StringIO) as out:
        fish.print_wind_section(day, is_today=True)
        output = out.getvalue()
    assert "12.5" in output
    assert "18.0" in output


def test_wind_section_shows_technique_verdicts():
    verdicts = {t: "go" for t in fish.TECHNIQUES}
    verdicts["Mouche sèche"] = "no-go"
    verdicts["Lancer UL"] = "difficult"
    day = _make_wind_day(
        [("10:00", 30.0, 45.0, "W")],
        technique_verdicts=verdicts,
    )
    with patch("sys.stdout", new_callable=StringIO) as out:
        fish.print_wind_section(day, is_today=True)
        output = out.getvalue()
    assert "Techniques:" in output
    assert "Mouche" in output
    assert "No-go" in output
    assert "Difficult" in output
    assert "Go" in output


def test_wind_section_dims_non_fishable():
    day = _make_wind_day(
        [
            ("05:00", 30.0, 40.0, "W", False),
            ("10:00", 8.0, 12.0, "N", True),
        ]
    )
    with patch("sys.stdout", new_callable=StringIO) as out:
        fish.print_wind_section(day, is_today=True)
        output = out.getvalue()
    assert fish.DIM in output


# --- print_day_verdicts ---


def test_day_verdicts_all_go():
    verdicts = {t: "go" for t in fish.TECHNIQUES}
    day = {"technique_verdicts": verdicts}
    with patch("sys.stdout", new_callable=StringIO) as out:
        fish.print_day_verdicts(day)
        output = out.getvalue()
    for t in fish.TECHNIQUES:
        assert t in output
    assert "Go" in output


def test_day_verdicts_mixed():
    verdicts = {t: "go" for t in fish.TECHNIQUES}
    verdicts["Mouche sèche"] = "no-go"
    verdicts["Lancer UL"] = "difficult"
    day = {"technique_verdicts": verdicts}
    with patch("sys.stdout", new_callable=StringIO) as out:
        fish.print_day_verdicts(day)
        output = out.getvalue()
    assert "Mouche sèche: No-go" in output
    assert "Lancer UL: Difficult" in output


def test_day_verdicts_with_label():
    verdicts = {t: "go" for t in fish.TECHNIQUES}
    day = {"technique_verdicts": verdicts, "hourly": []}
    with patch("sys.stdout", new_callable=StringIO) as out:
        fish.print_day_verdicts(day, label="2026-03-29")
        output = out.getvalue()
    assert "2026-03-29" in output


def test_day_verdicts_shows_levels():
    verdicts = {t: "go" for t in fish.TECHNIQUES}
    day = {"technique_verdicts": verdicts, "hourly": []}
    rows = [("La Loue", "Station A", "X001", 900.0, 800.0, 10)]
    with patch("sys.stdout", new_callable=StringIO) as out:
        fish.print_day_verdicts(day, rows=rows)
        output = out.getvalue()
    assert "+12%" in output


def test_day_verdicts_shows_rain():
    verdicts = {t: "go" for t in fish.TECHNIQUES}
    day = {
        "technique_verdicts": verdicts,
        "hourly": [
            {"hour": "10:00", "precipitation": 1.5, "fishable": True},
            {"hour": "11:00", "precipitation": 2.0, "fishable": True},
        ],
    }
    with patch("sys.stdout", new_callable=StringIO) as out:
        fish.print_day_verdicts(day)
        output = out.getvalue()
    assert "3.5" in output


def test_day_verdicts_dry():
    verdicts = {t: "go" for t in fish.TECHNIQUES}
    day = {
        "technique_verdicts": verdicts,
        "hourly": [
            {"hour": "10:00", "precipitation": 0.0, "fishable": True},
        ],
    }
    with patch("sys.stdout", new_callable=StringIO) as out:
        fish.print_day_verdicts(day)
        output = out.getvalue()
    assert "dry" in output.lower()


def test_day_verdicts_levels_below():
    verdicts = {t: "go" for t in fish.TECHNIQUES}
    day = {"technique_verdicts": verdicts, "hourly": []}
    rows = [("R", "S", "X", 800.0, 1000.0, 5)]
    with patch("sys.stdout", new_callable=StringIO) as out:
        fish.print_day_verdicts(day, rows=rows)
        output = out.getvalue()
    assert "-20%" in output


# --- --json flag ---


def test_json_flag_default_false():
    parser = fish.build_parser()
    args = parser.parse_args(["Paris"])
    assert args.json_output is False


def test_json_flag_set():
    parser = fish.build_parser()
    args = parser.parse_args(["Paris", "--json"])
    assert args.json_output is True


# --- Mode B: --station --json ---


@patch("fish.save_cache")
@patch("fish.load_cache")
@patch("fish.fetch_station_data")
@patch("fish.get_station_info")
def test_station_json_mode(mock_info, mock_data, mock_cache, mock_save):
    mock_info.return_value = {
        "code_station": "X1",
        "libelle_station": "TestSta",
        "libelle_cours_eau": "TestRiv",
    }
    mock_data.return_value = {
        "dates": ["2026-01-01", "2026-01-02"],
        "values": [100.0, 110.0],
        "grandeur": "HmnJ",
        "avg": 105.0,
        "avg_count": 8,
    }
    mock_cache.return_value = {"year": 2026, "data": {}}
    with patch("sys.argv", ["fish", "--station", "X1", "--json"]):
        with patch("sys.stdout", new_callable=StringIO) as out:
            fish.main()
    result = json.loads(out.getvalue())
    assert result["station"]["code"] == "X1"
    assert result["station"]["name"] == "TestSta"
    assert result["station"]["river"] == "TestRiv"
    assert result["history"]["dates"] == ["2026-01-01", "2026-01-02"]
    assert result["history"]["values"] == [100.0, 110.0]
    assert result["history"]["grandeur"] == "HmnJ"
    assert result["historical_average"]["value"] == 105.0
    assert result["historical_average"]["year_count"] == 8


@patch("fish.save_cache")
@patch("fish.load_cache")
@patch("fish.fetch_station_data")
@patch("fish.get_station_info")
def test_station_json_no_data(mock_info, mock_data, mock_cache, mock_save):
    mock_info.return_value = {
        "code_station": "X1",
        "libelle_station": "TestSta",
        "libelle_cours_eau": "TestRiv",
    }
    mock_data.return_value = {
        "dates": [],
        "values": [],
        "grandeur": "",
        "avg": None,
        "avg_count": 0,
    }
    mock_cache.return_value = {"year": 2026, "data": {}}
    with patch("sys.argv", ["fish", "--station", "X1", "--json"]):
        with patch("sys.stdout", new_callable=StringIO) as out:
            fish.main()
    result = json.loads(out.getvalue())
    assert result["history"]["dates"] == []
    assert result["historical_average"]["value"] is None
    assert result["historical_average"]["year_count"] == 0


# --- Mode C: location --json ---


@patch("fish.fetch_daily_forecast")
@patch("fish.save_cache")
@patch("fish.load_cache")
@patch("fish.fetch_station_data")
@patch("fish.fetch_today_level")
@patch("fish.search_stations_nearby")
@patch("fish.geocode")
def test_location_json_full(
    mock_geo,
    mock_nearby,
    mock_today,
    mock_data,
    mock_cache,
    mock_save,
    mock_daily,
):
    mock_geo.return_value = (45.75, 4.85)
    mock_nearby.return_value = [
        {
            "code_station": "V1",
            "libelle_station": "Lyon",
            "libelle_cours_eau": "Le Rhône",
        },
    ]
    mock_today.return_value = 1100.0
    mock_data.return_value = {
        "dates": ["2026-01-01"],
        "values": [1100.0],
        "grandeur": "HmnJ",
        "avg": 1000.0,
        "avg_count": 10,
    }
    mock_cache.return_value = {"year": 2026, "data": {}}
    mock_daily.return_value = [
        {
            "date": "2026-03-29",
            "temp_max": 15.0,
            "temp_min": 5.0,
            "precipitation_sum": 0.0,
            "wind_max_kmh": 12.0,
            "wind_direction_dominant": "S",
            "weathercode": 1,
            "sunrise": "07:00",
            "sunset": "19:30",
            "wind_verdict": "Great for casting",
            "hourly": [
                {
                    "hour": "10:00",
                    "temp": 10.0,
                    "precipitation": 0.0,
                    "wind_kmh": 8.0,
                    "wind_gust_kmh": 12.0,
                    "cloudcover": 30,
                    "fishable": True,
                }
            ],
        },
    ]
    with patch("sys.argv", ["fish", "Lyon", "--json"]):
        with patch("sys.stdout", new_callable=StringIO) as out:
            fish.main()
    result = json.loads(out.getvalue())
    assert result["location"] == "Lyon"
    assert result["coordinates"] == {"lat": 45.75, "lon": 4.85}
    st = result["stations"][0]
    assert st["code"] == "V1"
    assert st["level_mm"] == 1100.0
    assert st["avg_mm"] == 1000.0
    assert st["level_vs_avg_pct"] == 10
    assert len(result["forecast"]) == 1
    day0 = result["forecast"][0]
    assert day0["temp_max"] == 15.0
    assert day0["wind_verdict"] == "Great for casting"
    assert day0["hourly"][0]["fishable"] is True
    # Default days=1 when --days not specified
    mock_daily.assert_called_once_with(45.75, 4.85, 1, start_date=None)


@patch("fish.fetch_daily_forecast")
@patch("fish.save_cache")
@patch("fish.load_cache")
@patch("fish.fetch_station_data")
@patch("fish.fetch_today_level")
@patch("fish.search_stations_nearby")
@patch("fish.geocode")
def test_location_json_filters_null_stations(
    mock_geo,
    mock_nearby,
    mock_today,
    mock_data,
    mock_cache,
    mock_save,
    mock_daily,
):
    mock_geo.return_value = (45.0, 3.0)
    mock_nearby.return_value = [
        {"code_station": "X1", "libelle_station": "Good", "libelle_cours_eau": "River"},
        {
            "code_station": "X2",
            "libelle_station": "NoLevel",
            "libelle_cours_eau": "River",
        },
        {"code_station": "X3", "libelle_station": "NoRiver", "libelle_cours_eau": "?"},
    ]
    mock_today.side_effect = [500.0, None, 300.0]
    mock_data.return_value = {
        "dates": [],
        "values": [],
        "grandeur": "",
        "avg": None,
        "avg_count": 0,
    }
    mock_cache.return_value = {"year": 2026, "data": {}}
    mock_daily.return_value = []
    with patch("sys.argv", ["fish", "Test", "--json"]):
        with patch("sys.stdout", new_callable=StringIO) as out:
            fish.main()
    result = json.loads(out.getvalue())
    codes = [s["code"] for s in result["stations"]]
    assert "X1" in codes
    assert "X2" not in codes  # null level filtered
    assert "X3" not in codes  # river "?" filtered


def _has_ansi(obj):
    """Recursively check for ANSI escape codes in a parsed JSON object."""
    if isinstance(obj, str):
        return "\033" in obj
    if isinstance(obj, dict):
        return any(_has_ansi(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_ansi(v) for v in obj)
    return False


@patch("fish.fetch_daily_forecast")
@patch("fish.save_cache")
@patch("fish.load_cache")
@patch("fish.fetch_station_data")
@patch("fish.fetch_today_level")
@patch("fish.search_stations_nearby")
@patch("fish.geocode")
def test_json_output_no_ansi(
    mock_geo,
    mock_nearby,
    mock_today,
    mock_data,
    mock_cache,
    mock_save,
    mock_daily,
):
    mock_geo.return_value = (45.0, 3.0)
    mock_nearby.return_value = [
        {"code_station": "X1", "libelle_station": "S", "libelle_cours_eau": "R"},
    ]
    mock_today.return_value = 500.0
    mock_data.return_value = {
        "dates": ["2026-01-01"],
        "values": [500.0],
        "grandeur": "HmnJ",
        "avg": 480.0,
        "avg_count": 5,
    }
    mock_cache.return_value = {"year": 2026, "data": {}}
    mock_daily.return_value = [
        {
            "date": "2026-03-29",
            "temp_max": 15.0,
            "temp_min": 5.0,
            "precipitation_sum": 2.0,
            "wind_max_kmh": 30.0,
            "wind_direction_dominant": "E",
            "weathercode": 61,
            "sunrise": "07:00",
            "sunset": "20:00",
            "wind_verdict": "Too windy for fly fishing",
            "hourly": [],
        },
    ]
    with patch("sys.argv", ["fish", "Test", "--json"]):
        with patch("sys.stdout", new_callable=StringIO) as out:
            fish.main()
    result = json.loads(out.getvalue())
    assert not _has_ansi(result)


# --- --days flag ---


def test_days_flag_default_none():
    parser = fish.build_parser()
    args = parser.parse_args(["Paris", "--json"])
    assert args.days is None


def test_days_flag_set():
    parser = fish.build_parser()
    args = parser.parse_args(["Paris", "--json", "--days", "7"])
    assert args.days == 7


def test_days_rejects_zero():
    parser = fish.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["Paris", "--json", "--days", "0"])


def test_days_rejects_negative():
    parser = fish.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["Paris", "--json", "--days", "-1"])


def test_days_rejects_over_sixteen():
    parser = fish.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["Paris", "--json", "--days", "17"])


def test_days_with_station_errors(capsys):
    with patch("sys.argv", ["fish", "--station", "X1", "--json", "--days", "3"]):
        with pytest.raises(SystemExit):
            fish.main()
    assert "only valid" in capsys.readouterr().err


@patch("fish.print_day_verdicts")
@patch("fish.fetch_daily_forecast")
@patch("fish.save_cache")
@patch("fish.load_cache")
@patch("fish.fetch_station_data")
@patch("fish.fetch_today_level")
@patch("fish.search_stations_nearby")
@patch("fish.geocode")
def test_days_human_mode_fetches_multiple_days(
    mock_geo,
    mock_nearby,
    mock_today,
    mock_data,
    mock_cache,
    mock_save,
    mock_daily,
    mock_verdicts,
):
    mock_geo.return_value = (45.0, 3.0)
    mock_nearby.return_value = [
        {"code_station": "X1", "libelle_station": "S", "libelle_cours_eau": "R"},
    ]
    mock_today.return_value = 500.0
    mock_data.return_value = {
        "dates": [],
        "values": [],
        "grandeur": "",
        "avg": None,
        "avg_count": 0,
    }
    mock_cache.return_value = {"year": 2026, "data": {}}
    day1 = {
        "date": "2026-03-29",
        "sunrise": "07:00",
        "sunset": "19:30",
        "peak_start": "10:15",
        "peak_end": "14:15",
        "wind_verdict": "Great for casting",
        "hourly": [
            {
                "hour": "10:00",
                "precipitation": 0.0,
                "wind_kmh": 8.0,
                "wind_gust_kmh": 12.0,
                "direction_compass": "S",
                "fishable": True,
            },
        ],
    }
    day2 = {
        "date": "2026-03-30",
        "sunrise": "06:58",
        "sunset": "19:32",
        "peak_start": "10:15",
        "peak_end": "14:15",
        "wind_verdict": "Challenging conditions",
        "hourly": [
            {
                "hour": "10:00",
                "precipitation": 2.0,
                "wind_kmh": 18.0,
                "wind_gust_kmh": 25.0,
                "direction_compass": "W",
                "fishable": True,
            },
        ],
    }
    mock_daily.return_value = [day1, day2]
    with patch("sys.argv", ["fish", "Test", "--days", "2"]):
        with patch("sys.stdout", new_callable=StringIO) as out:
            fish.main()
    output = out.getvalue()
    # Should render weather for both days
    assert "2026-03-29" in output
    assert "2026-03-30" in output
    mock_daily.assert_called_once_with(45.0, 3.0, 2, start_date=None)


@patch("fish.fetch_daily_forecast")
@patch("fish.save_cache")
@patch("fish.load_cache")
@patch("fish.fetch_station_data")
@patch("fish.fetch_today_level")
@patch("fish.search_stations_nearby")
@patch("fish.geocode")
def test_days_prints_verdicts_per_day(
    mock_geo,
    mock_nearby,
    mock_today,
    mock_data,
    mock_cache,
    mock_save,
    mock_daily,
):
    mock_geo.return_value = (45.0, 3.0)
    mock_nearby.return_value = [
        {"code_station": "X1", "libelle_station": "S", "libelle_cours_eau": "R"},
    ]
    mock_today.return_value = 900.0
    mock_data.return_value = {
        "dates": [],
        "values": [],
        "grandeur": "",
        "avg": 800.0,
        "avg_count": 10,
    }
    mock_cache.return_value = {"year": 2026, "data": {}}
    verdicts1 = {t: "go" for t in fish.TECHNIQUES}
    verdicts2 = {t: "no-go" for t in fish.TECHNIQUES}
    day1 = {
        "date": "2026-03-29",
        "sunrise": "07:00",
        "sunset": "19:30",
        "peak_start": "10:15",
        "peak_end": "14:15",
        "wind_verdict": "Great for casting",
        "technique_verdicts": verdicts1,
        "hourly": [
            {
                "hour": "10:00",
                "precipitation": 0.0,
                "wind_kmh": 8.0,
                "wind_gust_kmh": 12.0,
                "direction_compass": "S",
                "fishable": True,
            },
        ],
    }
    day2 = {
        "date": "2026-03-30",
        "sunrise": "06:58",
        "sunset": "19:32",
        "peak_start": "10:15",
        "peak_end": "14:15",
        "wind_verdict": "Too windy for fly fishing",
        "technique_verdicts": verdicts2,
        "hourly": [
            {
                "hour": "10:00",
                "precipitation": 2.5,
                "wind_kmh": 18.0,
                "wind_gust_kmh": 25.0,
                "direction_compass": "W",
                "fishable": True,
            },
        ],
    }
    mock_daily.return_value = [day1, day2]
    with patch("sys.argv", ["fish", "Test", "--days", "2"]):
        with patch("sys.stdout", new_callable=StringIO) as out:
            fish.main()
    output = out.getvalue()
    assert "2026-03-29" in output
    assert "2026-03-30" in output
    assert "Go" in output
    assert "No-go" in output


# --- fetch_daily_forecast ---


@patch("fish.httpx.get")
def test_fetch_daily_forecast_parses(mock_get):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "daily": {
            "time": ["2026-03-29", "2026-03-30"],
            "temperature_2m_max": [15.0, 17.0],
            "temperature_2m_min": [5.0, 6.0],
            "precipitation_sum": [0.0, 3.2],
            "windspeed_10m_max": [12.0, 25.0],
            "winddirection_10m_dominant": [180, 270],
            "weathercode": [1, 61],
            "sunrise": ["2026-03-29T07:00", "2026-03-30T06:58"],
            "sunset": ["2026-03-29T19:30", "2026-03-30T19:32"],
        },
        "hourly": {
            "time": [
                "2026-03-29T08:00",
                "2026-03-29T09:00",
                "2026-03-30T08:00",
                "2026-03-30T09:00",
            ],
            "temperature_2m": [8.0, 10.0, 9.0, 11.0],
            "precipitation": [0.0, 0.0, 1.0, 2.2],
            "windspeed_10m": [8.0, 10.0, 20.0, 22.0],
            "wind_gusts_10m": [12.0, 15.0, 30.0, 35.0],
            "wind_direction_10m": [180, 200, 270, 290],
            "cloudcover": [30, 40, 80, 90],
            "pressure_msl": [1013.2, 1013.5, 1010.0, 1009.4],
        },
    }
    mock_get.return_value = resp
    result = fish.fetch_daily_forecast(45.0, 3.0, 2)
    assert len(result) == 2
    day0 = result[0]
    assert day0["date"] == "2026-03-29"
    assert day0["temp_max"] == 15.0
    assert day0["temp_min"] == 5.0
    assert day0["precipitation_sum"] == 0.0
    assert day0["wind_max_kmh"] == 12.0
    assert day0["wind_direction_dominant"] == "S"
    assert day0["weathercode"] == 1
    assert day0["sunrise"] == "07:00"
    assert day0["sunset"] == "19:30"
    assert "peak_start" in day0
    assert "peak_end" in day0
    assert len(day0["hourly"]) == 2
    h0 = day0["hourly"][0]
    assert h0["hour"] == "08:00"
    assert h0["temp"] == 8.0
    assert h0["precipitation"] == 0.0
    assert h0["wind_kmh"] == 8.0
    assert h0["wind_gust_kmh"] == 12.0
    assert h0["direction_deg"] == 180
    assert h0["direction_compass"] == "S"
    assert h0["cloudcover"] == 30
    assert h0["pressure_hpa"] == 1013.2
    assert "pressure_trend" in h0
    assert "fishable" in h0
    assert "wind_verdict" in day0
    # Technique verdicts present and correct for calm day0
    assert "technique_verdicts" in day0
    assert all(v == "go" for v in day0["technique_verdicts"].values())
    # Day1 has wind 20-22, gusts 30-35: mouche seche should be difficult
    # (wind < 25 no-go threshold, gust 35 >= 25 difficult threshold)
    day1 = result[1]
    assert day1["technique_verdicts"]["Mouche sèche"] == "difficult"
    # Thresholds metadata present
    assert "thresholds" in day0
    assert day0["thresholds"]["wind_unit"] == "km/h"
    assert day0["thresholds"]["precip_unit"] == "mm/h"
    assert "techniques" in day0["thresholds"]


@patch("fish.httpx.get")
def test_fetch_daily_forecast_storm_outside_fishable_not_nogo(mock_get):
    # Daily weathercode is a daily-max aggregate: a 03:00 storm stamps 95 on
    # the whole day. The fishable daytime hours here are calm and clear, so the
    # day must NOT be blanket no-go. Verdicts must come from hourly weathercode.
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "daily": {
            "time": ["2026-06-04"],
            "temperature_2m_max": [20.0],
            "temperature_2m_min": [10.0],
            "precipitation_sum": [2.0],
            "windspeed_10m_max": [12.0],
            "winddirection_10m_dominant": [180],
            "weathercode": [95],
            "sunrise": ["2026-06-04T06:00"],
            "sunset": ["2026-06-04T21:00"],
        },
        "hourly": {
            "time": ["2026-06-04T03:00", "2026-06-04T12:00"],
            "temperature_2m": [11.0, 19.0],
            "precipitation": [1.0, 0.0],
            "windspeed_10m": [6.0, 8.0],
            "wind_gusts_10m": [12.0, 14.0],
            "wind_direction_10m": [180, 200],
            "cloudcover": [90, 20],
            "pressure_msl": [1010.0, 1012.0],
            "weathercode": [95, 1],
        },
    }
    mock_get.return_value = resp
    result = fish.fetch_daily_forecast(45.0, 3.0, 1)
    verdicts = result[0]["technique_verdicts"]
    # 03:00 storm is outside the fishable window; midday is calm and clear.
    assert not all(v == "no-go" for v in verdicts.values())


@patch("fish.httpx.get")
def test_fetch_daily_forecast_storm_during_fishable_is_nogo(mock_get):
    # A storm during the fishable window must still blanket the day no-go.
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "daily": {
            "time": ["2026-06-04"],
            "temperature_2m_max": [20.0],
            "temperature_2m_min": [10.0],
            "precipitation_sum": [2.0],
            "windspeed_10m_max": [12.0],
            "winddirection_10m_dominant": [180],
            "weathercode": [95],
            "sunrise": ["2026-06-04T06:00"],
            "sunset": ["2026-06-04T21:00"],
        },
        "hourly": {
            "time": ["2026-06-04T03:00", "2026-06-04T12:00"],
            "temperature_2m": [11.0, 19.0],
            "precipitation": [0.0, 1.0],
            "windspeed_10m": [6.0, 8.0],
            "wind_gusts_10m": [12.0, 14.0],
            "wind_direction_10m": [180, 200],
            "cloudcover": [20, 90],
            "pressure_msl": [1012.0, 1010.0],
            "weathercode": [1, 95],
        },
    }
    mock_get.return_value = resp
    result = fish.fetch_daily_forecast(45.0, 3.0, 1)
    verdicts = result[0]["technique_verdicts"]
    assert all(v == "no-go" for v in verdicts.values())


@patch("fish.httpx.get")
def test_fetch_daily_forecast_future_uses_start_date(mock_get):
    # A future target date must be honored: the forecast API (not archive) is
    # queried with start_date/end_date, not forecast_days from today.
    future = date.today() + timedelta(days=3)
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "daily": {
            "time": [future.isoformat()],
            "temperature_2m_max": [20.0],
            "temperature_2m_min": [10.0],
            "precipitation_sum": [0.0],
            "windspeed_10m_max": [10.0],
            "winddirection_10m_dominant": [180],
            "weathercode": [1],
            "sunrise": [f"{future.isoformat()}T06:00"],
            "sunset": [f"{future.isoformat()}T21:00"],
        },
        "hourly": {
            "time": [f"{future.isoformat()}T12:00"],
            "temperature_2m": [18.0],
            "precipitation": [0.0],
            "windspeed_10m": [8.0],
            "wind_gusts_10m": [12.0],
            "wind_direction_10m": [180],
            "cloudcover": [20],
            "pressure_msl": [1015.0],
            "weathercode": [1],
        },
    }
    mock_get.return_value = resp
    result = fish.fetch_daily_forecast(45.0, 3.0, 1, start_date=future)
    call_url = mock_get.call_args[0][0]
    params = mock_get.call_args[1]["params"]
    assert "archive" not in call_url
    assert params.get("start_date") == future.isoformat()
    assert params.get("end_date") == future.isoformat()
    assert "forecast_days" not in params
    assert result[0]["date"] == future.isoformat()


@patch("fish.httpx.get")
def test_fetch_daily_forecast_empty_on_network_error(mock_get):
    mock_get.side_effect = httpx.ConnectError("network error")
    result = fish.fetch_daily_forecast(45.0, 3.0, 7)
    assert result == []


@patch("fish.httpx.get")
def test_fetch_daily_forecast_empty_on_malformed_response(mock_get):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"daily": {"time": ["2026-03-29"]}}  # missing keys
    mock_get.return_value = resp
    result = fish.fetch_daily_forecast(45.0, 3.0, 1)
    assert result == []


@patch("fish.httpx.get")
def test_fetch_daily_forecast_uses_archive_for_past(mock_get):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "daily": {
            "time": ["2026-01-15"],
            "temperature_2m_max": [5.0],
            "temperature_2m_min": [-2.0],
            "precipitation_sum": [1.0],
            "windspeed_10m_max": [8.0],
            "winddirection_10m_dominant": [90],
            "weathercode": [3],
            "sunrise": ["2026-01-15T08:15"],
            "sunset": ["2026-01-15T17:10"],
        },
        "hourly": {
            "time": ["2026-01-15T10:00"],
            "temperature_2m": [2.0],
            "precipitation": [0.5],
            "windspeed_10m": [6.0],
            "wind_gusts_10m": [10.0],
            "wind_direction_10m": [90],
            "cloudcover": [70],
        },
    }
    mock_get.return_value = resp
    past = date(2026, 1, 15)
    result = fish.fetch_daily_forecast(45.0, 3.0, 1, start_date=past)
    assert len(result) == 1
    # Verify archive URL was used
    call_url = mock_get.call_args[0][0]
    assert "archive" in call_url


# --- --days in JSON location output ---


@patch("fish.fetch_daily_forecast")
@patch("fish.save_cache")
@patch("fish.load_cache")
@patch("fish.fetch_station_data")
@patch("fish.fetch_today_level")
@patch("fish.search_stations_nearby")
@patch("fish.geocode")
def test_location_json_with_days(
    mock_geo,
    mock_nearby,
    mock_today,
    mock_data,
    mock_cache,
    mock_save,
    mock_daily,
):
    mock_geo.return_value = (45.0, 3.0)
    mock_nearby.return_value = [
        {"code_station": "X1", "libelle_station": "S", "libelle_cours_eau": "R"},
    ]
    mock_today.return_value = 500.0
    mock_data.return_value = {
        "dates": [],
        "values": [],
        "grandeur": "",
        "avg": None,
        "avg_count": 0,
    }
    mock_cache.return_value = {"year": 2026, "data": {}}
    mock_daily.return_value = [
        {
            "date": "2026-03-29",
            "temp_max": 15.0,
            "temp_min": 5.0,
            "precipitation_sum": 0.0,
            "wind_max_kmh": 12.0,
            "wind_direction_dominant": "S",
            "weathercode": 1,
            "sunrise": "07:00",
            "sunset": "19:30",
            "wind_verdict": "Great for casting",
            "hourly": [
                {
                    "hour": "08:00",
                    "temp": 8.0,
                    "precipitation": 0.0,
                    "wind_kmh": 8.0,
                    "wind_gust_kmh": 12.0,
                    "cloudcover": 30,
                    "fishable": True,
                }
            ],
        },
    ]
    with patch("sys.argv", ["fish", "Test", "--json", "--days", "3"]):
        with patch("sys.stdout", new_callable=StringIO) as out:
            fish.main()
    result = json.loads(out.getvalue())
    assert "forecast" in result
    assert len(result["forecast"]) == 1
    assert result["forecast"][0]["date"] == "2026-03-29"
    assert result["forecast"][0]["temp_max"] == 15.0
    assert result["forecast"][0]["wind_verdict"] == "Great for casting"
    mock_daily.assert_called_once_with(45.0, 3.0, 3, start_date=None)


@patch("fish.fetch_daily_forecast")
@patch("fish.save_cache")
@patch("fish.load_cache")
@patch("fish.fetch_station_data")
@patch("fish.fetch_today_level")
@patch("fish.search_stations_nearby")
@patch("fish.geocode")
def test_location_json_without_days_defaults_to_one(
    mock_geo,
    mock_nearby,
    mock_today,
    mock_data,
    mock_cache,
    mock_save,
    mock_daily,
):
    mock_geo.return_value = (45.0, 3.0)
    mock_nearby.return_value = [
        {"code_station": "X1", "libelle_station": "S", "libelle_cours_eau": "R"},
    ]
    mock_today.return_value = 500.0
    mock_data.return_value = {
        "dates": [],
        "values": [],
        "grandeur": "",
        "avg": None,
        "avg_count": 0,
    }
    mock_cache.return_value = {"year": 2026, "data": {}}
    mock_daily.return_value = [{"date": "2026-03-29", "temp_max": 15.0}]
    with patch("sys.argv", ["fish", "Test", "--json"]):
        with patch("sys.stdout", new_callable=StringIO) as out:
            fish.main()
    result = json.loads(out.getvalue())
    assert "forecast" in result
    mock_daily.assert_called_once_with(45.0, 3.0, 1, start_date=None)


@patch("fish.fetch_daily_forecast")
@patch("fish.save_cache")
@patch("fish.load_cache")
@patch("fish.fetch_station_data")
@patch("fish.fetch_today_level")
@patch("fish.search_stations_nearby")
@patch("fish.geocode")
def test_location_json_date_passes_start_date(
    mock_geo,
    mock_nearby,
    mock_today,
    mock_data,
    mock_cache,
    mock_save,
    mock_daily,
):
    # --json with --date must forward the requested date to the forecast fetch,
    # not silently return today's forecast.
    mock_geo.return_value = (45.0, 3.0)
    mock_nearby.return_value = [
        {"code_station": "X1", "libelle_station": "S", "libelle_cours_eau": "R"},
    ]
    mock_today.return_value = 500.0
    mock_data.return_value = {
        "dates": [],
        "values": [],
        "grandeur": "",
        "avg": None,
        "avg_count": 0,
    }
    mock_cache.return_value = {"year": 2026, "data": {}}
    mock_daily.return_value = [{"date": "2026-06-04", "temp_max": 20.0}]
    target = date.today() + timedelta(days=2)
    with patch("sys.argv", ["fish", "Test", "--json", "--date", target.isoformat()]):
        with patch("sys.stdout", new_callable=StringIO) as out:
            fish.main()
    json.loads(out.getvalue())
    mock_daily.assert_called_once_with(45.0, 3.0, 1, start_date=target)
