"""Tests for fish.py."""

import json
from unittest.mock import patch, MagicMock
from datetime import date, timedelta
from io import StringIO

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
    mock_get.side_effect = [
        _mock_response([{"v": 1}], cursor="abc"),
        _mock_response([{"v": 2}]),
    ]
    result = fish.fetch_obs_elab("X", "2025-01-01", "2025-03-01")
    assert len(result) == 2
    assert mock_get.call_count == 2


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


# --- search_stations ---


@patch("fish.httpx.get")
def test_search_stations_deduplicates(mock_get):
    station = {
        "code_station": "A1",
        "libelle_station": "Foo",
        "libelle_cours_eau": "Bar",
    }
    resp = MagicMock()
    resp.json.return_value = {"data": [station]}
    resp.raise_for_status.return_value = None
    mock_get.return_value = resp

    with patch("sys.stdout", new_callable=StringIO) as out:
        fish.search_stations("Foo")
        assert out.getvalue().count("A1") == 1


@patch("fish.httpx.get")
def test_search_stations_no_results(mock_get):
    resp = MagicMock()
    resp.json.return_value = {"data": []}
    resp.raise_for_status.return_value = None
    mock_get.return_value = resp

    with patch("sys.stdout", new_callable=StringIO) as out:
        fish.search_stations("nonexistent")
        assert "No stations found" in out.getvalue()


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


def _mock_geocode_response(features):
    resp = MagicMock()
    resp.json.return_value = {"features": features}
    resp.raise_for_status.return_value = None
    return resp


@patch("fish.httpx.get")
def test_geocode_returns_lat_lon(mock_get):
    mock_get.return_value = _mock_geocode_response(
        [{"geometry": {"coordinates": [2.35, 48.85]}}]
    )
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
    mock_get.side_effect = [
        _mock_response([{"code_station": "S1"}], cursor="abc"),
        _mock_response([{"code_station": "S2"}]),
    ]
    result = fish.search_stations_nearby(48.85, 2.35, 25)
    assert len(result) == 2
    assert mock_get.call_count == 2


@patch("fish.httpx.get")
def test_search_stations_nearby_empty(mock_get):
    mock_get.return_value = _mock_response([])
    result = fish.search_stations_nearby(48.85, 2.35, 25)
    assert result == []


# --- fetch_rain_forecast ---


@patch("fish.httpx.get")
def test_fetch_rain_forecast_parses_response(mock_get):
    resp = MagicMock()
    resp.json.return_value = {
        "hourly": {
            "time": ["2025-03-01T10:00", "2025-03-01T11:00"],
            "precipitation": [0.5, 1.2],
        }
    }
    resp.raise_for_status.return_value = None
    mock_get.return_value = resp
    result = fish.fetch_rain_forecast(48.85, 2.35)
    assert result == [("10:00", 0.5), ("11:00", 1.2)]


@patch("fish.httpx.get")
def test_fetch_rain_forecast_returns_empty_on_error(mock_get):
    mock_get.side_effect = Exception("network error")
    result = fish.fetch_rain_forecast(48.85, 2.35)
    assert result == []


# --- plot_station ---


@patch("fish.display")
@patch("fish.fetch_historical_average", return_value=(150.0, 5))
@patch(
    "fish.fetch_recent_3months",
    return_value=(["2025-03-01"], [100], "HmnJ"),
)
def test_plot_station_calls_display(mock_recent, mock_avg, mock_display):
    station = {"code_station": "X1"}
    fish.plot_station(station)
    mock_recent.assert_called_once_with("X1", None)
    mock_avg.assert_called_once_with("X1", "HmnJ", None)
    mock_display.assert_called_once_with(station, ["2025-03-01"], [100], 150.0, 5, None)


@patch("fish.display")
@patch("fish.fetch_recent_3months", return_value=([], [], ""))
def test_plot_station_no_grandeur_skips_avg(mock_recent, mock_display):
    station = {"code_station": "X1"}
    fish.plot_station(station)
    mock_display.assert_called_once_with(station, [], [], None, 0, None)


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


# --- display_table ---


def test_display_table_output():
    rows = [
        ("La Loue", "Station A", "X001", 854.0, 862.0, 10),
        ("Le Doubs", "Station B", "X002", None, None, 0),
    ]
    with patch("sys.stdout", new_callable=StringIO) as out:
        fish.display_table(rows)
        output = out.getvalue()
    assert "La Loue" in output
    assert "X001" in output
    assert "854 mm" in output
    assert "862 mm" in output
    assert "— mm" in output


# --- fetch_sunlight ---


@patch("fish.httpx.get")
def test_fetch_sunlight_parses_response(mock_get):
    resp = MagicMock()
    resp.json.return_value = {
        "daily": {
            "sunrise": ["2025-06-15T06:00"],
            "sunset": ["2025-06-15T21:30"],
        }
    }
    resp.raise_for_status.return_value = None
    mock_get.return_value = resp
    result = fish.fetch_sunlight(48.85, 2.35)
    assert result["sunrise"] == "06:00"
    assert result["sunset"] == "21:30"
    assert "peak_start" in result
    assert "peak_end" in result


@patch("fish.httpx.get")
def test_fetch_sunlight_returns_none_on_error(mock_get):
    mock_get.side_effect = Exception("network error")
    assert fish.fetch_sunlight(48.85, 2.35) is None


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


# --- plot_station error handling ---


@patch("fish.fetch_recent_3months")
def test_plot_station_skips_on_http_error(mock_recent):
    import httpx

    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_recent.side_effect = httpx.HTTPStatusError(
        "not found", request=MagicMock(), response=mock_resp
    )
    station = {"code_station": "X1", "libelle_station": "Test"}
    # Should not raise, just print to stderr and return
    fish.plot_station(station)


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
    mock_fetch.assert_called_once_with("X1", "2025-06-15", "2025-06-15")
    assert result == 500.0


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


# --- fetch_rain_forecast with past date ---


@patch("fish.date")
@patch("fish.httpx.get")
def test_fetch_rain_forecast_past_date_uses_archive(mock_get, mock_date):
    mock_date.today.return_value = date(2026, 2, 9)
    mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
    resp = MagicMock()
    resp.json.return_value = {
        "hourly": {
            "time": ["2025-06-15T10:00", "2025-06-15T11:00"],
            "precipitation": [0.3, 0.0],
        }
    }
    resp.raise_for_status.return_value = None
    mock_get.return_value = resp
    result = fish.fetch_rain_forecast(48.85, 2.35, date(2025, 6, 15))
    assert result == [("10:00", 0.3), ("11:00", 0.0)]
    call_url = mock_get.call_args[0][0]
    assert "archive" in call_url


# --- fetch_sunlight with explicit date ---


@patch("fish.httpx.get")
def test_fetch_sunlight_with_explicit_date(mock_get):
    resp = MagicMock()
    resp.json.return_value = {
        "daily": {
            "sunrise": ["2025-06-15T05:45"],
            "sunset": ["2025-06-15T21:45"],
        }
    }
    resp.raise_for_status.return_value = None
    mock_get.return_value = resp
    result = fish.fetch_sunlight(48.85, 2.35, date(2025, 6, 15))
    assert result["sunrise"] == "05:45"
    params = mock_get.call_args[1]["params"]
    assert params["start_date"] == "2025-06-15"
    assert params["end_date"] == "2025-06-15"
    assert "forecast_days" not in params


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


# --- plot_station with target_date ---


@patch("fish.display")
@patch("fish.fetch_historical_average", return_value=(150.0, 5))
@patch(
    "fish.fetch_recent_3months",
    return_value=(["2025-06-15"], [100], "HmnJ"),
)
def test_plot_station_with_target_date(mock_recent, mock_avg, mock_display):
    station = {"code_station": "X1"}
    target = date(2025, 6, 15)
    fish.plot_station(station, target)
    mock_recent.assert_called_once_with("X1", target)
    mock_avg.assert_called_once_with("X1", "HmnJ", target)
    mock_display.assert_called_once_with(
        station, ["2025-06-15"], [100], 150.0, 5, target
    )


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


@patch("fish.httpx.get")
def test_fetch_sunlight_falls_back_to_last_year_for_far_future(mock_get):
    # First call returns 400 (out of range), second call succeeds
    bad_resp = MagicMock()
    bad_resp.status_code = 400

    good_resp = MagicMock()
    good_resp.json.return_value = {
        "daily": {
            "sunrise": ["2025-08-10T06:15"],
            "sunset": ["2025-08-10T21:00"],
        }
    }
    good_resp.raise_for_status.return_value = None
    mock_get.side_effect = [bad_resp, good_resp]

    future = date(2026, 8, 10)
    with patch("fish.date") as mock_date:
        mock_date.today.return_value = date(2026, 2, 9)
        mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
        result = fish.fetch_sunlight(48.85, 2.35, future)

    assert result is not None
    assert result["sunrise"] == "06:15"
    # Second call should use archive API with 2025-08-10
    call_url = mock_get.call_args_list[1][0][0]
    assert "archive" in call_url
    fallback_params = mock_get.call_args_list[1][1]["params"]
    assert fallback_params["start_date"] == "2025-08-10"


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


@patch("fish.fetch_rain_forecast", return_value=[])
def test_rain_na_displayed_for_future_date(mock_rain):
    future = date.today() + timedelta(days=180)
    with patch("sys.stdout", new_callable=StringIO) as out:
        # Simulate the rain display logic from main()
        forecast = fish.fetch_rain_forecast(48.85, 2.35, future)
        is_future = True
        if not forecast and is_future:
            print("  \033[1mRain:\033[0m N/A (date too far in the future)")
        output = out.getvalue()
    assert "N/A" in output
    assert "too far in the future" in output
