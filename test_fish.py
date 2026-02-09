"""Tests for fish.py."""

from unittest.mock import patch, MagicMock
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
    result = fish.search_stations_nearby(48.85, 2.35, 100)
    assert len(result) == 2
    params = mock_get.call_args[1]["params"]
    assert params["latitude"] == 48.85
    assert params["longitude"] == 2.35
    assert params["distance"] == 100


@patch("fish.httpx.get")
def test_search_stations_nearby_empty(mock_get):
    mock_get.return_value = _mock_response([])
    result = fish.search_stations_nearby(48.85, 2.35, 50)
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
    mock_recent.assert_called_once_with("X1")
    mock_avg.assert_called_once_with("X1", "HmnJ")
    mock_display.assert_called_once_with(station, ["2025-03-01"], [100], 150.0, 5)


@patch("fish.display")
@patch("fish.fetch_recent_3months", return_value=([], [], ""))
def test_plot_station_no_grandeur_skips_avg(mock_recent, mock_display):
    station = {"code_station": "X1"}
    fish.plot_station(station)
    mock_display.assert_called_once_with(station, [], [], None, 0)
