# fish

[![CI](https://github.com/msornay/fish/actions/workflows/ci.yml/badge.svg)](https://github.com/msornay/fish/actions/workflows/ci.yml)

French river water height CLI tool using the [Hub'Eau](https://hubeau.eaufrance.fr/) API.

Shows water levels, 10-year historical averages, rain forecasts, and sunrise/sunset times for any location in France.

## Installation

```bash
pip install httpx plotext
```

## Usage

### Water levels near a location

```bash
fish Paris
```

Finds hydrometric stations within 25 km, shows current water height vs. 10-year average, rain forecast, and sunlight times.

### Plot a specific station

```bash
fish --station F700000103
```

Shows a 3-month water height graph with historical average for the given station code.

### Search stations

```bash
fish --station-list Seine
fish --station-list "Pont Neuf"
```

Searches by river name or station name.

### Historical or future dates

```bash
fish Paris --date 2025-06-15
fish Paris --tomorrow
```

`--date` shows archived data for a past date or forecast for a near-future date. `--tomorrow` is a shortcut.

## Features

- Water height table with current level and 10-year average per station
- 3-month terminal graph (braille characters via plotext)
- Rain forecast from [Open-Meteo](https://open-meteo.com/)
- Sunrise/sunset and peak sunlight window
- Historical average cache (`~/.cache/fish/hist_avg.json`), invalidated yearly
- Geocoding via the [French address API](https://adresse.data.gouv.fr/)

## Development

```bash
pip install httpx plotext pytest ruff
pytest test_fish.py -v
ruff check .
```

Or use the Makefile:

```bash
make test
```
