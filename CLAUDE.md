# CLAUDE.md

## Commands

- `make test` — run tests + lint + format check
- `make lint` — ruff check + format check

## Style

- Lint and format with `ruff`

## Design notes

- **Grandeur matching for level vs 10y avg.** Some stations expose only one
  elaborated daily height grandeur — e.g. Chenecey-Buillon (U262401001) only
  publishes `HIXnJ` (daily max), not `HmnJ` (daily mean). `fetch_today_level`
  takes the grandeur picked from the 3-month obs_elab series and aggregates
  today's real-time `H` readings to match (`HIXnJ`→max, `HINnJ`→min,
  `HmnJ`→mean). Comparison stays apples-to-apples. When today's real-time
  obs are empty (early morning, station gap), falls back to the latest instant
  reading so the station still displays.

- **A target date is honored via `start_date`/`end_date`, not `forecast_days`.**
  `fetch_daily_forecast` uses `start_date`/`end_date` on whichever API (archive
  for past, forecast for today/future) whenever a date is requested, and only
  falls back to `forecast_days` when `start_date is None`. Earlier the future
  branch dropped `start_date` and the `--json` path never passed it, so
  `--date <future>` (and any `--date --json`) silently returned *today's*
  forecast. Both the human and JSON paths now forward `target_date`.

- **Thunderstorm verdicts are per fishable hour, from hourly `weathercode`.**
  Open-Meteo's daily `weathercode` is a daily-max aggregate: one stormy hour
  stamps 95/96/99 on the whole day. `technique_verdicts(hours)` instead checks
  each fishable hour's own `weathercode`, so an overnight storm no longer
  blankets every technique to no-go — only a storm during the fishable window
  does. Hourly `weathercode` is fetched alongside the other hourly fields. The
  daily `weathercode` is kept only for display.
