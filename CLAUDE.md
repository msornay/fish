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
