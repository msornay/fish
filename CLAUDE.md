# Project Notes

## Environment
- Use the virtualenv at `~/venv/dev` for all dependency installation and script execution.
- Activate with: `source ~/venv/dev/bin/activate`

## Testing
- Run tests with: `pytest test_fish.py -v`

## Linting & Formatting
- Lint: `ruff check .`
- Format: `ruff format .`
- Both are enforced in CI and required to pass before merging to `main`.
