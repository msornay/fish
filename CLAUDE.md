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

## Git Workflow
- Never commit directly to `main`. Always create a feature branch and open a PR.
- Branch naming: use descriptive kebab-case (e.g. `add-rain-forecast`, `fix-pagination-bug`).
- Push the branch and create a PR with `gh pr create`.
- Wait for CI to pass and the PR to be reviewed before merging.
- Always add `@copilot` as a reviewer on PRs (`gh pr edit --add-reviewer copilot`).
- Wait for Copilot's review comments before proceeding. Address comments that are valid; dismiss ones that are not.
