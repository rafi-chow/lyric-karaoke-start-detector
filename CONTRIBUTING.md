# Contributing

Thanks for your interest in contributing!

## Ground rules
- Be kind and constructive (see `CODE_OF_CONDUCT.md`).
- Keep PRs small and focused.
- Include tests for bug fixes and new features when practical.

## Development setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Running checks
```bash
pytest -q
python -m compileall src app scripts
```

## Project structure
- `src/` — core feature extraction, labeling, segmentation, start-time logic
- `scripts/` — training / dataset utilities
- `app/` — web API and static frontend

## Pull requests
- Describe what you changed and why.
- Link any relevant issues.
- Include reproduction steps for bugs.

