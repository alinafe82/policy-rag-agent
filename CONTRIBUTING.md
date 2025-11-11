# Contributing

## Dev with uv
```bash
uv venv
source .venv/bin/activate
uv pip install -e .[dev]
pre-commit install
```
Run checks:
```bash
ruff check . && mypy src && pytest -q
```
