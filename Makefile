.PHONY: setup lint test format run docker

setup:
	uv venv && . .venv/bin/activate && uv pip install -e .[dev]

lint:
	ruff check . && mypy src

format:
	ruff check . --fix && ruff format . && isort . && black .

test:
	pytest -q --maxfail=1

run:
	uv run uvicorn src.app:app --reload --port 8000

docker:
	docker build -t ghcr.io/${USER}/policy-rag-agent:local .
