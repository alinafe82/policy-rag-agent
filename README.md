# Policy RAG Agent

Policy question-answering API with citation validation.

The service answers questions against a small local document store and refuses answers that do
not include valid citations. It is a public-safe demo of the guardrail and API shape, not a
production document platform.

## Why It Exists

Internal policy answers are only useful when people can see the source. This repo keeps the
retrieval, answer generation, citation validation, and refusal logic small enough to test.

## Quickstart

```bash
uv venv && source .venv/bin/activate
uv pip install -e .[dev]
uv run uvicorn src.app:app --reload
```

Run tests and linting:

```bash
uv run --extra dev pytest
uv run --extra dev ruff check .
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H 'content-type: application/json' \
  -d '{"query":"What are the MFA requirements?"}'
```

## Architecture Overview

- `src.app` exposes the FastAPI endpoints.
- `src.store` provides the local document store and search behavior.
- `src.llm` isolates provider-backed or mock answer generation.
- `src.guard` validates citations, confidence, and unsafe answer patterns.
- `src.cache` keeps repeat requests cheap in local runs.

See [docs/architecture.md](docs/architecture.md) for design details.

## Limitations

- The document store is in memory.
- The default provider is mock/local behavior.
- This repo does not include tenant isolation, document ingestion, or access control.

## Future Improvements

- Add a real document ingestion pipeline.
- Add source-level authorization before retrieval.
- Track refusal reasons and answer quality metrics.

## Interview Notes

See [docs/interview-notes.md](docs/interview-notes.md).
