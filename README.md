# Policy RAG Agent

A FastAPI service that answers policy questions against a small local document store **and refuses to answer when it cannot cite the source**. The refusal path is the actual engineering point of this repo; the answer path is the easy part.

> **Access control is not implemented.** The local document store has no tenant isolation, no per-document ACL, no row-level filtering. Anyone who can reach the API can see every document. Treat this repo as a guardrail demo, not a multi-tenant policy platform. Any production use needs source-level authorization wired in before retrieval.

## Why the refusal path matters more than the answer path

A RAG system that answers confidently without citing a source is a liability dressed up as a productivity tool. People will quote its output to make decisions; if there is no traceable source, the quoted thing is now part of the institutional record with no provenance.

So this repo's guardrail does four things in order, and any one of them sends the request down the refusal path:

1. Confirm the model's response cites a document the retriever actually returned.
2. Confirm the cited document is not a hallucinated identifier.
3. Confirm the response confidence clears the threshold.
4. Confirm the response does not match unsafe-answer patterns (long uncited paragraphs, "as an AI" preambles, etc.).

Refusal returns a structured response that says *what* failed, not just "I don't know". That is the part a reviewer can audit.

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

## Service layout

- `src.app` — FastAPI endpoints, request validation, error envelope.
- `src.store` — in-memory document store and lookup (this is the part that would be replaced by a real retriever).
- `src.llm` — provider boundary; mock backend is default so the repo runs without keys.
- `src.guard` — the refusal logic: citation validation, confidence threshold, unsafe-pattern check.
- `src.cache` — repeat-question cache for local runs.

Design notes: [docs/architecture.md](docs/architecture.md).

## What the tests prove

`tests/test_guard.py` covers the refusal contract:

- responses without citations are refused.
- responses citing documents the retriever did not return are refused.
- responses below the confidence threshold are refused.
- empty responses are refused.
- responses matching hallucination patterns are refused.
- approved responses carry decision metadata explaining why they were approved.

`tests/test_app.py` covers the API envelope and `tests/test_cache.py` covers cache behaviour.

## Adapter work left before this would serve real policies

- Replace the in-memory `src.store` with a real retriever (Elasticsearch, pgvector, etc.).
- Add source-level authorization before retrieval. The current code retrieves first and answers second, which is the wrong order if some users should not see some documents.
- Add a real LLM provider, not the mock backend. Token, timeout, and cost limits go in `src.llm`.
- Add a refusal-reason metric so a dashboard can show the rate of each refusal cause over time.

## Operational notes

- [docs/runbook.md](docs/runbook.md) if present
- [docs/security-notes.md](docs/security-notes.md)
- [docs/production-readiness.md](docs/production-readiness.md)
- [docs/interview-notes.md](docs/interview-notes.md)
