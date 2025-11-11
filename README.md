# Policy RAG Agent (Cited Answers over Enterprise Docs)

**uv-native + GitLab CI** demo for safe, auditable RAG.

## Dev
```bash
uv venv && source .venv/bin/activate
uv pip install -e .[dev]
uv run uvicorn src.app:app --reload
```
