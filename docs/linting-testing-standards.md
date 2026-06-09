# Linting and Testing Standards

These standards define the checks expected before a pull request is marked ready. Run the sections for the
languages touched by the change.

## Required Gates

- Start from the default branch and keep the PR focused on one reviewable change.
- Run `git diff --check` before committing.
- Run `repowave scan .` when `repowave.toml` is present.
- Run every applicable language command below. If a command needs credentials, a live service, or unavailable
  platform tooling, state that in the PR and run the closest local gate.
- Add or update tests for behavior changes. Documentation-only changes still need the diff and repository gates.

## Python

- Use `uv` with the checked-in lockfile.
- Run Ruff for linting and MyPy for typed boundaries.
- Run Pytest for retrieval, policy parsing, authorization, and response generation behavior.
- Do not require live vector databases or model calls for unit tests.

## Current Command Map

- Install: `uv sync`.
- Lint: `make lint`.
- Tests: `make test`.
- Format: `make format` before committing formatting-only changes.
