# Architecture

## Problem

Internal policy search is risky when answers cannot be traced back to source material. The goal
is to answer simple questions while refusing responses that lack citations or confidence.

## Intended User

The intended user is an internal tools team building a policy Q&A service where source
traceability matters.

## Components

- FastAPI app: request validation, health, metrics, and answer endpoint.
- Store: local document chunks and search behavior.
- LLM provider interface: local answer generation behind a narrow adapter.
- Guard: citation validation, confidence scoring, and refusal logic.
- Cache: short-term response reuse for repeated questions.

## Data Flow

A question enters `/ask`. The app searches the local store, asks the configured provider to
draft an answer, validates citations and confidence, then returns either an answer with sources
or a refusal.

## Design Choices

I kept citation validation as a separate guard because generation and verification are different
concerns. The service should be able to refuse an answer even when the provider returns text.

The local store keeps the repo safe to publish. A real deployment would need ingestion,
authorization, and tenant boundaries before retrieval.

## What Is Not Built

This is not a full enterprise search platform. It does not ingest private documents, enforce
document ACLs, or persist audit records.

## Extension Points

- Replace the in-memory store with a real retrieval backend.
- Add document-level authorization.
- Persist answer/refusal metrics.
- Add provider-specific timeout and fallback behavior.

## Operational Considerations

A production service should protect policy content, log correlation IDs, enforce source access,
and track refusal rates.

## Testing Strategy

Tests cover citation validation, store behavior, cache behavior, app endpoints, and provider
selection. Adapter tests would be needed for retrieval or provider integrations.
