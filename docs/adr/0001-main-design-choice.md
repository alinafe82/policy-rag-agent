# ADR 0001: Refuse Answers Without Valid Citations

## Status

Accepted

## Context

Policy answers are dangerous when they cannot be traced to source material. A confident answer
without evidence is worse than a clear refusal.

## Decision

The service validates citations after answer generation and refuses responses that do not meet
the guard rules.

## Consequences

This makes the system more conservative. The tradeoff is that some usable answers may be
rejected until retrieval and generation quality improve.
