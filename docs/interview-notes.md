# Interview Notes

## 60-Second Explanation

This is a FastAPI policy Q&A service. It retrieves local policy chunks, generates or mocks an
answer, validates citations, and refuses answers that cannot be tied back to source documents.

## Decisions I Can Defend

- Citation validation happens after generation so provider output is not trusted blindly.
- Refusals are a first-class outcome because unsupported policy answers create risk.
- The store is local so the public repo does not contain private documents.

## Tradeoffs

The repo shows the API and guardrail pattern, not a full retrieval platform. Production use
would need ingestion, ACLs, persistence, observability, and provider reliability controls.

## Fixes Made During Portfolio Hardening

- Removed inflated production language.
- Fixed pytest import configuration for fresh clones.
- Added GitHub Actions CI.
- Added architecture notes, ADR, and interview notes.

## Likely Questions

**Why refuse answers?**
For policy questions, unsupported answers are risky. A refusal with a reason is safer than a
guess.

**What is missing for production?**
Document ingestion, source authorization, persistent audit logs, retrieval evaluation, and
provider timeout/fallback behavior.

**What does this show for Engineering Productivity?**
It shows how I would build internal API guardrails around tools that developers rely on for
operational decisions.
