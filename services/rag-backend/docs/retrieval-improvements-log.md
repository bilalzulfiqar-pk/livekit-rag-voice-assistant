# Retrieval Improvements Log

This file tracks retrieval-specific improvements made in the backend, why they were made, and how they affect grounded search behavior.

## 2026-04-30 - FAQ Stemming And Short-Query Lexical Rescue

Status: completed

Area: FAQ retrieval reliability for short voice-style questions

Goal:

- make short FAQ questions like `What services do you offer?` retrieve grounded matches more reliably
- keep the API contract unchanged while improving the backend's retrieval tolerance
- preserve the existing `exact` and `ann_rerank` request options

What changed:

- switched PostgreSQL full-text search from `simple` to `english` for the stored `search_vector` expression
- switched sparse-query generation from `websearch_to_tsquery('simple', ...)` to `websearch_to_tsquery('english', ...)`
- added schema-upgrade logic that rebuilds the generated `search_vector` column when an older database still uses `simple`
- lowered the lexical-rescue gate from `3` normalized terms to `2` so short FAQ questions can use the rescue path
- changed sparse-query text building so it preserves raw plural forms like `services` and lets PostgreSQL stemming handle normalization instead of singularizing first
- added regression tests for:
  - `english` search-vector generation
  - rebuilding old `simple` search-vector definitions
  - plural-preserving sparse query text
  - lexical rescue for 2-term FAQ queries

Why this matters:

- the old `simple` text-search config treated `service` and `services` as different tokens
- short natural questions often only contain two strong terms, so the old lexical-rescue gate could skip the exact fallback that should have saved the retrieval
- this change improves FAQ reliability without weakening the main vector retrieval API or changing the agent integration

Easy explanation:

- the backend now stems words more intelligently and gives short FAQ questions a second chance instead of dropping straight to `no context`.

Verification:

- focused backend regression tests:
  - `python -m unittest tests.unit.test_retrieval_ann tests.unit.test_schema_hnsw -v`
- full backend unit suite:
  - `python -m unittest discover -s tests/unit -v`
- live local verification after backend rebuild:
  - `POST /chat/ask` with `What services do you offer?` now returns the grounded FAQ answer

Important note:

- `POST /retrieval/search` is still the lower-level retrieval endpoint and currently does not apply the chat-only sparse-plus-lexical rescue path.
- the confirmed end-user fix is on `POST /chat/ask`, which is the path used by the LiveKit voice agent.
