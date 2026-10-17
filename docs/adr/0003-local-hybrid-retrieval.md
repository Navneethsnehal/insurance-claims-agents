# ADR 0003: Local hybrid retrieval, with Postgres available beside it

## Status

Accepted

## Context

Coverage citations have to be verbatim spans of a known form. The corpus is three policies and thirty clauses. A hosted embedding model would add a key, a network hop, and a source of nondeterminism to a decision that has to match in CI. A mandatory pgvector service would make `make test` depend on Docker.

## Decision

`HybridRetriever` indexes the clauses of one policy with BM25 (`rank_bm25`) and 256-dimensional hashed token embeddings. Hashes are SHA-256 of the token, mapped to a signed dimension. Ranks are fused with reciprocal rank fusion (`k=60`). The index is rebuilt in process from `data/policies/corpus.json`.

The coverage engine may cite a clause that missed the top-k. That citation is marked `source=lookup`. Faithfulness does not care which source produced the quote. Retrieval recall does, and it is reported next to the gated metrics.

`docker-compose.yml` starts `pgvector/pgvector:pg16` so a deployment can point `CLAIMS_CHECKPOINTER=postgres` at a durable database. The API process still answers retrieval from the local index. The compose file does not load embeddings into Postgres.

## Consequences

`make test` and `make eval` need no services. Citation text cannot drift from the form unless the corpus file changes.

The hashed vectors are not semantic embeddings. They are stable lexical features. That is enough for this form library and the wrong tool for a multi-thousand-page guideline set. Replacing the vector half with a real embedding model is a local change inside `retrieval/hybrid.py`; the citation contract stays the same.

The Postgres checkpointer is implemented and untested against a live server in this repository. Memory is the runtime CI and the local API use. A process restart drops in-flight human reviews.

## Alternatives

pgvector as the only index was rejected for the reason above: it couples the decision path to an operational dependency the tests do not need. Keeping the image in compose records the intended production neighbor without pretending the hot path uses it today.
