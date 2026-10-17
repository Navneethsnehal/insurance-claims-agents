# ADR 0002: A deterministic fake LLM is the default provider

## Status

Accepted

## Context

CI has no API keys. Extraction and the memo are the only model calls, and both have a contract: JSON that validates as `ClaimExtraction`, and a short paragraph that does not invent clause ids. If tests call a hosted model, failures are flakes and the eval numbers are not reproducible.

## Decision

`CLAIMS_LLM_PROVIDER` defaults to `fake`. `FakeLLM` dispatches on a task marker in the system prompt.

- `TASK: extract_claim` parses the FNOL with the same deterministic parser the normalizer trusts. A document that does not parse returns `{}` so the structured-output retry path fails closed.
- `TASK: write_memo` fills a template from the JSON it was given. It does not look up clauses.

`build_llm` returns an OpenAI or Anthropic client when the provider and the matching API key are set. Those clients are thin and take an injectable SDK client so the tests can assert token accounting without a network call.

## Consequences

The golden-set metrics in `reports/` are properties of the graph, the form, and the fraud model. They do not depend on a vendor. A change to a prompt that the fake client ignores will not show up in CI; the structured schema and the memo template are what CI locks.

Hosted providers stay available for a demo that wants model prose. They are not on the path that gates a merge.

## Alternatives

Recorded HTTP cassettes were rejected. They rot when the SDK changes, and they still encode one vendor's output. The fake client is smaller than the cassette and has an explicit contract.
