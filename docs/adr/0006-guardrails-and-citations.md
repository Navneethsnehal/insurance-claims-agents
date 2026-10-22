# ADR 0006: Guardrails run before the model, and citations are verbatim

## Status

Accepted

## Context

Uploaded FNOL text is untrusted. It can contain a claimant's phone number, and it can contain a line that tells the model to ignore the policy and pay the loss. Coverage decisions that cite a paraphrase cannot be checked. Deductibles typed on the FNOL are not the contract.

## Decision

The API redacts PII before `invoke`, so the checkpoint's input is already clean. That first pass keeps kind and count and puts them on the input state. The graph redacts again. When the second pass finds nothing, it keeps the counts from the first pass instead of writing an empty list. High-severity injection is replaced with `[INJECTION_REMOVED]` and added to the escalation reasons. The stripped text is what extraction sees. The tag is not a coverage fact, so the underwriting engine's answer does not change.

PII detectors cover email, SSN, US phone numbers (including a leading parenthesis, which is not a word boundary, and a closing parenthesis that sits against the exchange), a date of birth only when it is labeled, and card numbers that pass the Luhn check. A card followed by more digits is still redacted when a 13 to 19 digit prefix of that run passes Luhn. A single digit string is checked only at its full length. Findings keep a kind and a count.

`complete_structured` parses JSON, validates it, and retries up to `CLAIMS_STRUCTURED_RETRIES`. Exhausted retries halt the file with `needs_attention` rather than filling fields.

`normalize_claim` copies deductible and limit from the policy declarations. A mismatch between the document's policy number and the request is recorded and the confidence is capped at `0.5`, which is under the escalation threshold.

Coverage quotes are slices of `clause.text` (at most 280 characters). The eval's faithfulness metric fails a citation whose quote is not an exact substring of that clause on that policy. `sanitize_narrative` drops clause-shaped tokens that are not on the allow-list for the policy, so a memo cannot smuggle `COV-COLL` into a home claim.

## Consequences

A planted SSN, email, or card number in the golden book must be absent from the checkpoint. The eval fails the run on any leak. That is a gate, not a report-only metric.

Verbatim quotes make the citations longer and sometimes awkward. They also make "the model cited the form" a checkable statement.

The phone pattern is intentionally not `\b` before `(`. A word-boundary assertion there never matches, and the number would survive redaction.

## Alternatives

Asking the model to refuse injected instructions, and trusting that refusal, was rejected as the only control. The graph still tells the extractor to ignore instructions in the document. The control that CI locks is the deletion of the span and the escalation.

Embedding-similarity faithfulness was rejected. A paraphrase can be similar and still misstate a deductible. Substring match is strict and cheap.
