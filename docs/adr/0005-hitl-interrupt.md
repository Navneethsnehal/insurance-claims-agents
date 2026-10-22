# ADR 0005: Human review is a LangGraph interrupt inside the node

## Status

Accepted

## Context

High-fraud and high-value files have to stop before a memo is issued, survive long enough for a reviewer to answer, and then continue on the same thread. A side queue that the API polls would fork the decision state from the graph state.

## Decision

`human_review` builds the payload the reviewer sees and calls `interrupt(payload)`. The compiled graph uses a checkpointer (`MemorySaver` by default). The first `invoke` returns with the interrupt set. `read_interrupt` lifts that payload into `pending_review` on the API response, and the status is `awaiting_human`.

Resume is `graph.invoke(Command(resume={"decision", "reviewer", "notes"}), config)` on the same `thread_id`. The decision must be `approve` or `reject`. `ClaimService.resume` raises `ClaimStateError` before `invoke` when the value is anything else, so the interrupt is still pending. The node itself does not raise on a bad value: `interrupt()` has already consumed that resume, and an exception there leaves a later approve undeliverable. It calls `interrupt` again and stays paused.

Approve stores the decision and lets the supervisor write the memo with the system indemnity. Reject stores the decision and `compose_memo` forces `deny` and indemnity `0`. An excluded outcome stays `deny` on approve; the reviewer is not a coverage override.

The node does not append trace or usage events before the interrupt. LangGraph re-runs the node from the start on resume, and a reducer append before the interrupt would double-count.

## Consequences

The pause and the resume are the same code path the tests call. There is no second implementation in the API.

Memory checkpoints vanish when the process exits. A reviewer who comes back after a restart gets a 404, not a stale approval. Durable review needs the Postgres checkpointer, which is opt-in and not part of the CI gate.

The golden book treats an interrupt as success when the expected nodes are `guardrails`, `intake`, `coverage`, `fraud`, `payout`, `human_review` and the memo is absent. The eval does not resume those files; resume behavior is covered by the graph tests.

## Alternatives

A breakpoint set from outside the node was rejected because the payload (reasons, score, recommended indemnity) is easiest to build where the state is already in hand. A database row written beside the graph was rejected because the API would then have two sources of truth for "is this file waiting."
