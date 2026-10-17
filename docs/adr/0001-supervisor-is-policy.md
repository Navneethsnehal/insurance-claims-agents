# ADR 0001: The supervisor is policy, not a router model

## Status

Accepted

## Context

A claims file has a fixed set of questions: what happened, does the form respond, does the file look wrong, what do we reserve, and does a person need to see it. Those questions have an order. A model that picks the next agent can skip coverage, call the memo twice, or loop.

## Decision

`choose_next` is ordinary Python. The supervisor node records the hop and returns the name of the next worker. Conditional edges map those names onto nodes. The language model is used in two places: structured extraction of the FNOL, and one narrative paragraph after the recommendation has already been computed.

Escalation thresholds (`0.65` fraud, `$25,000` damage, `0.55` confidence, high-severity injection) are settings, not prose in a prompt.

## Consequences

Routing accuracy is a property of the policy and of whether each worker filled its slot. It does not move when the chat model changes. Adding a new worker means extending the state, the policy, and the golden book together.

The tradeoff is flexibility. The graph will not invent an investigation step for an odd file. That step is a human interrupt, which is the behavior we want for a triage service.

## Alternatives

An LLM router with a tool list was rejected. It is harder to test, it spends tokens on a decision the code already knows, and a jailbroken document can argue the router into the wrong node. The injection guardrail still exists, but it does not have to also defend the control plane.
