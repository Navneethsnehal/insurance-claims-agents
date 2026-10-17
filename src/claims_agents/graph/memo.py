"""Assemble the decision memo from agent outputs.

The recommendation is computed here. The language model only writes the
narrative paragraph, and that paragraph is not allowed to add clause ids.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def as_dict(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def compose_memo(state: Mapping[str, Any]) -> dict[str, Any]:
    human = state.get("human_decision")
    coverage = as_dict(state.get("coverage"))
    payout = as_dict(state.get("payout"))
    fraud = as_dict(state.get("fraud"))
    outcome = coverage.get("outcome")
    system_indemnity = float(payout.get("indemnity") or 0)
    rejected = isinstance(human, dict) and human.get("decision") == "reject"
    if state.get("halt_reason") and not coverage:
        recommendation = "refer"
        final_indemnity = 0.0
        workflow_status = "needs_attention"
    elif rejected:
        recommendation = "deny"
        final_indemnity = 0.0
        workflow_status = "rejected"
    elif outcome == "excluded":
        recommendation = "deny"
        final_indemnity = 0.0
        workflow_status = "completed"
    elif outcome == "partial":
        recommendation = "partial_pay"
        final_indemnity = system_indemnity
        workflow_status = "completed"
    elif outcome == "covered":
        recommendation = "pay"
        final_indemnity = system_indemnity
        workflow_status = "completed"
    else:
        recommendation = "refer"
        final_indemnity = 0.0
        workflow_status = "needs_attention"
    return {
        "claim_id": str(state.get("claim_id") or ""),
        "recommendation": recommendation,
        "outcome": outcome,
        "final_indemnity": round(final_indemnity, 2),
        "system_indemnity": round(system_indemnity, 2),
        "fraud_score": fraud.get("score"),
        "fraud_band": fraud.get("band"),
        "escalation_reasons": list(state.get("escalation_reasons") or []),
        "human_decision": human if isinstance(human, dict) else None,
        "citations": list(coverage.get("citations") or []),
        "workflow_status": workflow_status,
        "halt_reason": state.get("halt_reason"),
    }


def fallback_narrative(draft: Mapping[str, Any]) -> str:
    outcome = draft.get("outcome") or "undetermined"
    indemnity = float(draft.get("final_indemnity") or 0)
    return (
        f"Claim {draft.get('claim_id')} is {outcome}. "
        f"The recommendation is {draft.get('recommendation')} with indemnity {indemnity:.2f}."
    )
