"""Supervisor policy.

Routing is code. A model does not pick the next agent. The sequence is
intake, coverage, fraud, payout, then human review when a published threshold
trips, then the memo.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from claims_agents.config import Settings
from claims_agents.guardrails.injection import has_high_severity

WORK_NODES = (
    "guardrails",
    "intake",
    "coverage",
    "fraud",
    "payout",
    "human_review",
    "memo",
)


def choose_next(state: Mapping[str, Any]) -> str:
    if state.get("memo"):
        return "end"
    if state.get("halt_reason"):
        return "memo"
    if not state.get("claim"):
        return "intake"
    if not state.get("coverage"):
        return "coverage"
    if not state.get("fraud"):
        return "fraud"
    if not state.get("payout"):
        return "payout"
    if state.get("needs_human") and not state.get("human_decision"):
        return "human_review"
    return "memo"


def _as_float(value: object, default: float) -> float:
    """Parse a metric. Missing values use ``default``; numeric zero does not."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def escalation_reasons(state: Mapping[str, Any], settings: Settings) -> list[str]:
    reasons: list[str] = []
    findings = state.get("injection_findings") or []
    if isinstance(findings, list) and has_high_severity(findings):
        reasons.append("prompt_injection")
    fraud = state.get("fraud") or {}
    fraud_score = _as_float(fraud.get("score"), 0.0) if isinstance(fraud, dict) else 0.0
    if fraud_score >= settings.fraud_hitl_threshold:
        reasons.append("fraud_score")
    claim = state.get("claim") or {}
    if isinstance(claim, dict):
        if _as_float(claim.get("estimated_damage"), 0.0) >= settings.value_hitl_threshold:
            reasons.append("high_value")
        confidence = _as_float(claim.get("extraction_confidence"), 1.0)
        if confidence < settings.confidence_escalation_threshold:
            reasons.append("low_extraction_confidence")
    coverage = state.get("coverage") or {}
    low_coverage = isinstance(coverage, dict) and bool(coverage)
    if low_coverage and _as_float(coverage.get("confidence"), 1.0) < (
        settings.confidence_escalation_threshold
    ):
        reasons.append("low_coverage_confidence")
    return reasons


def worker_path(trace: list[dict[str, Any]], pending: str | None) -> list[str]:
    nodes = [str(event["node"]) for event in trace if event.get("node") in WORK_NODES]
    if pending in WORK_NODES and (not nodes or nodes[-1] != pending):
        nodes.append(str(pending))
    return nodes
