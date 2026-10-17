"""Estimate indemnity and the case reserve from a coverage decision.

The reserve equals indemnity when coverage confidence is at least 0.8.
Below that we load the reserve by 15 percent so a weak file is not reserved
at a point estimate.
"""

from __future__ import annotations

from typing import Any


def estimate_payout(claim: dict[str, Any], coverage: dict[str, Any]) -> dict[str, Any]:
    deductible = float(claim["deductible"])
    limit = float(claim["limit"])
    if coverage["outcome"] == "excluded":
        basis = 0.0
        indemnity = 0.0
    else:
        basis = float(coverage["covered_basis"])
        indemnity = max(0.0, min(basis, limit) - deductible)
    confidence = float(coverage.get("confidence") or 0)
    reserve = indemnity if confidence >= 0.8 else indemnity * 1.15
    outcome = coverage["outcome"]
    rationale = (
        f"{outcome} loss, basis {basis:.2f}, deductible {deductible:.2f}, "
        f"limit {limit:.2f}, indemnity {indemnity:.2f}, reserve {reserve:.2f}."
    )
    return {
        "indemnity": round(indemnity, 2),
        "reserve": round(reserve, 2),
        "deductible": round(deductible, 2),
        "limit": round(limit, 2),
        "covered_basis": round(basis, 2),
        "rationale": rationale,
    }
