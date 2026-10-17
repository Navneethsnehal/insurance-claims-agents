"""Turn an extraction into the claim record the other agents consume.

Financial terms come from the policy declarations, not from the uploaded
document. A notice that says the deductible is zero does not make it zero.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from claims_agents.fraud.features import FEATURE_NAMES, assemble_features
from claims_agents.retrieval.corpus import Corpus, Policy
from claims_agents.schemas import ClaimExtraction

_COMPREHENSIVE = {"comprehensive", "theft", "vandalism", "weather", "animal", "fire"}


def policy_deductible(policy: Policy, loss_type: str) -> float:
    terms = policy.terms
    if policy.product == "homeowners":
        return float(terms["deductible"])
    if loss_type == "liability":
        return 0.0
    if loss_type == "collision":
        return float(terms["collision_deductible"])
    if loss_type in _COMPREHENSIVE:
        return float(terms["comprehensive_deductible"])
    return float(terms.get("collision_deductible", terms.get("deductible", 0.0)))


def policy_limit(policy: Policy) -> float:
    return float(policy.terms["limit"])


def normalize_claim(
    extraction: ClaimExtraction,
    *,
    claim_id: str,
    policy_id: str,
    corpus: Corpus,
) -> dict[str, Any]:
    """Build the checkpointed claim dict.

    ``claim_id`` and ``policy_id`` are the system of record, not the document.
    """
    policy = corpus.require(policy_id)
    deductible = policy_deductible(policy, extraction.loss_type)
    limit = policy_limit(policy)
    confidence = extraction.extraction_confidence
    notes: list[str] = []
    mismatch = extraction.policy_id != policy_id
    if mismatch:
        notes.append("policy_id_mismatch")
        confidence = min(confidence, 0.5)
    if (
        extraction.stated_deductible is not None
        and abs(extraction.stated_deductible - deductible) > 0.01
    ):
        notes.append("deductible_conflict")
        confidence = min(confidence, 0.5)

    loss = date.fromisoformat(extraction.date_of_loss)
    reported = date.fromisoformat(extraction.date_reported)
    days_to_report = max(0, (reported - loss).days)
    features = assemble_features(
        estimated_damage=extraction.estimated_damage,
        vehicle_acv=extraction.vehicle_acv,
        police_report=extraction.police_report,
        days_to_report=days_to_report,
        supplied=extraction.fraud_features.model_dump(),
    )
    missing_features = [name for name in FEATURE_NAMES if name not in features]
    if missing_features:  # pragma: no cover - assemble_features always fills the vector
        notes.append("features_imputed")

    return {
        "claim_id": claim_id,
        "document_claim_id": extraction.claim_id,
        "policy_id": policy_id,
        "document_policy_id": extraction.policy_id,
        "policy_mismatch": mismatch,
        "product": policy.product,
        "claimant_name": extraction.claimant_name,
        "loss_type": extraction.loss_type,
        "date_of_loss": extraction.date_of_loss,
        "date_reported": extraction.date_reported,
        "description": extraction.description,
        "estimated_damage": round(float(extraction.estimated_damage), 2),
        "vehicle_acv": extraction.vehicle_acv,
        "deductible": round(deductible, 2),
        "stated_deductible": extraction.stated_deductible,
        "limit": round(limit, 2),
        "injuries": extraction.injuries,
        "police_report": extraction.police_report,
        "fact_tags": list(extraction.fact_tags),
        "aftermarket_amount": round(float(extraction.aftermarket_amount), 2),
        "mold_amount": round(float(extraction.mold_amount), 2),
        "fraud_features": features,
        "extraction_confidence": round(confidence, 4),
        "notes": notes,
    }
