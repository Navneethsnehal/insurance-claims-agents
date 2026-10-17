"""Pydantic contracts shared by extraction, decisions, and the API.

Graph state stores ``model_dump()`` dictionaries so the LangGraph checkpointer
can serialize them. These models are the validation boundary.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

CoverageOutcome = Literal["covered", "excluded", "partial"]
Recommendation = Literal["pay", "deny", "partial_pay", "refer"]
FraudBand = Literal["low", "medium", "high"]


class FraudFeatureInput(BaseModel):
    """Internal-file features that do not come from the loss narrative."""

    policy_age_days: float = 400
    prior_claims_3y: float = 0
    late_night: float = 0
    attorney_involved: float = 0
    inconsistent_statements: float = 0
    prior_siu_referral: float = 0
    photo_count: float = 0
    witness_count: float = 0
    provider_repeat_count: float = 0
    repair_to_acv_ratio: float | None = None


class ClaimExtraction(BaseModel):
    """Structured notice of loss produced by the intake model."""

    claim_id: str
    policy_id: str
    claimant_name: str
    loss_type: str
    date_of_loss: str
    date_reported: str
    description: str
    estimated_damage: float
    vehicle_acv: float | None = None
    stated_deductible: float | None = None
    injuries: bool = False
    police_report: bool = False
    fact_tags: list[str] = Field(default_factory=list)
    aftermarket_amount: float = 0
    mold_amount: float = 0
    fraud_features: FraudFeatureInput = Field(default_factory=FraudFeatureInput)
    extraction_confidence: float = 1


class Citation(BaseModel):
    clause_id: str
    title: str
    role: Literal["grant", "grant_overridden", "exclusion", "limit", "condition"]
    quote: str
    source: Literal["retrieval", "lookup"]


class CoverageDecision(BaseModel):
    outcome: CoverageOutcome
    citations: list[Citation]
    rationale: str
    covered_basis: float
    confidence: float
    retrieved_ids: list[str]


class RuleFlag(BaseModel):
    code: str
    weight: float
    detail: str


class FeatureContribution(BaseModel):
    feature: str
    value: float
    baseline: float
    delta: float
    direction: Literal["increases_risk", "decreases_risk", "neutral"]


class FraudAssessment(BaseModel):
    score: float
    model_score: float
    rule_score: float
    band: FraudBand
    flags: list[RuleFlag]
    explanations: list[FeatureContribution]


class PayoutEstimate(BaseModel):
    indemnity: float
    reserve: float
    deductible: float
    limit: float
    covered_basis: float
    rationale: str


class HumanDecision(BaseModel):
    decision: Literal["approve", "reject"]
    reviewer: str
    notes: str = ""


class DecisionMemo(BaseModel):
    claim_id: str
    recommendation: Recommendation
    outcome: str | None
    final_indemnity: float
    system_indemnity: float
    fraud_score: float | None
    fraud_band: str | None
    escalation_reasons: list[str]
    human_decision: HumanDecision | None
    citations: list[Citation]
    workflow_status: Literal["completed", "rejected", "needs_attention"]
    halt_reason: str | None
    narrative: str
