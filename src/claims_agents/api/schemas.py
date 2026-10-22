"""Request and response models for the claims API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SubmitClaimRequest(BaseModel):
    policy_id: str = Field(min_length=3, max_length=40)
    documents: list[str] = Field(min_length=1)
    claim_id: str | None = Field(default=None, min_length=4, max_length=40)


class ResumeClaimRequest(BaseModel):
    decision: Literal["approve", "reject"]
    reviewer: str = Field(min_length=1, max_length=80)
    notes: str = ""


class UsageTotals(BaseModel):
    input_tokens: float
    output_tokens: float
    latency_ms: float
    cost_usd: float


class ClaimResponse(BaseModel):
    claim_id: str
    policy_id: str
    status: str
    created_at: str
    updated_at: str
    claim: dict[str, Any] | None = None
    coverage: dict[str, Any] | None = None
    fraud: dict[str, Any] | None = None
    payout: dict[str, Any] | None = None
    memo: dict[str, Any] | None = None
    trace: list[dict[str, Any]]
    usage: list[dict[str, Any]]
    usage_totals: UsageTotals
    pending_review: dict[str, Any] | None = None
    errors: list[str]
    escalation_reasons: list[str]
    pii_findings: list[dict[str, Any]] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok"]
    llm_provider: str
    vector_store: Literal["local_hybrid"]
    checkpointer: str
    fraud_model: Literal["loaded"]
    policies: int
    clauses: int
