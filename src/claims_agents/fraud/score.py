"""Score a claim with the saved model and the SIU rules."""

from __future__ import annotations

from typing import Any

import numpy as np

from claims_agents.fraud.explain import explain_contributions
from claims_agents.fraud.features import FEATURE_NAMES, vectorize
from claims_agents.fraud.rules import apply_rules, rule_score


def fraud_band(score: float, high_threshold: float) -> str:
    if score >= high_threshold:
        return "high"
    if score >= 0.35:
        return "medium"
    return "low"


class FraudScorer:
    def __init__(
        self,
        model: Any,
        baseline: np.ndarray,
        high_threshold: float,
    ) -> None:
        self.model = model
        self.baseline = np.asarray(baseline, dtype=np.float64)
        self.high_threshold = high_threshold

    def assess(self, claim: dict[str, Any]) -> dict[str, Any]:
        features = claim["fraud_features"]
        vector = vectorize(features)
        model_score = float(self.model.predict_proba(vector.reshape(1, -1))[0, 1])
        hits = apply_rules(features)
        rules = rule_score(hits)
        score = min(1.0, max(model_score, rules))
        return {
            "score": round(score, 4),
            "model_score": round(model_score, 4),
            "rule_score": round(rules, 4),
            "band": fraud_band(score, self.high_threshold),
            "flags": [
                {"code": hit.code, "weight": hit.weight, "detail": hit.detail} for hit in hits
            ],
            "explanations": explain_contributions(self.model, vector, self.baseline, FEATURE_NAMES),
        }
