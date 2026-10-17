"""SIU rules that sit beside the gradient-boosting score.

Rules are conservative on purpose: a single strong flag can outrank a calm
model score. The fusion function takes the max, which favors referral recall.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class RuleHit:
    code: str
    weight: float
    detail: str


def apply_rules(features: Mapping[str, float]) -> list[RuleHit]:
    hits: list[RuleHit] = []
    amount = features["claim_amount"]
    if features["policy_age_days"] < 30 and amount >= 15_000:
        hits.append(
            RuleHit(
                "early_high_value",
                0.72,
                "Loss reported on a policy less than 30 days old with a high amount.",
            )
        )
    if features["days_to_report"] > 45:
        hits.append(
            RuleHit(
                "delayed_report",
                0.55,
                "Notice was given more than 45 days after the date of loss.",
            )
        )
    if features["inconsistent_statements"] >= 1:
        hits.append(
            RuleHit(
                "inconsistent_statements",
                0.70,
                "The file contains inconsistent statements.",
            )
        )
    if features["prior_siu_referral"] >= 1:
        hits.append(
            RuleHit(
                "prior_siu",
                0.80,
                "The insured has a prior special-investigations referral.",
            )
        )
    if features["police_report"] < 1 and amount >= 20_000:
        hits.append(
            RuleHit(
                "no_police_high_value",
                0.60,
                "No police report on a high-value loss.",
            )
        )
    if (
        features["attorney_involved"] >= 1
        and features["repair_to_acv_ratio"] < 0.25
        and amount >= 10_000
    ):
        hits.append(
            RuleHit(
                "low_damage_attorney",
                0.66,
                "Attorney involvement with little physical damage relative to value.",
            )
        )
    if features["provider_repeat_count"] >= 4:
        hits.append(
            RuleHit(
                "repeat_provider",
                0.58,
                "The repair provider appears repeatedly across recent files.",
            )
        )
    return hits


def rule_score(hits: list[RuleHit]) -> float:
    if not hits:
        return 0.0
    return max(hit.weight for hit in hits)
