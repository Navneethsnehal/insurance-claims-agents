"""Fraud-model feature contract.

Training, scoring, and explanation all read this column order. Changing it
requires retraining the saved model.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

FEATURE_NAMES: tuple[str, ...] = (
    "claim_amount",
    "policy_age_days",
    "prior_claims_3y",
    "days_to_report",
    "late_night",
    "attorney_involved",
    "inconsistent_statements",
    "police_report",
    "repair_to_acv_ratio",
    "new_policy_high_value",
    "prior_siu_referral",
    "photo_count",
    "witness_count",
    "provider_repeat_count",
)


def assemble_features(
    *,
    estimated_damage: float,
    vehicle_acv: float | None,
    police_report: bool,
    days_to_report: int,
    supplied: Mapping[str, float | None],
) -> dict[str, float]:
    """Merge internal-file inputs with values derived from the notice itself."""
    supplied_ratio = supplied.get("repair_to_acv_ratio")
    if vehicle_acv is not None and vehicle_acv > 0:
        ratio = float(estimated_damage) / float(vehicle_acv)
    elif supplied_ratio is not None:
        ratio = float(supplied_ratio)
    else:
        ratio = 0.5
    age_raw = supplied.get("policy_age_days")
    age = 400.0 if age_raw is None else float(age_raw)
    amount = float(estimated_damage)

    def _num(name: str) -> float:
        raw = supplied.get(name)
        return 0.0 if raw is None else float(raw)

    return {
        "claim_amount": amount,
        "policy_age_days": age,
        "prior_claims_3y": _num("prior_claims_3y"),
        "days_to_report": float(days_to_report),
        "late_night": _num("late_night"),
        "attorney_involved": _num("attorney_involved"),
        "inconsistent_statements": _num("inconsistent_statements"),
        "police_report": 1.0 if police_report else 0.0,
        "repair_to_acv_ratio": ratio,
        "new_policy_high_value": 1.0 if age < 30 and amount >= 15_000 else 0.0,
        "prior_siu_referral": _num("prior_siu_referral"),
        "photo_count": _num("photo_count"),
        "witness_count": _num("witness_count"),
        "provider_repeat_count": _num("provider_repeat_count"),
    }


def vectorize(features: Mapping[str, float]) -> np.ndarray:
    return np.array([float(features[name]) for name in FEATURE_NAMES], dtype=np.float64)
