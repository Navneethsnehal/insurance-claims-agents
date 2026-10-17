"""Local feature contributions via one-at-a-time ablation.

For each feature we score the training-median baseline with that feature set
to the claim's value. The difference from the baseline probability is the
contribution. This is not interaction-aware TreeSHAP; it is a stable,
dependency-free attribution we can test. An ADR records the tradeoff.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def explain_contributions(
    model: Any,
    vector: np.ndarray,
    baseline: np.ndarray,
    feature_names: tuple[str, ...],
) -> list[dict[str, float | str]]:
    base = _positive_proba(model, baseline.reshape(1, -1))
    rows: list[dict[str, float | str]] = []
    for index, name in enumerate(feature_names):
        perturbed = baseline.copy()
        perturbed[index] = vector[index]
        delta = _positive_proba(model, perturbed.reshape(1, -1)) - base
        if delta > 1e-4:
            direction = "increases_risk"
        elif delta < -1e-4:
            direction = "decreases_risk"
        else:
            direction = "neutral"
        rows.append(
            {
                "feature": name,
                "value": round(float(vector[index]), 4),
                "baseline": round(float(baseline[index]), 4),
                "delta": round(float(delta), 4),
                "direction": direction,
            }
        )
    rows.sort(key=lambda row: abs(float(row["delta"])), reverse=True)
    return rows


def _positive_proba(model: Any, row: np.ndarray) -> float:
    proba = model.predict_proba(row)
    return float(proba[0, 1])
