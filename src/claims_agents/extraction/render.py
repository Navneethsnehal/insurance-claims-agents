"""Render a synthetic FNOL in the layout the intake parser expects."""

from __future__ import annotations

from collections.abc import Mapping


def render_fnol(
    *,
    claim_id: str,
    policy_id: str,
    claimant_name: str,
    loss_type: str,
    date_of_loss: str,
    date_reported: str,
    estimated_damage: float,
    description: str,
    fact_tags: list[str] | tuple[str, ...],
    features: Mapping[str, float],
    vehicle_acv: float | None = None,
    injuries: bool = False,
    police_report: bool = False,
    aftermarket_amount: float = 0,
    mold_amount: float = 0,
    stated_deductible: float | None = None,
) -> str:
    tags = ", ".join(fact_tags) if fact_tags else "none"
    lines = [
        "FIRST NOTICE OF LOSS",
        f"Claim reference: {claim_id}",
        f"Policy number: {policy_id}",
        f"Named insured: {claimant_name}",
        f"Loss type: {loss_type}",
        f"Date of loss: {date_of_loss}",
        f"Date reported: {date_reported}",
        f"Estimated damage USD: {estimated_damage}",
    ]
    if vehicle_acv is not None:
        lines.append(f"Vehicle ACV USD: {vehicle_acv}")
    if aftermarket_amount:
        lines.append(f"Aftermarket amount USD: {aftermarket_amount}")
    if mold_amount:
        lines.append(f"Mold amount USD: {mold_amount}")
    if stated_deductible is not None:
        lines.append(f"Stated deductible USD: {stated_deductible}")
    lines.extend(
        [
            f"Injuries reported: {'yes' if injuries else 'no'}",
            f"Police report filed: {'yes' if police_report else 'no'}",
            "Description:",
            description.strip(),
            f"Fact tags: {tags}",
            "Internal file supplements:",
        ]
    )
    for key in (
        "policy_age_days",
        "prior_claims_3y",
        "late_night",
        "attorney_involved",
        "inconsistent_statements",
        "prior_siu_referral",
        "photo_count",
        "witness_count",
        "provider_repeat_count",
        "repair_to_acv_ratio",
    ):
        if key in features:
            lines.append(f"{key}: {features[key]}")
    return "\n".join(lines) + "\n"
