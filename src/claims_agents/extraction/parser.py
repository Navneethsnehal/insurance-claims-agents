"""Deterministic FNOL parser.

The production intake node asks a model for this same JSON shape. The fake
provider delegates to this parser so tests do not depend on an API, and the
parser is also the schema the real provider is graded against.
"""

from __future__ import annotations

import re
from datetime import date

from claims_agents.errors import ExtractionError
from claims_agents.schemas import ClaimExtraction, FraudFeatureInput

_LOSS_ALIASES = {
    "collision": "collision",
    "crash": "collision",
    "comprehensive": "comprehensive",
    "comp": "comprehensive",
    "theft": "theft",
    "burglary": "theft",
    "vandalism": "vandalism",
    "weather": "weather",
    "hail": "weather",
    "wind": "weather",
    "animal": "animal",
    "deer": "animal",
    "liability": "liability",
    "water": "water",
    "fire": "fire",
    "other": "other",
}

_FEATURE_KEYS = (
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
)

_REQUIRED = (
    "claim_id",
    "policy_id",
    "claimant_name",
    "loss_type",
    "date_of_loss",
    "date_reported",
    "description",
    "estimated_damage",
)


def canonical_loss_type(value: str) -> str | None:
    """Map a notice's loss label onto the coverage engine's vocabulary."""
    return _LOSS_ALIASES.get(value.strip().lower())


def canonical_fact_tag(value: str) -> str:
    """Lowercase a fact tag and turn spaces into the underscores the form uses."""
    return "_".join(value.strip().lower().split())


def _line(text: str, label: str) -> str | None:
    match = re.search(rf"(?im)^{re.escape(label)}\s*:\s*(.+?)\s*$", text)
    if match is None:
        return None
    value = match.group(1).strip()
    return value or None


def _money(value: str | None) -> float | None:
    if value is None:
        return None
    cleaned = value.replace(",", "").replace("$", "")
    try:
        return float(cleaned)
    except ValueError as exc:
        raise ExtractionError(f"could not parse amount {value!r}") from exc


def _yes(value: str | None) -> bool:
    return (value or "").strip().lower() in {"yes", "true", "y", "1"}


def _iso_date(value: str | None, label: str) -> str:
    if value is None:
        raise ExtractionError(f"missing {label}")
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise ExtractionError(f"invalid {label}: {value}") from exc


def parse_fnol(text: str) -> ClaimExtraction:
    """Parse one FNOL document. Raises ``ExtractionError`` when required fields are absent."""
    if not text or not text.strip():
        raise ExtractionError("empty document")

    claim_id = _line(text, "Claim reference")
    policy_id = _line(text, "Policy number")
    claimant = _line(text, "Named insured")
    raw_loss = _line(text, "Loss type")
    loss_type = canonical_loss_type(raw_loss) if raw_loss else None
    description_match = re.search(
        r"(?is)^description:\s*(.+?)(?:\nfact tags:|\Z)",
        text,
        re.MULTILINE,
    )
    description = description_match.group(1).strip() if description_match else None
    tags_raw = _line(text, "Fact tags")
    if tags_raw is None or tags_raw.lower() == "none":
        fact_tags: list[str] = []
    else:
        fact_tags = [canonical_fact_tag(tag) for tag in tags_raw.split(",") if tag.strip()]

    features: dict[str, float] = {}
    for key in _FEATURE_KEYS:
        raw = _line(text, key)
        if raw is not None:
            try:
                features[key] = float(raw)
            except ValueError as exc:
                raise ExtractionError(f"invalid {key}: {raw}") from exc

    present = {
        "claim_id": claim_id,
        "policy_id": policy_id,
        "claimant_name": claimant,
        "loss_type": loss_type,
        "date_of_loss": _line(text, "Date of loss"),
        "date_reported": _line(text, "Date reported"),
        "description": description,
        "estimated_damage": _line(text, "Estimated damage USD"),
    }
    missing = [name for name in _REQUIRED if not present[name]]
    if missing:
        raise ExtractionError("missing fields: " + ", ".join(missing))

    assert claim_id and policy_id and claimant and loss_type and description
    confidence = 1.0
    if len(description) < 20:
        confidence = 0.85

    damage = _money(present["estimated_damage"])
    if damage is None:
        raise ExtractionError("missing estimated damage")

    return ClaimExtraction(
        claim_id=claim_id,
        policy_id=policy_id,
        claimant_name=claimant,
        loss_type=loss_type,
        date_of_loss=_iso_date(present["date_of_loss"], "date of loss"),
        date_reported=_iso_date(present["date_reported"], "date reported"),
        description=description,
        estimated_damage=damage,
        vehicle_acv=_money(_line(text, "Vehicle ACV USD")),
        stated_deductible=_money(_line(text, "Stated deductible USD")),
        injuries=_yes(_line(text, "Injuries reported")),
        police_report=_yes(_line(text, "Police report filed")),
        fact_tags=fact_tags,
        aftermarket_amount=_money(_line(text, "Aftermarket amount USD")) or 0.0,
        mold_amount=_money(_line(text, "Mold amount USD")) or 0.0,
        fraud_features=FraudFeatureInput.model_validate(features),
        extraction_confidence=confidence,
    )
