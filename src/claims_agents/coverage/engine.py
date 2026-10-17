"""Decide covered, excluded, or partial from the policy form.

The model may explain a decision. It does not choose one. Grants, exclusions,
and sublimits are code so an uploaded document cannot talk the system out of
an exclusion. Retrieval still has to surface the clauses we cite; clauses
that miss the top-k are marked ``lookup`` and counted against retrieval recall.
"""

from __future__ import annotations

from typing import Any

from claims_agents.config import Settings
from claims_agents.retrieval.corpus import Corpus, Policy
from claims_agents.retrieval.hybrid import HybridRetriever

_AUTO_GRANTS = {
    "collision": "COV-COLL",
    "comprehensive": "COV-COMP",
    "theft": "COV-COMP",
    "vandalism": "COV-COMP",
    "weather": "COV-COMP",
    "animal": "COV-COMP",
    "fire": "COV-COMP",
    "liability": "COV-LIAB",
}
_HOME_GRANTS = {
    "water": "COV-WATER",
    "fire": "COV-FIRE",
    "theft": "COV-PP",
    "vandalism": "COV-PP",
}
_HARD_TAGS = {
    "intentional": "EXC-INTENT",
    "racing": "EXC-RACING",
    "wear_and_tear": "EXC-WEAR",
    "business_use": "EXC-BUSINESS",
    "flood": "EXC-FLOOD",
    "earth_movement": "EXC-EARTH",
    "neglect": "EXC-NEGLECT",
}


def build_query(claim: dict[str, Any]) -> str:
    loss = str(claim["loss_type"])
    tags = " ".join(claim.get("fact_tags") or [])
    description = str(claim.get("description") or "")
    return f"{loss} {loss} {loss} {tags} {description} deductible limit exclusion"


def _grant_for(policy: Policy, loss_type: str) -> str | None:
    table = _AUTO_GRANTS if policy.product == "personal_auto" else _HOME_GRANTS
    return table.get(loss_type)


def _unique(ids: list[str]) -> list[str]:
    seen: list[str] = []
    for clause_id in ids:
        if clause_id not in seen:
            seen.append(clause_id)
    return seen


def verbatim_quote(text: str, limit: int = 280) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip()


def decide_coverage(
    claim: dict[str, Any],
    corpus: Corpus,
    retriever: HybridRetriever,
    settings: Settings,
) -> dict[str, Any]:
    policy = corpus.require(str(claim["policy_id"]))
    hits = retriever.search(build_query(claim), policy.policy_id, settings.retrieval_k)
    retrieved_ids = [hit.clause_id for hit in hits]
    retrieved = set(retrieved_ids)
    tags = {str(tag) for tag in claim.get("fact_tags") or []}
    loss = str(claim["loss_type"])

    hard = [_HARD_TAGS[tag] for tag in _HARD_TAGS if tag in tags]
    if loss == "theft" and not claim.get("police_report"):
        hard.append("COND-THEFT-POLICE")
    if "mold" in tags and "sudden_discharge" not in tags:
        hard.append("EXC-MOLD")
    partial: list[str] = []
    if "mold" in tags and "sudden_discharge" in tags and float(claim.get("mold_amount") or 0) > 0:
        partial.append("EXC-MOLD")
    if float(claim.get("aftermarket_amount") or 0) > settings.aftermarket_sublimit:
        partial.append("EXC-AFTERMARKET")
    hard = _unique(hard)
    partial = [clause_id for clause_id in _unique(partial) if clause_id not in hard]
    grant_id = _grant_for(policy, loss)
    damage = float(claim["estimated_damage"])

    selected: list[tuple[str, str]]
    if hard:
        outcome = "excluded"
        basis = 0.0
        confidence = 0.93
        selected = [
            (clause_id, "condition" if clause_id.startswith("COND") else "exclusion")
            for clause_id in hard
        ]
        if grant_id:
            selected.append((grant_id, "grant_overridden"))
    elif partial:
        outcome = "partial"
        basis = damage
        if "EXC-AFTERMARKET" in partial:
            excess = float(claim.get("aftermarket_amount") or 0) - settings.aftermarket_sublimit
            basis -= max(0.0, excess)
        if "EXC-MOLD" in partial:
            basis -= float(claim.get("mold_amount") or 0)
        basis = max(0.0, basis)
        confidence = 0.86
        selected = []
        if grant_id:
            selected.append((grant_id, "grant"))
        selected.extend((clause_id, "exclusion") for clause_id in partial)
        selected.append(("LIM-TERMS", "limit"))
    elif grant_id is None:
        outcome = "excluded"
        basis = 0.0
        confidence = 0.45
        selected = []
    else:
        outcome = "covered"
        basis = damage
        confidence = 0.92
        selected = [(grant_id, "grant"), ("LIM-TERMS", "limit")]

    if claim.get("policy_mismatch"):
        confidence = min(confidence, 0.5)

    citations: list[dict[str, str]] = []
    for clause_id, role in selected:
        clause = policy.clause(clause_id)
        if clause is None:
            continue
        citations.append(
            {
                "clause_id": clause.clause_id,
                "title": clause.title,
                "role": role,
                "quote": verbatim_quote(clause.text),
                "source": "retrieval" if clause_id in retrieved else "lookup",
            }
        )
    rationale_bits = [f"The {loss} loss is {outcome} under {policy.policy_id}."]
    rationale_bits.extend(
        f'{item["clause_id"]} ({item["role"]}): "{item["quote"]}"' for item in citations
    )
    return {
        "outcome": outcome,
        "citations": citations,
        "rationale": " ".join(rationale_bits),
        "covered_basis": round(basis, 2),
        "confidence": round(confidence, 2),
        "retrieved_ids": retrieved_ids,
    }
