"""Load the versioned policy-form corpus."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from claims_agents.errors import PolicyNotFoundError

ClauseKind = Literal["coverage", "exclusion", "condition", "limit"]


class Clause(BaseModel):
    model_config = ConfigDict(frozen=True)

    clause_id: str
    policy_id: str
    kind: ClauseKind
    title: str
    text: str
    perils: list[str] = []
    keywords: list[str] = []


class Policy(BaseModel):
    model_config = ConfigDict(frozen=True)

    policy_id: str
    product: Literal["personal_auto", "homeowners"]
    name: str
    terms: dict[str, float]
    clauses: list[Clause]

    def clause(self, clause_id: str) -> Clause | None:
        for item in self.clauses:
            if item.clause_id == clause_id:
                return item
        return None


class CorpusFile(BaseModel):
    policies: list[Policy]


class Corpus:
    """In-memory form library keyed by policy id."""

    def __init__(self, policies: list[Policy]) -> None:
        self.policies = {policy.policy_id: policy for policy in policies}
        self.clauses = [clause for policy in policies for clause in policy.clauses]

    def get(self, policy_id: str) -> Policy | None:
        return self.policies.get(policy_id)

    def require(self, policy_id: str) -> Policy:
        policy = self.get(policy_id)
        if policy is None:
            raise PolicyNotFoundError(f"unknown policy {policy_id}")
        return policy


def load_corpus(path: Path) -> Corpus:
    payload = json.loads(path.read_text(encoding="utf-8"))
    parsed = CorpusFile.model_validate(payload)
    return Corpus(parsed.policies)


@lru_cache(maxsize=4)
def cached_corpus(path: str) -> Corpus:
    return load_corpus(Path(path))
