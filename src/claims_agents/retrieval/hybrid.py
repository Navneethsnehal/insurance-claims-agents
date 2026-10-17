"""Hybrid clause retrieval: BM25 plus hashed embeddings, fused with RRF.

Embeddings are hashed token vectors, not a hosted model. They are
deterministic, need no network, and are enough for a form library of this
size. See docs/adr/0003-local-hybrid-retrieval.md.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from claims_agents.retrieval.corpus import Clause

_TOKEN = re.compile(r"[a-z0-9]+")
_DIMS = 256
_RRF_K = 60


@dataclass(frozen=True)
class ScoredClause:
    clause_id: str
    policy_id: str
    title: str
    text: str
    score: float
    bm25_rank: int
    vector_rank: int


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def embed(text: str, dims: int = _DIMS) -> np.ndarray:
    vector = np.zeros(dims, dtype=np.float64)
    for token in tokenize(text):
        digest = hashlib.sha256(token.encode()).digest()
        index = int.from_bytes(digest[:4], "little") % dims
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def _index_text(clause: Clause) -> str:
    return " ".join([clause.title, clause.text, " ".join(clause.keywords), " ".join(clause.perils)])


def _ranks(scores: np.ndarray) -> dict[int, int]:
    order = np.argsort(-scores)
    return {int(index): rank for rank, index in enumerate(order, start=1)}


class HybridRetriever:
    """Search clauses that belong to one policy."""

    def __init__(self, clauses: list[Clause]) -> None:
        self._clauses = list(clauses)

    def search(self, query: str, policy_id: str, k: int) -> list[ScoredClause]:
        subset = [clause for clause in self._clauses if clause.policy_id == policy_id]
        if not subset or k <= 0:
            return []
        documents = [tokenize(_index_text(clause)) or ["empty"] for clause in subset]
        query_tokens = tokenize(query)
        if not query_tokens:
            return [
                ScoredClause(
                    clause_id=clause.clause_id,
                    policy_id=clause.policy_id,
                    title=clause.title,
                    text=clause.text,
                    score=0.0,
                    bm25_rank=index + 1,
                    vector_rank=index + 1,
                )
                for index, clause in enumerate(subset[:k])
            ]

        bm25 = BM25Okapi(documents)
        bm25_scores = np.asarray(bm25.get_scores(query_tokens), dtype=np.float64)
        query_vector = embed(query)
        doc_matrix = np.vstack([embed(_index_text(clause)) for clause in subset])
        vector_scores = doc_matrix @ query_vector
        bm25_rank = _ranks(bm25_scores)
        vector_rank = _ranks(vector_scores)
        fused: list[tuple[float, int]] = []
        for index in range(len(subset)):
            score = (1.0 / (_RRF_K + bm25_rank[index])) + (1.0 / (_RRF_K + vector_rank[index]))
            fused.append((score, index))
        fused.sort(key=lambda item: item[0], reverse=True)
        results: list[ScoredClause] = []
        for score, index in fused[:k]:
            clause = subset[index]
            results.append(
                ScoredClause(
                    clause_id=clause.clause_id,
                    policy_id=clause.policy_id,
                    title=clause.title,
                    text=clause.text,
                    score=round(float(score), 6),
                    bm25_rank=bm25_rank[index],
                    vector_rank=vector_rank[index],
                )
            )
        return results
