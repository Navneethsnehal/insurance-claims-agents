from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from claims_agents.config import Settings
from claims_agents.coverage.engine import decide_coverage
from claims_agents.extraction.normalize import normalize_claim
from claims_agents.extraction.parser import parse_fnol
from claims_agents.fraud.explain import explain_contributions
from claims_agents.fraud.features import FEATURE_NAMES, vectorize
from claims_agents.fraud.rules import apply_rules, rule_score
from claims_agents.fraud.score import FraudScorer, fraud_band
from claims_agents.fraud.train import precision_at_k, synthesize, train_and_save
from claims_agents.payout.estimator import estimate_payout
from claims_agents.retrieval.corpus import load_corpus
from claims_agents.retrieval.hybrid import HybridRetriever
from claims_agents.scenarios import scenarios


def _settings() -> Settings:
    return Settings(_env_file=None)


def _claim_for(claim_id: str, policy_id: str | None = None) -> dict[str, object]:
    scenario = next(item for item in scenarios() if item.claim_id == claim_id)
    parsed = parse_fnol(scenario.document())
    corpus = load_corpus(_settings().corpus_path)
    return normalize_claim(
        parsed,
        claim_id=scenario.claim_id,
        policy_id=policy_id or scenario.policy_id,
        corpus=corpus,
    )


def test_hybrid_search_ranks_the_racing_exclusion_first() -> None:
    corpus = load_corpus(_settings().corpus_path)
    retriever = HybridRetriever(corpus.clauses)
    hits = retriever.search(
        "informal street race speed contest collision",
        "POL-AUTO-STD",
        k=3,
    )
    assert hits[0].clause_id == "EXC-RACING"
    assert all(hit.policy_id == "POL-AUTO-STD" for hit in hits)


def test_search_does_not_cross_policies() -> None:
    corpus = load_corpus(_settings().corpus_path)
    retriever = HybridRetriever(corpus.clauses)
    hits = retriever.search("collision deductible", "POL-HO-STD", k=5)
    assert hits
    assert all(hit.policy_id == "POL-HO-STD" for hit in hits)
    assert "COV-COLL" not in {hit.clause_id for hit in hits}


def test_empty_query_returns_a_bounded_list() -> None:
    corpus = load_corpus(_settings().corpus_path)
    hits = HybridRetriever(corpus.clauses).search("...", "POL-AUTO-STD", k=2)
    assert len(hits) == 2


def test_coverage_outcomes_match_the_book() -> None:
    corpus = load_corpus(_settings().corpus_path)
    retriever = HybridRetriever(corpus.clauses)
    settings = _settings()
    for scenario in scenarios():
        claim = _claim_for(scenario.claim_id)
        decision = decide_coverage(claim, corpus, retriever, settings)
        assert decision["outcome"] == scenario.expected_outcome, scenario.claim_id
        cited = {item["clause_id"] for item in decision["citations"]}
        assert set(scenario.expected_citations).issubset(cited), scenario.claim_id
        for citation in decision["citations"]:
            clause = corpus.require(scenario.policy_id).clause(citation["clause_id"])
            assert clause is not None
            assert citation["quote"] in clause.text
        estimate = estimate_payout(claim, decision)
        assert estimate["indemnity"] == scenario.expected_indemnity


def test_low_confidence_loads_the_reserve() -> None:
    estimate = estimate_payout(
        {"deductible": 500, "limit": 50000},
        {"outcome": "covered", "covered_basis": 2000, "confidence": 0.5},
    )
    assert estimate["indemnity"] == 1500
    assert estimate["reserve"] == 1725


def test_no_grant_for_homeowners_collision() -> None:
    claim = _claim_for("CLM-1001", policy_id="POL-HO-STD")
    corpus = load_corpus(_settings().corpus_path)
    decision = decide_coverage(claim, corpus, HybridRetriever(corpus.clauses), _settings())
    assert decision["outcome"] == "excluded"
    assert decision["confidence"] < 0.55


def test_rules_fire_on_prior_siu_and_stay_quiet_on_a_clean_file() -> None:
    noisy = _claim_for("CLM-1011")
    hits = apply_rules(noisy["fraud_features"])  # type: ignore[arg-type]
    assert rule_score(hits) >= 0.8
    assert "prior_siu" in {hit.code for hit in hits}
    clean = _claim_for("CLM-1001")
    assert apply_rules(clean["fraud_features"]) == []  # type: ignore[arg-type]


def test_fraud_band_thresholds() -> None:
    assert fraud_band(0.2, 0.65) == "low"
    assert fraud_band(0.5, 0.65) == "medium"
    assert fraud_band(0.8, 0.65) == "high"


def test_precision_at_k_and_explanations() -> None:
    labels = np.array([1, 1, 0, 0], dtype=np.int64)
    scores = np.array([0.9, 0.2, 0.8, 0.1], dtype=np.float64)
    assert precision_at_k(labels, scores, 2) == 0.5
    features, target = synthesize(300, seed=7, fraud_rate=0.3, label_noise=0.0)
    model = GradientBoostingClassifier(random_state=0, n_estimators=20, max_depth=2)
    model.fit(features, target)
    baseline = np.median(features, axis=0)
    rows = explain_contributions(model, features[0], baseline, FEATURE_NAMES)
    assert {row["feature"] for row in rows} == set(FEATURE_NAMES)
    mapping = {name: float(features[0][index]) for index, name in enumerate(FEATURE_NAMES)}
    vector = vectorize(mapping)
    scored = FraudScorer(model, baseline, 0.65).assess(
        {"fraud_features": {name: float(vector[index]) for index, name in enumerate(FEATURE_NAMES)}}
    )
    assert 0 <= scored["score"] <= 1
    assert scored["explanations"]


def test_training_is_deterministic(tmp_path: Path) -> None:
    first = train_and_save(
        tmp_path / "a.joblib",
        tmp_path / "train-a.csv",
        tmp_path / "test-a.csv",
        seed=11,
    )
    second = train_and_save(
        tmp_path / "b.joblib",
        tmp_path / "train-b.csv",
        tmp_path / "test-b.csv",
        seed=11,
    )
    assert first == second
    assert first["roc_auc"] >= 0.8
    assert (tmp_path / "a.metrics.json").exists()
