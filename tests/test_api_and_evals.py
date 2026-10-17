from pathlib import Path

from fastapi.testclient import TestClient

from claims_agents.api.app import create_app
from claims_agents.api.service import ClaimService
from claims_agents.config import Settings
from claims_agents.evals.harness import run_evaluation, write_report
from claims_agents.evals.metrics import gate_failures
from claims_agents.runtime import build_dependencies
from claims_agents.scenarios import scenarios


def _service() -> ClaimService:
    return ClaimService(build_dependencies(Settings(_env_file=None)))


def test_api_submit_review_and_resume() -> None:
    app = create_app(_service())
    straight = next(item for item in scenarios() if item.claim_id == "CLM-1001")
    review = next(item for item in scenarios() if item.claim_id == "CLM-1010")
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["vector_store"] == "local_hybrid"
        assert health.json()["policies"] == 3

        unknown = client.post(
            "/claims",
            json={"policy_id": "POL-MISSING", "documents": ["notice"]},
        )
        assert unknown.status_code == 400

        empty = client.post("/claims", json={"policy_id": "POL-AUTO-STD", "documents": []})
        assert empty.status_code == 422

        created = client.post(
            "/claims",
            json={
                "claim_id": straight.claim_id,
                "policy_id": straight.policy_id,
                "documents": [straight.document()],
            },
        )
        assert created.status_code == 201, created.text
        body = created.json()
        assert body["status"] == "completed"
        assert body["memo"]["recommendation"] == "pay"
        assert body["memo"]["final_indemnity"] == 5900
        assert body["usage_totals"]["input_tokens"] > 0
        assert body["usage_totals"]["cost_usd"] == 0

        duplicate = client.post(
            "/claims",
            json={
                "claim_id": straight.claim_id,
                "policy_id": straight.policy_id,
                "documents": [straight.document()],
            },
        )
        assert duplicate.status_code == 409

        assert client.get("/claims/CLM-DOES-NOT-EXIST").status_code == 404
        assert client.get(f"/claims/{straight.claim_id}").status_code == 200
        listed = client.get("/claims")
        assert listed.status_code == 200
        assert {row["claim_id"] for row in listed.json()} == {straight.claim_id}

        early = client.post(
            f"/claims/{straight.claim_id}/resume",
            json={"decision": "approve", "reviewer": "sam"},
        )
        assert early.status_code == 409

        paused = client.post(
            "/claims",
            json={
                "claim_id": review.claim_id,
                "policy_id": review.policy_id,
                "documents": [review.document()],
            },
        )
        assert paused.status_code == 201
        paused_body = paused.json()
        assert paused_body["status"] == "awaiting_human"
        assert "high_value" in paused_body["escalation_reasons"]
        assert paused_body["pending_review"]["recommended_indemnity"] == 41500

        resumed = client.post(
            f"/claims/{review.claim_id}/resume",
            json={"decision": "approve", "reviewer": "riley", "notes": "within authority"},
        )
        assert resumed.status_code == 200, resumed.text
        final = resumed.json()
        assert final["status"] == "completed"
        assert final["memo"]["recommendation"] == "pay"
        assert final["memo"]["human_decision"]["reviewer"] == "riley"

        samples = client.get("/samples")
        assert samples.status_code == 200
        assert len(samples.json()) == len(scenarios())


def test_gate_math_and_harness(tmp_path: Path) -> None:
    bad = {
        "routing_accuracy": 0.5,
        "coverage_accuracy": 1,
        "citation_faithfulness": 1,
        "payout_accuracy": 1,
        "fraud_roc_auc": 1,
        "fraud_precision_at_k": 1,
        "pii_leaks": 2,
    }
    failures = gate_failures(bad)
    assert any(item.startswith("routing_accuracy") for item in failures)
    assert any(item.startswith("pii_leaks") for item in failures)

    report = run_evaluation(Settings(_env_file=None))
    assert report["passed"], report["failures"]
    assert report["metrics"]["pii_leaks"] == 0
    write_report(report, tmp_path)
    assert (tmp_path / "eval_report.json").exists()
    assert "routing_accuracy" in (tmp_path / "eval_report.md").read_text(encoding="utf-8")
