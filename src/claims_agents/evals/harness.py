"""Run the labeled book and the fraud holdout, then write a report."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from claims_agents.config import Settings
from claims_agents.evals.metrics import (
    GATES,
    citations_faithful,
    contains_planted_pii,
    coverage_correct,
    gate_failures,
    retrieval_hit_rate,
)
from claims_agents.fraud.features import FEATURE_NAMES
from claims_agents.fraud.train import precision_at_k
from claims_agents.graph.routing import worker_path
from claims_agents.graph.workflow import build_graph, initial_state, thread_config
from claims_agents.guardrails.pii import redact_pii
from claims_agents.runtime import Dependencies, build_dependencies
from claims_agents.scenarios import Scenario, scenarios

PRECISION_K = 100


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((pct / 100) * (len(ordered) - 1))
    return float(ordered[max(0, min(index, len(ordered) - 1))])


def _holdout_metrics(deps: Dependencies) -> dict[str, float]:
    path = deps.settings.fraud_test_path
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: list[list[float]] = []
        labels: list[int] = []
        for record in reader:
            if record is None:
                continue
            rows.append([float(record[name]) for name in FEATURE_NAMES])
            labels.append(int(float(record["is_fraud"])))
    features = np.asarray(rows, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(deps.scorer.model.predict_proba(features)[:, 1], dtype=np.float64)
    return {
        "fraud_roc_auc": float(roc_auc_score(target, scores)),
        "fraud_precision_at_k": precision_at_k(target, scores, PRECISION_K),
        "fraud_test_rows": float(len(target)),
        "fraud_test_prevalence": float(target.mean()),
    }


def _run_case(graph: Any, deps: Dependencies, scenario: Scenario) -> dict[str, Any]:
    document = redact_pii(scenario.document())[0]
    started = time.perf_counter()
    graph.invoke(
        initial_state(scenario.claim_id, scenario.policy_id, [document]),
        thread_config(scenario.claim_id),
    )
    latency_ms = (time.perf_counter() - started) * 1000
    snapshot = graph.get_state(thread_config(scenario.claim_id))
    values = dict(snapshot.values)
    pending = snapshot.next[0] if snapshot.next else None
    path = worker_path(list(values.get("trace") or []), str(pending) if pending else None)
    coverage = values.get("coverage") if isinstance(values.get("coverage"), dict) else None
    payout = values.get("payout") if isinstance(values.get("payout"), dict) else None
    policy = deps.corpus.require(scenario.policy_id)
    indemnity = float(payout["indemnity"]) if payout else None
    leaked = contains_planted_pii(values, scenario.planted_pii)
    usage = list(values.get("usage") or [])
    return {
        "claim_id": scenario.claim_id,
        "path": path,
        "routing_ok": path == list(scenario.expected_nodes),
        "coverage_ok": coverage_correct(
            coverage, scenario.expected_outcome, scenario.expected_citations
        ),
        "faithful": citations_faithful(coverage, policy),
        "retrieval_recall": retrieval_hit_rate(coverage),
        "payout_ok": indemnity is not None and abs(indemnity - scenario.expected_indemnity) < 0.01,
        "interrupted": pending is not None,
        "pii_leak": leaked,
        "latency_ms": latency_ms,
        "input_tokens": sum(int(row["input_tokens"]) for row in usage),
        "output_tokens": sum(int(row["output_tokens"]) for row in usage),
        "cost_usd": sum(float(row["cost_usd"]) for row in usage),
        "outcome": None if coverage is None else coverage.get("outcome"),
        "indemnity": indemnity,
        "fraud_score": _fraud_score(values),
    }


def _fraud_score(values: dict[str, Any]) -> float | None:
    fraud = values.get("fraud")
    if not isinstance(fraud, dict):
        return None
    score = fraud.get("score")
    return None if score is None else float(score)


def run_evaluation(
    settings: Settings | None = None,
    deps: Dependencies | None = None,
) -> dict[str, Any]:
    resolved = deps or build_dependencies(settings)
    graph = build_graph(resolved)
    cases = [_run_case(graph, resolved, scenario) for scenario in scenarios()]
    count = len(cases)
    holdout = _holdout_metrics(resolved)
    latencies = [float(case["latency_ms"]) for case in cases]
    metrics: dict[str, float] = {
        "routing_accuracy": sum(bool(case["routing_ok"]) for case in cases) / count,
        "coverage_accuracy": sum(bool(case["coverage_ok"]) for case in cases) / count,
        "citation_faithfulness": sum(bool(case["faithful"]) for case in cases) / count,
        "payout_accuracy": sum(bool(case["payout_ok"]) for case in cases) / count,
        "retrieval_recall": sum(float(case["retrieval_recall"]) for case in cases) / count,
        "pii_leaks": float(sum(bool(case["pii_leak"]) for case in cases)),
        "mean_latency_ms": sum(latencies) / count,
        "p50_latency_ms": _percentile(latencies, 50),
        "p95_latency_ms": _percentile(latencies, 95),
        "mean_input_tokens": sum(int(case["input_tokens"]) for case in cases) / count,
        "mean_output_tokens": sum(int(case["output_tokens"]) for case in cases) / count,
        "total_cost_usd": sum(float(case["cost_usd"]) for case in cases),
        **holdout,
    }
    failures = gate_failures(metrics)
    return {
        "passed": not failures,
        "failures": failures,
        "gates": dict(GATES),
        "metrics": metrics,
        "cases": cases,
        "n_cases": count,
        "precision_k": PRECISION_K,
        "llm_provider": resolved.settings.llm_provider,
    }


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        "# Evaluation report",
        "",
        f"Provider: `{report['llm_provider']}`. Cases: {report['n_cases']}. "
        f"Fraud holdout: {int(metrics['fraud_test_rows'])} rows.",
        "",
        "| Metric | Result | Gate |",
        "| --- | ---: | ---: |",
    ]
    for name, threshold in GATES.items():
        lines.append(f"| {name} | {metrics[name]:.4f} | {threshold:.2f} |")
    lines.extend(
        [
            f"| retrieval_recall | {metrics['retrieval_recall']:.4f} | report |",
            f"| pii_leaks | {int(metrics['pii_leaks'])} | 0 |",
            f"| fraud_test_prevalence | {metrics['fraud_test_prevalence']:.3f} | report |",
            f"| mean_latency_ms | {metrics['mean_latency_ms']:.2f} | report |",
            f"| p50_latency_ms | {metrics['p50_latency_ms']:.2f} | report |",
            f"| p95_latency_ms | {metrics['p95_latency_ms']:.2f} | report |",
            f"| mean_input_tokens | {metrics['mean_input_tokens']:.1f} | report |",
            f"| mean_output_tokens | {metrics['mean_output_tokens']:.1f} | report |",
            f"| total_cost_usd | {metrics['total_cost_usd']:.6f} | report |",
            "",
            "Gate result: **pass**." if report["passed"] else "Gate result: **fail**.",
            "",
        ]
    )
    if report["failures"]:
        lines.append("Failures:")
        lines.extend(f"- {failure}" for failure in report["failures"])
        lines.append("")
    lines.append("## Cases")
    lines.append("")
    lines.append("| Claim | Route | Coverage | Payout | Recall | Latency ms |")
    lines.append("| --- | --- | --- | --- | ---: | ---: |")
    for case in report["cases"]:
        lines.append(
            f"| {case['claim_id']} | {'ok' if case['routing_ok'] else 'miss'} | "
            f"{case['outcome']} | {'ok' if case['payout_ok'] else 'miss'} | "
            f"{case['retrieval_recall']:.2f} | {case['latency_ms']:.1f} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_report(report: dict[str, Any], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "eval_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (directory / "eval_report.md").write_text(render_markdown(report), encoding="utf-8")
