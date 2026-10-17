# Insurance claims agents

A supervisor-routed LangGraph service that turns a first notice of loss into a cited coverage decision, a fraud score with reasons, a reserve, and a memo a reviewer can approve or reject.

The routing policy, the form, and the fraud model are code. The language model extracts the FNOL and writes the paragraph. Tests and CI run with a deterministic fake provider, so the numbers below do not depend on an API key.

[![CI](https://github.com/Navneethsnehal/insurance-claims-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/Navneethsnehal/insurance-claims-agents/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-3776AB)
![coverage](https://img.shields.io/badge/coverage-90.7%25-2EA44F)
![license](https://img.shields.io/badge/license-MIT-blue)

```mermaid
flowchart TD
    start([START]) --> guardrails
    guardrails --> supervisor
    supervisor -->|no claim| intake
    intake --> supervisor
    supervisor -->|no coverage| coverage
    coverage --> supervisor
    supervisor -->|no fraud| fraud
    fraud --> supervisor
    supervisor -->|no payout| payout
    payout --> supervisor
    supervisor -->|threshold tripped| human
    human[human_review] -->|interrupt, then resume| supervisor
    supervisor -->|decision ready| memo
    memo --> finish([END])
```

## Features

- **Intake.** Labeled FNOL, police-report, and estimate text become a typed `ClaimExtraction`. Deductible and limit come from the policy declarations, not from the document.
- **Coverage.** Hybrid retrieval (BM25 + hashed embeddings, reciprocal rank fusion) over three sample forms. The decision is symbolic: covered, excluded, or partial, with verbatim clause quotes.
- **Fraud.** A seeded gradient-boosting model on 14 synthetic features, fused with SIU rules by `max(model, rules)`. Explanations are one-feature ablations against the training median.
- **Payout.** Indemnity is `max(0, min(basis, limit) - deductible)`. Confidence below `0.8` loads the reserve by 15%.
- **Supervisor.** A deterministic policy. High fraud (`≥ 0.65`), high value (`≥ $25,000`), low confidence (`< 0.55`), or high-severity prompt injection pauses the thread with `interrupt`.
- **Guardrails.** PII redaction, injection stripping, structured-output retries, narrative sanitizing of clause ids.
- **Providers.** `fake` (default), OpenAI, or Anthropic via `CLAIMS_LLM_PROVIDER`.
- **Service.** FastAPI: submit, status, resume, health, sample book. Streamlit reviewer for the trace, memo, citations, and fraud reasons.
- **Evals.** A 17-claim golden book plus a 1,200-row fraud holdout. CI fails the job when a gate misses.

## Quickstart

Python 3.11 or newer.

```bash
python -m pip install -e ".[dev,ui]"
make test
make eval
make api
```

The API listens on `http://127.0.0.1:8000`. In another shell:

```bash
make ui
```

The reviewer listens on `http://127.0.0.1:8501` and reads `CLAIMS_API_URL` (default `http://localhost:8000`).

Train the fraud artifact again (seed 42; the committed model was produced this way):

```bash
make train
```

### Docker Compose

```bash
docker compose up --build
```

That starts the API on port 8000, the reviewer on port 8501, and Postgres 16 with pgvector on port 5432. The API container sets `CLAIMS_CHECKPOINTER=memory` and answers retrieval from the local hybrid index. Pointing the checkpointer at Postgres requires the `postgres` extra and `CLAIMS_DATABASE_URL`; that path is not what CI runs. See [ADR 0003](docs/adr/0003-local-hybrid-retrieval.md).

Copy `.env.example` to `.env` to switch the chat provider. `fake` needs no keys.

## Example

`POST /claims` with the synthetic rear-end collision in [`data/samples/CLM-1001.txt`](data/samples/CLM-1001.txt):

```bash
curl -s -X POST http://127.0.0.1:8000/claims \
  -H 'content-type: application/json' \
  -d "{\"claim_id\":\"CLM-1001\",\"policy_id\":\"POL-AUTO-STD\",\"documents\":[$(python -c 'import json,pathlib; print(json.dumps(pathlib.Path("data/samples/CLM-1001.txt").read_text()))')]}"
```

The fake provider returns a completed file. Figures below are from a local run of `ClaimService.submit` on that document.

| Field | Value |
| --- | --- |
| Status | `completed` |
| Loss | collision, estimated damage `6400` |
| Deductible / limit | `500` / `50000` from `POL-AUTO-STD` |
| Coverage | `covered`, confidence `0.92`, basis `6400` |
| Citations | `COV-COLL`, `LIM-TERMS` (both `retrieval`, quotes are substrings of the form) |
| Fraud | score `0.0642` (model `0.0642`, rules `0`), band `low`, no flags |
| Indemnity / reserve | `5900` / `5900` |
| Recommendation | `pay` |
| Escalation | none |
| Trace | `guardrails → supervisor → intake → supervisor → coverage → supervisor → fraud → supervisor → payout → supervisor → memo` |
| Usage | 281 input tokens, 223 output tokens, cost `$0` |

```json
{
  "claim_id": "CLM-1001",
  "policy_id": "POL-AUTO-STD",
  "status": "completed",
  "coverage": {
    "outcome": "covered",
    "confidence": 0.92,
    "covered_basis": 6400.0,
    "citations": [
      {
        "clause_id": "COV-COLL",
        "source": "retrieval",
        "quote": "We pay for direct and accidental collision loss to a covered auto when it hits another vehicle or object, subject to the deductible and the actual cash value li"
      },
      {
        "clause_id": "LIM-TERMS",
        "source": "retrieval",
        "quote": "The most we pay for a covered auto loss is the limit or actual cash value, whichever is less, minus a 500 dollar collision deductible or a 250 dollar comprehens"
      }
    ]
  },
  "fraud": { "score": 0.0642, "model_score": 0.0642, "rule_score": 0.0, "band": "low", "flags": [] },
  "payout": { "indemnity": 5900.0, "reserve": 5900.0, "deductible": 500.0, "limit": 50000.0 },
  "memo": {
    "recommendation": "pay",
    "final_indemnity": 5900.0,
    "narrative": "Claim CLM-1001 is covered. The recommendation is pay with indemnity 5900.00. Fraud band is low."
  }
}
```

A `$42,000` collision on the same form (`CLM-1010` in the sample book) stops in `awaiting_human` with reason `high_value` and a recommended indemnity of `41500`. `POST /claims/CLM-1010/resume` with `{"decision":"approve","reviewer":"riley"}` writes the memo. `reject` forces indemnity `0`.

## Evaluation

`make eval` writes [`reports/eval_report.md`](reports/eval_report.md). The run below used the fake provider, 17 golden claims, and the 1,200-row fraud holdout. Gate result: **pass**.

| Metric | Result | Gate |
| --- | ---: | ---: |
| Routing accuracy | 1.0000 | 0.90 |
| Coverage accuracy | 1.0000 | 0.85 |
| Citation faithfulness | 1.0000 | 0.90 |
| Payout accuracy | 1.0000 | 0.90 |
| Fraud ROC AUC | 0.9037 | 0.80 |
| Fraud precision@100 | 0.9700 | 0.70 |
| Retrieval recall | 1.0000 | reported |
| PII leaks | 0 | 0 |
| Mean latency | 12.15 ms | reported |
| p50 / p95 latency | 11.33 ms / 14.68 ms | reported |
| Mean tokens (in / out) | 266.5 / 216.4 | reported |
| Total cost | $0.000000 | reported |

Fraud holdout prevalence on that split is `0.207` (`1200` rows). The saved model file reports the same AUC and precision@100 in `data/models/fraud_gbm.metrics.json`.

Faithfulness requires every quote to be an exact substring of the cited clause. Retrieval recall is the fraction of citations that came from the top-k search rather than a direct lookup. PII leaks are planted values that survived into the checkpoint; any leak fails the run.

The golden indemnities are hand-computed literals. They are not the output of the payout function copied back into the label.

## Project structure

```
src/claims_agents/     library: graph, agents, API, evals
tests/                 unit tests and graph integration tests
data/policies/         POL-AUTO-STD, POL-AUTO-PREM, POL-HO-STD
data/samples/          rendered FNOL text for the golden book
data/fraud/            seeded train.csv and test.csv
data/models/           fraud_gbm.joblib and holdout metrics
ui/dashboard.py        Streamlit reviewer
docs/architecture.md   component notes and the graph
docs/adr/              design records
reports/               last eval run
```

`make ci` runs ruff, mypy, pytest with a branch-coverage floor of 80%, and the eval gate. The latest local suite is 39 tests at 90.7% branch coverage.

## Design decisions

The short version. The records are in [`docs/adr/`](docs/adr/).

- The supervisor is a policy function. A model does not pick the next agent ([0001](docs/adr/0001-supervisor-is-policy.md)).
- The default LLM is a deterministic fake so CI is hermetic ([0002](docs/adr/0002-deterministic-fake-llm.md)).
- Retrieval is a local hybrid index. Postgres in Compose is the optional checkpointer neighbor, not the index the decision reads ([0003](docs/adr/0003-local-hybrid-retrieval.md)).
- The fraud score is `max(model, rules)`. Explanations are ablations, not TreeSHAP ([0004](docs/adr/0004-fraud-fusion-and-ablation.md)).
- Human review is `interrupt` inside the node, resumed with `Command` ([0005](docs/adr/0005-hitl-interrupt.md)).
- PII and injection are stripped before the model. Citations are verbatim. Contract terms come from the declarations ([0006](docs/adr/0006-guardrails-and-citations.md)).

A longer walkthrough of the nodes, the state reducers, and the metric definitions is in [`docs/architecture.md`](docs/architecture.md).

## Roadmap

- Exercise `PostgresSaver` in Compose and keep human reviews across a process restart.
- Swap the hashed vectors for a real embedding model behind the same citation contract when the form library outgrows lexical fusion.
- Optional TreeSHAP next to the ablation explanations, without changing the score contract.
- OCR for photo estimates, with the same extraction schema and the same refusal to trust a stated deductible.
- Threshold calibration on a book that is not drawn from the training generator.

## License

[MIT](LICENSE). Copyright 2026 Navneeth Jadav.
