# Engineering notes

These notes describe the tree as it is. Thresholds, file roles, and test names come from the code. The eval and coverage figures are from one local run of the commands in the last section.

## Files

| Path | Role |
| --- | --- |
| `src/claims_agents/config.py` | `CLAIMS_` settings. Locates `data/` from the checkout. |
| `src/claims_agents/schemas.py` | Pydantic contracts for extraction, citations, fraud, payout, and the memo. Graph state stores the dumped dicts. |
| `src/claims_agents/errors.py` | Expected failures the API maps to HTTP status codes. |
| `src/claims_agents/pricing.py` | Token estimate (`chars / 4`, minimum 1) and a small price table. Unknown models cost 0. |
| `src/claims_agents/runtime.py` | Wires corpus, fraud bundle, LLM, retriever, scorer, and checkpointer. |
| `src/claims_agents/scenarios.py` | 17 labeled FNOL documents. Expected indemnities are literals on the scenario, not a call to `estimate_payout`. |
| `src/claims_agents/extraction/parser.py` | Labeled-FNOL parser used by the fake provider. |
| `src/claims_agents/extraction/render.py` | Renders a scenario into that labeled layout. |
| `src/claims_agents/extraction/normalize.py` | Replaces deductible and limit with the declarations and builds fraud features. |
| `src/claims_agents/coverage/engine.py` | Symbolic covered / excluded / partial decision and verbatim quotes. |
| `src/claims_agents/retrieval/corpus.py` | Loads `data/policies/corpus.json` (three policies). |
| `src/claims_agents/retrieval/hybrid.py` | BM25 plus 256-d hashed token vectors, fused with RRF (`k=60`). |
| `src/claims_agents/fraud/features.py` | Fourteen-column feature contract shared by training and scoring. |
| `src/claims_agents/fraud/rules.py` | SIU rules. The rule score is the max weight, not a sum. |
| `src/claims_agents/fraud/train.py` | Seeded synthetic generator, gradient boosting, `load_bundle`. |
| `src/claims_agents/fraud/score.py` | Publishes `min(1, max(model, rules))` and a band. |
| `src/claims_agents/fraud/explain.py` | One-feature ablation against the training-median vector. |
| `src/claims_agents/payout/estimator.py` | `max(0, min(basis, limit) - deductible)`. Reserve is indemnity, or indemnity × 1.15 when coverage confidence is below 0.8. |
| `src/claims_agents/graph/routing.py` | `choose_next` and the escalation thresholds. |
| `src/claims_agents/graph/memo.py` | Recommendation and final indemnity. The model does not choose them. |
| `src/claims_agents/graph/workflow.py` | LangGraph `StateGraph`, node bodies, interrupt, trace. |
| `src/claims_agents/guardrails/pii.py` | Email, SSN, phone, labeled DOB, Luhn cards. Findings are kind and count. |
| `src/claims_agents/guardrails/injection.py` | High-severity injection patterns, replaced with `[INJECTION_REMOVED]`. |
| `src/claims_agents/guardrails/narrative.py` | Drops clause-shaped tokens that are not on the decision's allow-list. `CLM-` and `POL-` stay. |
| `src/claims_agents/llm/fake.py` | Default provider. Extraction calls the parser. The memo is a fixed template. |
| `src/claims_agents/llm/providers.py` | OpenAI and Anthropic adapters. SDKs are imported only when constructed. |
| `src/claims_agents/llm/factory.py` | Selects the provider. Missing keys raise `LLMConfigError`. |
| `src/claims_agents/llm/structured.py` | JSON extraction, Pydantic validation, retries. |
| `src/claims_agents/observability/logging.py` | JSON logs. Extra fields are claim id, node, duration, and counts. |
| `src/claims_agents/observability/tracing.py` | In-process spans. Optional OpenTelemetry and a LangSmith-shaped hook. |
| `src/claims_agents/observability/usage.py` | Per-call and total token records. |
| `src/claims_agents/api/app.py` | FastAPI: health, samples, submit, get, list, resume. |
| `src/claims_agents/api/service.py` | Claim index, pre-invoke PII redaction, graph invoke, resume. |
| `src/claims_agents/api/schemas.py` | Request and response models. |
| `src/claims_agents/evals/harness.py` | Runs the 17 scenarios and the fraud holdout. Writes `reports/`. |
| `src/claims_agents/evals/metrics.py` | Faithfulness, retrieval recall, and the gate table. |
| `src/claims_agents/evals/__main__.py` | `python -m claims_agents.evals`. Exit 1 when a gate fails. |
| `ui/dashboard.py` | Streamlit reviewer. Reads the API; it does not score claims itself. |
| `scripts/train_fraud_model.py` | Calls `fraud.train.main`. |
| `data/policies/corpus.json` | `POL-AUTO-STD`, `POL-AUTO-PREM`, `POL-HO-STD`. |
| `data/samples/` | Rendered text for the 17 scenarios. |
| `data/fraud/` | Seeded `train.csv` and `test.csv`. |
| `data/models/fraud_gbm.joblib` | Saved model, baseline, and feature names. |
| `docs/adr/` | Records for routing, the fake LLM, retrieval, fraud fusion, interrupt, and guardrails. |
| `docs/architecture.md` | Walkthrough of nodes, state, and metrics. |

## Decisions and tradeoffs

The ADRs match the code on these points.

- **The supervisor is Python** (`choose_next`). A model cannot skip coverage or jump to the memo. The cost is that an odd file does not grow a new investigation step; it hits `human_review` or a halt memo. See ADR 0001.
- **The default model is fake.** CI has no API key and the golden numbers do not depend on a vendor. A prompt change the fake client ignores will not move the gate. Hosted providers exist as thin adapters and are not on the CI path. See ADR 0002.
- **Retrieval is local.** BM25 and hashed vectors are deterministic and need no network. They are lexical, not semantic embeddings. Compose starts Postgres with the pgvector image for an optional checkpointer. The decision path does not query it. See ADR 0003.
- **Fraud score is `max(model, rules)`.** A prior SIU referral (weight 0.80) escalates even when the model is calm. Stacked weak rules do not add. Explanations are ablations, so two features that matter only together can each look neutral. The training rows are synthetic. See ADR 0004.
- **Human review is `interrupt` inside the node.** The same function pauses and resumes. Memory checkpoints disappear when the process exits. See ADR 0005.
- **Money fields come from the declarations.** A stated deductible on the FNOL does not change the deductible. Quotes must be substrings of the clause. See ADR 0006.
- **Indemnity formula** is `max(0, min(basis, limit) - deductible)`, not `min(limit, max(0, basis - deductible))`. A loss above the limit pays `limit - deductible`. That is what the golden indemnities use.
- **State reducers.** `trace`, `usage`, and `errors` use `operator.add`. Other fields are overwritten by the node that owns them. The human-review node does not append to a reducer before `interrupt`, because LangGraph re-runs the node from the start on resume.

## Bugs fixed

1. **Zero confidence did not escalate.** `escalation_reasons` used `value or 1` for extraction and coverage confidence. `0` is falsy, so confidence `0` became `1` and skipped the 0.55 threshold. Numeric zero now stays zero. A missing value still defaults to `1` for confidence and `0` for score and damage.

2. **A bad human decision stuck the thread.** The node raised `ClaimStateError` after `interrupt()` had already consumed the resume. A later `approve` was rejected with the same error and the memo was never written. The node now calls `interrupt` again until the value is `approve` or `reject`. `ClaimService.resume` still raises `ClaimStateError` before `invoke` when the decision is anything else, so the API caller's bad value never reaches the node.

3. **Tight phone numbers were left in the text.** `(214)555-0198` and `+1(214)555-0199` failed the phone pattern because it required a separator after `)`. A parenthesized area code may now sit against the exchange. A bare area code still needs a separator, so a 10-digit amount is not treated as a phone number.

4. **A card plus extra digits survived.** `4111111111111111 123` was one 19-digit candidate, failed Luhn, and was kept. A separator-joined run now redacts a 13 to 19 digit prefix that passes Luhn. A single digit string is checked only at its full length, so `1234567890123456` is still not redacted.

5. **A failed submit burned the claim id.** The id was inserted into the in-memory index before `graph.invoke`. If invoke raised, the next submit of that id was `409 DuplicateClaimError`. The index entry is removed when invoke raises.

6. **PII counts were dropped on the API path.** The service redacted before `invoke` and threw away the findings. The graph redacted again, saw clean text, and stored an empty list. Kind and count from the first pass are kept when the second pass finds nothing. `ClaimResponse.pii_findings` returns them. The raw values are not in that list.

7. **A bad feature value crashed intake.** `policy_age_days: unknown` raised `ValueError` from `float()`. `FakeLLM` only catches `ExtractionError`, so the exception left the graph. The parser now raises `ExtractionError`. The fake client returns `{}`, structured validation fails, and the file halts with `extraction_failed` and recommendation `refer`.

8. **Non-ISO dates were accepted by the extraction schema.** `normalize_claim` then called `date.fromisoformat` outside the structured-output retry. `ClaimExtraction` now rejects a date that is not ISO `YYYY-MM-DD`.

9. **Model JSON could miss the coverage vocabulary.** The labeled parser already maps `Collision` and `crash`. A `ClaimExtraction` built from model JSON did not. `normalize_claim` now maps loss aliases and rewrites fact tags to lowercase underscore form (`Racing` → `racing`, `wear and tear` → `wear_and_tear`) before the engine looks them up.

10. **A reported date before the loss was silent.** `days_to_report` was clamped to 0 and confidence stayed high, so the file could be paid with no review. The claim records `reported_before_loss` and caps extraction confidence at 0.5, which is under the 0.55 escalation threshold. The feature stays clamped at 0 so a negative lag is not fed to the model.

11. **The Postgres checkpointer object was the context manager.** `PostgresSaver.from_conn_string` is a `@contextmanager`. Returning it skips `setup` and never opens the connection. `enter_checkpointer` enters it, calls `setup` when present, and keeps the context alive so garbage collection does not close the connection. This is covered with a fake context manager. CI still does not start Postgres.

12. **The eval report labeled metric values as gates.** `report["gates"]` copied the scores (routing `1.0`, not the threshold `0.90`). The markdown table also duplicated the thresholds and omitted `p95_latency_ms`, which the harness already computed. Both now read `GATES` from `metrics.py`. The markdown also prints holdout prevalence.

13. **A provider response with `usage is None` crashed the adapter.** OpenAI and Anthropic now fall back to `token_response` (the same character estimate the fake provider uses).

## Tests added

Regression tests, one or more per fix above:

| Test | Why |
| --- | --- |
| `test_phone_without_a_space_after_the_area_code_is_redacted` | Fix 3. |
| `test_card_with_trailing_digits_is_still_redacted` | Fix 4. The existing random-digit test still guards the false positive. |
| `test_supervisor_policy_unit` (zero-confidence case) | Fix 1. |
| `test_invalid_human_decision_stays_resumable` | Fix 2 at the graph. A `maybe` resume stays on `human_review`; a later approve pays CLM-1010 at 41500. |
| `test_bad_resume_does_not_consume_the_review` | Fix 2 at the service. `ClaimStateError` before invoke, then approve still works. |
| `test_failed_submit_releases_the_claim_id` | Fix 5. First invoke raises; the same id can be submitted again. |
| `test_submit_records_pii_counts_without_the_values` | Fix 6. CLM-1016 findings are email, phone, ssn, dob, and card, and the planted values are absent from findings and the claim dict. |
| `test_bad_feature_value_is_an_extraction_error` | Fix 7 at the parser. |
| `test_unparseable_feature_halts_instead_of_crashing` | Fix 7 through the graph: `refer` / `extraction_failed`. |
| `test_extraction_rejects_a_non_iso_date` | Fix 8. |
| `test_model_loss_labels_use_the_form_vocabulary` | Fix 9 for `Collision` and `Racing`. |
| `test_alias_and_spaced_tag_match_the_form` | Fix 9 for `crash` and `wear and tear`. |
| `test_reported_before_loss_caps_confidence` | Fix 10. |
| `test_postgres_saver_context_is_entered` | Fix 11. Also checks that a failed `setup` exits the context. |
| `test_gate_math_and_harness` | Fix 12. `report["gates"] == GATES`, and the markdown contains `p95_latency_ms`. |
| `test_factory_and_provider_adapters` | Fix 13. OpenAI stub with `usage=None` still returns a token count. |

Boundary tests that were not covering a new defect:

| Test | Why |
| --- | --- |
| `test_indemnity_is_capped_by_the_limit_and_floored_at_zero` | Basis above the limit pays `limit - deductible`. Basis below the deductible pays 0. |
| `test_aftermarket_at_the_sublimit_stays_covered` | The sublimit test is `>`, so 1000 stays covered and 1000.01 is partial. |
| `test_mold_without_a_sudden_discharge_is_excluded` | The golden mold file is the partial path (sudden discharge). The hard exclusion was untested. |
| `test_rule_boundaries` | Early high value is age `< 30`. Delayed report is days `> 45`. The equal boundaries must not fire. |
| `test_liability_deductible_is_zero` | Liability uses deductible 0 and grant `COV-LIAB`. No golden scenario is liability. |

This run of `pytest --cov=claims_agents --cov-report=term-missing --cov-fail-under=80` is 56 tests at 91.77% branch coverage.

## Not done

- The claim index is still process-local. With `CLAIMS_CHECKPOINTER=postgres`, a restart would keep the checkpoint and lose the index, so `get` returns 404. CI does not run that path. `enter_checkpointer` is not tested against a live server.
- A loss type with no grant is `excluded` at confidence 0.45, so it pauses for a person, but approve still denies. The reviewer cannot turn that file into a payment. ADR 0005 says approval is not a coverage override. That is consistent for a listed exclusion and awkward for an unrecognized peril.
- Reserve is not capped at the policy limit. Confidence below 0.8 multiplies indemnity by 1.15, and indemnity can already be `limit - deductible`.
- `load_bundle` retrains and rewrites `train.csv` and `test.csv` when the joblib is missing, unreadable, or fit on a different feature list (ADR 0004). A corrupt model on API startup rewrites the datasets.
- Rules `no_police_high_value` and `low_damage_attorney` have no direct test. `evals/__main__.py` (the exit-1 path) is not called by pytest; CI runs it as `make eval`.
- The dashboard does not show `pii_findings`.
- Bare 10-digit phone numbers are not redacted.
- The narrative sanitizer only matches uppercase clause tokens.
- Description prose does not become a fact tag. `flood` in the narrative does nothing unless the fact-tag line or the model JSON says so.
- Hashed vectors are not a semantic embedder, and the BM25 index is rebuilt on every search. Both are the local-index tradeoff in ADR 0003.
- The fraud high band uses `fraud_hitl_threshold`. The medium band is a fixed 0.35.

## Commands from this run

```text
ruff 0.17.1 check src tests ui scripts
ruff 0.17.1 format --check src tests ui scripts
mypy src
pytest --cov=claims_agents --cov-report=term-missing --cov-fail-under=80
python3 -m claims_agents.evals
```

Eval gate: pass. 17 cases. Holdout 1200 rows, prevalence 0.207, ROC AUC 0.9037, precision at 100 = 0.9700. Routing, coverage, citation faithfulness, payout accuracy, and retrieval recall were 1.0000. PII leaks were 0. Mean / p50 / p95 latency were 11.58 / 11.17 / 11.87 ms. Mean tokens were 266.6 in and 216.4 out. Cost was 0. Those latency figures are from this run; they move. The accuracy figures are the fake provider plus the saved model.
