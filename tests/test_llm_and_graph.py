import json
import logging
from types import SimpleNamespace
from typing import Any

import pytest
from langgraph.types import Command
from pydantic import BaseModel

from claims_agents.config import Settings, get_settings
from claims_agents.errors import ClaimStateError, LLMConfigError, StructuredOutputError
from claims_agents.graph.routing import choose_next, escalation_reasons, worker_path
from claims_agents.graph.workflow import build_graph, initial_state, read_interrupt, thread_config
from claims_agents.llm.base import token_response
from claims_agents.llm.factory import build_llm
from claims_agents.llm.providers import AnthropicProvider, OpenAIProvider
from claims_agents.llm.structured import complete_structured, extract_json
from claims_agents.observability.logging import JsonFormatter
from claims_agents.observability.tracing import (
    SpanLedger,
    SpanRecord,
    TracingHooks,
    langsmith_metadata,
)
from claims_agents.pricing import cost_usd
from claims_agents.runtime import build_checkpointer, build_dependencies
from claims_agents.scenarios import scenarios


class _Echo(BaseModel):
    value: int


class _Flaky:
    model = "fake-claims-v1"

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def complete(self, *, system: str, user: str) -> Any:
        del system, user
        self.calls += 1
        text = "not-json" if self.calls <= self.failures else '{"value": 3}'
        return token_response(text, self.model, "s", "u", 1.0)


def test_extract_json_strips_fences() -> None:
    assert json.loads(extract_json('```json\n{"value": 1}\n```')) == {"value": 1}


def test_structured_output_retries_then_accepts() -> None:
    llm = _Flaky(failures=2)
    result = complete_structured(llm, system="s", user="u", schema=_Echo, retries=2)
    assert result.parsed.value == 3
    assert result.attempts == 3


def test_structured_output_gives_up() -> None:
    llm = _Flaky(failures=5)
    with pytest.raises(StructuredOutputError):
        complete_structured(llm, system="s", user="u", schema=_Echo, retries=1)


def test_factory_and_provider_adapters() -> None:
    settings = Settings(_env_file=None, llm_provider="fake")
    assert build_llm(settings).model == "fake-claims-v1"
    with pytest.raises(LLMConfigError):
        build_llm(Settings(_env_file=None, llm_provider="openai"))
    with pytest.raises(LLMConfigError):
        build_llm(Settings(_env_file=None, llm_provider="anthropic"))

    usage = SimpleNamespace(
        prompt_tokens=20,
        completion_tokens=4,
        input_tokens=20,
        output_tokens=4,
    )
    openai_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))],
        usage=usage,
    )

    class _Completions:
        def create(self, **kwargs: object) -> SimpleNamespace:
            assert kwargs["temperature"] == 0
            return openai_response

    openai_client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    openai_result = OpenAIProvider("key", "gpt-4o-mini", client=openai_client).complete(
        system="sys", user="user"
    )
    assert openai_result.input_tokens == 20
    assert openai_result.cost_usd == cost_usd("gpt-4o-mini", 20, 4)
    assert openai_result.cost_usd > 0

    anthropic_response = SimpleNamespace(
        content=[SimpleNamespace(text='{"ok": true}')],
        usage=usage,
    )

    class _Messages:
        def create(self, **kwargs: object) -> SimpleNamespace:
            del kwargs
            return anthropic_response

    anthropic_client = SimpleNamespace(messages=_Messages())
    anthropic_result = AnthropicProvider(
        "key",
        "claude-3-5-sonnet-latest",
        client=anthropic_client,
    ).complete(system="sys", user="user")
    assert anthropic_result.output_tokens == 4
    assert cost_usd("unknown-model", 10, 10) == 0


def test_checkpointer_selection() -> None:
    memory = build_checkpointer(Settings(_env_file=None, checkpointer="memory"))
    assert memory is not None
    with pytest.raises(Exception, match="DATABASE_URL"):
        build_checkpointer(Settings(_env_file=None, checkpointer="postgres"))


def test_tracing_hooks_and_json_logs() -> None:
    class _Sink:
        def __init__(self) -> None:
            self.runs: list[dict[str, object]] = []

        def create_run(self, **payload: object) -> None:
            self.runs.append(payload)

    sink = _Sink()
    ledger = SpanLedger(TracingHooks(otel_enabled=True, langsmith_client=sink))
    ledger.record("intake", 1.25, {"claim_id": "CLM-1", "raw": {"skip": True}})
    assert ledger.spans[0].attributes == {"claim_id": "CLM-1"}
    assert sink.runs[0]["name"] == "intake"
    meta = langsmith_metadata(SpanRecord("coverage", 2.0, {"claim_id": "CLM-1"}))
    assert meta["run_type"] == "chain"
    formatter = JsonFormatter()
    record = logging.LogRecord(
        "claims_agents", logging.INFO, __file__, 1, "documents screened", (), None
    )
    record.claim_id = "CLM-1"  # type: ignore[attr-defined]
    record.pii_count = 2  # type: ignore[attr-defined]
    rendered = formatter.format(record)
    assert "CLM-1" in rendered
    assert "123-45-6789" not in rendered


def test_supervisor_policy_unit() -> None:
    settings = Settings(_env_file=None)
    assert choose_next({"halt_reason": "no_documents"}) == "memo"
    assert choose_next({}) == "intake"
    state = {
        "claim": {"estimated_damage": 1000, "extraction_confidence": 1},
        "coverage": {"confidence": 0.9},
        "fraud": {"score": 0.1},
        "payout": {"indemnity": 1},
        "needs_human": False,
    }
    assert choose_next(state) == "memo"
    state["needs_human"] = True
    assert choose_next(state) == "human_review"
    state["human_decision"] = {"decision": "approve"}
    assert choose_next(state) == "memo"
    reasons = escalation_reasons(
        {
            "injection_findings": [{"severity": "high", "code": "role_override"}],
            "fraud": {"score": 0.9},
            "claim": {"estimated_damage": 40000, "extraction_confidence": 0.2},
            "coverage": {"confidence": 0.4},
        },
        settings,
    )
    assert reasons == [
        "prompt_injection",
        "fraud_score",
        "high_value",
        "low_extraction_confidence",
        "low_coverage_confidence",
    ]
    assert worker_path([{"node": "guardrails"}, {"node": "supervisor"}], "intake") == [
        "guardrails",
        "intake",
    ]


def test_graph_pays_denies_and_reviews() -> None:
    deps = build_dependencies(Settings(_env_file=None, data_dir=None))
    graph = build_graph(deps)

    def run(scenario_id: str) -> dict[str, Any]:
        scenario = next(item for item in scenarios() if item.claim_id == scenario_id)
        graph.invoke(
            initial_state(scenario.claim_id, scenario.policy_id, [scenario.document()]),
            thread_config(scenario.claim_id),
        )
        snapshot = graph.get_state(thread_config(scenario.claim_id))
        return {"values": dict(snapshot.values), "snapshot": snapshot}

    paid = run("CLM-1001")
    assert paid["values"]["memo"]["recommendation"] == "pay"
    assert paid["values"]["memo"]["final_indemnity"] == 5900
    assert paid["snapshot"].next == ()
    assert "EXC-RACING" not in paid["values"]["memo"]["narrative"]

    denied = run("CLM-1005")
    assert denied["values"]["coverage"]["outcome"] == "excluded"
    assert denied["values"]["memo"]["recommendation"] == "deny"

    partial = run("CLM-1008")
    assert partial["values"]["memo"]["recommendation"] == "partial_pay"
    assert partial["values"]["payout"]["indemnity"] == 6000

    paused = run("CLM-1010")
    assert paused["snapshot"].next == ("human_review",)
    pending = read_interrupt(paused["snapshot"])
    assert pending is not None
    assert "high_value" in pending["reasons"]

    rejected = graph.invoke(
        Command(resume={"decision": "reject", "reviewer": "sam", "notes": "hold"}),
        thread_config("CLM-1010"),
    )
    assert rejected["memo"]["recommendation"] == "deny"
    assert rejected["memo"]["final_indemnity"] == 0
    assert rejected["human_decision"]["reviewer"] == "sam"

    injection = run("CLM-1015")
    assert injection["values"]["coverage"]["outcome"] == "excluded"
    assert injection["snapshot"].next == ("human_review",)
    assert any(item["severity"] == "high" for item in injection["values"]["injection_findings"])
    approved = graph.invoke(
        Command(resume={"decision": "approve", "reviewer": "sam", "notes": "agree"}),
        thread_config("CLM-1015"),
    )
    assert approved["memo"]["recommendation"] == "deny"
    assert approved["memo"]["workflow_status"] == "completed"

    planted = run("CLM-1016")
    blob = json.dumps(planted["values"])
    for needle in ("123-45-6789", "morgan.lee@example.com", "4111111111111111"):
        assert needle not in blob


def test_invalid_human_decision_is_rejected() -> None:
    deps = build_dependencies(Settings(_env_file=None))
    graph = build_graph(deps)
    scenario = next(item for item in scenarios() if item.claim_id == "CLM-1010")
    graph.invoke(
        initial_state("CLM-BAD", scenario.policy_id, [scenario.document()]),
        thread_config("CLM-BAD"),
    )
    with pytest.raises(ClaimStateError):
        graph.invoke(Command(resume={"decision": "maybe"}), thread_config("CLM-BAD"))


def test_empty_submission_refers() -> None:
    deps = build_dependencies(Settings(_env_file=None))
    graph = build_graph(deps)
    graph.invoke(initial_state("CLM-EMPTY", "POL-AUTO-STD", ["  "]), thread_config("CLM-EMPTY"))
    values = graph.get_state(thread_config("CLM-EMPTY")).values
    assert values["memo"]["recommendation"] == "refer"
    assert values["halt_reason"] == "no_documents"


def test_get_settings_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAIMS_LLM_PROVIDER", "fake")
    get_settings.cache_clear()
    assert get_settings().llm_provider == "fake"
    get_settings.cache_clear()
