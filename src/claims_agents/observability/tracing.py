"""Per-node spans with OpenTelemetry and LangSmith-shaped hooks.

The in-process ledger is the source of truth for tests. When OpenTelemetry
is enabled, each record is also emitted on the global tracer (a no-op unless
the process installed an SDK provider). A LangSmith client, if injected,
receives the same metadata dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from opentelemetry import trace


@dataclass(frozen=True)
class SpanRecord:
    name: str
    duration_ms: float
    attributes: dict[str, str | int | float | bool]


def langsmith_metadata(record: SpanRecord) -> dict[str, Any]:
    return {
        "name": record.name,
        "run_type": "chain",
        "extra": {"metadata": dict(record.attributes), "duration_ms": record.duration_ms},
    }


def emit_otel(record: SpanRecord) -> None:
    tracer = trace.get_tracer("claims_agents")
    span = tracer.start_span(record.name)
    for key, value in record.attributes.items():
        span.set_attribute(key, value)
    span.set_attribute("duration_ms", record.duration_ms)
    span.end()


@dataclass
class TracingHooks:
    otel_enabled: bool = False
    langsmith_client: Any | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    def emit(self, record: SpanRecord) -> None:
        event = langsmith_metadata(record)
        self.events.append(event)
        if self.otel_enabled:
            emit_otel(record)
        if self.langsmith_client is not None:
            self.langsmith_client.create_run(**event)


@dataclass
class SpanLedger:
    hooks: TracingHooks
    spans: list[SpanRecord] = field(default_factory=list)

    def record(self, name: str, duration_ms: float, attributes: dict[str, Any]) -> None:
        clean: dict[str, str | int | float | bool] = {}
        for key, value in attributes.items():
            if isinstance(value, (str, int, float, bool)):
                clean[key] = value
        span = SpanRecord(name=name, duration_ms=round(duration_ms, 3), attributes=clean)
        self.spans.append(span)
        self.hooks.emit(span)
