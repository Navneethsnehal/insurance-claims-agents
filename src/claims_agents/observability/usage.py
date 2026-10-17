"""Token and dollar accounting.

Prices are list rates per token for the models this service knows about.
Unknown model names cost zero rather than a guessed rate. The fake provider
is free; it still records tokens so the eval report has a cost column.
"""

from __future__ import annotations

from typing import Any

from claims_agents.llm.base import LLMResponse
from claims_agents.pricing import cost_usd, estimate_tokens

__all__ = ["cost_usd", "estimate_tokens", "usage_record", "usage_totals"]


def usage_record(response: LLMResponse, node: str) -> dict[str, Any]:
    return {
        "node": node,
        "model": response.model,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "latency_ms": round(response.latency_ms, 3),
        "cost_usd": round(response.cost_usd, 8),
    }


def usage_totals(records: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "input_tokens": float(sum(int(row["input_tokens"]) for row in records)),
        "output_tokens": float(sum(int(row["output_tokens"]) for row in records)),
        "latency_ms": round(sum(float(row["latency_ms"]) for row in records), 3),
        "cost_usd": round(sum(float(row["cost_usd"]) for row in records), 8),
    }
