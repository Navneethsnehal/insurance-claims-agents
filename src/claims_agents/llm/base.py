"""Completion types shared by every provider."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from claims_agents.pricing import cost_usd, estimate_tokens


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float

    @property
    def cost_usd(self) -> float:
        return cost_usd(self.model, self.input_tokens, self.output_tokens)


class LLMClient(Protocol):
    model: str

    def complete(self, *, system: str, user: str) -> LLMResponse:
        """Return a completion. Implementations must not log the raw user text."""


def token_response(text: str, model: str, system: str, user: str, latency_ms: float) -> LLMResponse:
    return LLMResponse(
        text=text,
        model=model,
        input_tokens=estimate_tokens(system) + estimate_tokens(user),
        output_tokens=estimate_tokens(text),
        latency_ms=latency_ms,
    )
