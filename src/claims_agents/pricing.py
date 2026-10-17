"""List-price token accounting shared by providers and the eval report."""

from __future__ import annotations

# USD per token. Unknown models cost zero rather than a guessed rate.
_PRICES: dict[str, tuple[float, float]] = {
    "fake-claims-v1": (0.0, 0.0),
    "gpt-4o-mini": (0.15 / 1_000_000, 0.60 / 1_000_000),
    "claude-3-5-sonnet-latest": (3.0 / 1_000_000, 15.0 / 1_000_000),
}


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    input_price, output_price = _PRICES.get(model, (0.0, 0.0))
    return input_tokens * input_price + output_tokens * output_price
