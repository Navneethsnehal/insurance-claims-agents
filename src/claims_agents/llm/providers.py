"""OpenAI and Anthropic adapters.

Both clients are optional imports. The fake provider is the default and does
not import either SDK. Pass ``client`` in tests to avoid a network call.
"""

from __future__ import annotations

import time
from typing import Any

from claims_agents.llm.base import LLMResponse


class OpenAIProvider:
    def __init__(self, api_key: str, model: str, client: Any | None = None) -> None:
        self.api_key = api_key
        self.model = model
        self._client = client

    def complete(self, *, system: str, user: str) -> LLMResponse:
        client = self._client if self._client is not None else self._build_client()
        started = time.perf_counter()
        response = client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = response.choices[0].message.content or ""
        usage = response.usage
        return LLMResponse(
            text=text,
            model=self.model,
            input_tokens=int(usage.prompt_tokens),
            output_tokens=int(usage.completion_tokens),
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    def _build_client(self) -> Any:
        from openai import OpenAI

        return OpenAI(api_key=self.api_key)


class AnthropicProvider:
    def __init__(self, api_key: str, model: str, client: Any | None = None) -> None:
        self.api_key = api_key
        self.model = model
        self._client = client

    def complete(self, *, system: str, user: str) -> LLMResponse:
        client = self._client if self._client is not None else self._build_client()
        started = time.perf_counter()
        response = client.messages.create(
            model=self.model,
            max_tokens=2000,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(getattr(block, "text", "") for block in response.content)
        usage = response.usage
        return LLMResponse(
            text=text,
            model=self.model,
            input_tokens=int(usage.input_tokens),
            output_tokens=int(usage.output_tokens),
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    def _build_client(self) -> Any:
        from anthropic import Anthropic

        return Anthropic(api_key=self.api_key)
