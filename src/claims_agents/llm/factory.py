"""Select a chat provider from settings."""

from __future__ import annotations

from claims_agents.config import Settings
from claims_agents.errors import LLMConfigError
from claims_agents.llm.base import LLMClient
from claims_agents.llm.fake import FakeLLM
from claims_agents.llm.providers import AnthropicProvider, OpenAIProvider


def build_llm(settings: Settings) -> LLMClient:
    if settings.llm_provider == "fake":
        return FakeLLM()
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise LLMConfigError(
                "CLAIMS_OPENAI_API_KEY is required when CLAIMS_LLM_PROVIDER=openai"
            )
        return OpenAIProvider(api_key=settings.openai_api_key, model=settings.openai_model)
    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise LLMConfigError(
                "CLAIMS_ANTHROPIC_API_KEY is required when CLAIMS_LLM_PROVIDER=anthropic"
            )
        return AnthropicProvider(api_key=settings.anthropic_api_key, model=settings.anthropic_model)
    raise LLMConfigError(f"unsupported provider {settings.llm_provider}")
