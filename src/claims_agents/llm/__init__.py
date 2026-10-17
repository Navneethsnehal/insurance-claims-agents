"""Provider-agnostic chat completion."""

from claims_agents.llm.base import LLMResponse
from claims_agents.llm.factory import build_llm
from claims_agents.llm.fake import FakeLLM
from claims_agents.llm.structured import StructuredCompletion, complete_structured

__all__ = ["FakeLLM", "LLMResponse", "StructuredCompletion", "build_llm", "complete_structured"]
