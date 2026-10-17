"""Validate structured model output and retry with the validation error attached."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

from claims_agents.errors import StructuredOutputError
from claims_agents.llm.base import LLMClient, LLMResponse

T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


@dataclass(frozen=True)
class StructuredCompletion(Generic[T]):
    parsed: T
    response: LLMResponse
    attempts: int


def extract_json(text: str) -> str:
    stripped = _FENCE.sub("", text.strip())
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return stripped


def complete_structured(
    llm: LLMClient,
    *,
    system: str,
    user: str,
    schema: type[T],
    retries: int,
) -> StructuredCompletion[T]:
    prompt = user
    last_error = "no attempt"
    for attempt in range(1, retries + 2):
        response = llm.complete(system=system, user=prompt)
        try:
            parsed = schema.model_validate_json(extract_json(response.text))
        except ValidationError as exc:
            last_error = json.dumps(exc.errors(), default=str)
            prompt = (
                f"{user}\n\nThe previous JSON failed validation:\n{last_error}\n"
                "Return only a JSON object matching the schema."
            )
            continue
        return StructuredCompletion(parsed=parsed, response=response, attempts=attempt)
    raise StructuredOutputError(last_error)
