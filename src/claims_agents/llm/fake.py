"""Deterministic stand-in for a hosted chat model.

Extraction delegates to the FNOL parser. The memo writer fills a fixed
template from the JSON it is given. Neither path calls a network.
"""

from __future__ import annotations

import json
import time

from claims_agents.errors import ExtractionError
from claims_agents.extraction.parser import parse_fnol
from claims_agents.llm.base import LLMResponse, token_response
from claims_agents.llm.structured import extract_json


class FakeLLM:
    model = "fake-claims-v1"

    def complete(self, *, system: str, user: str) -> LLMResponse:
        started = time.perf_counter()
        if "TASK: extract_claim" in system:
            text = self._extract(user)
        elif "TASK: write_memo" in system:
            text = self._memo(user)
        else:
            text = "Acknowledged."
        latency_ms = (time.perf_counter() - started) * 1000
        return token_response(text, self.model, system, user, latency_ms)

    def _extract(self, user: str) -> str:
        try:
            extracted = parse_fnol(user)
        except ExtractionError:
            return "{}"
        return extracted.model_dump_json()

    def _memo(self, user: str) -> str:
        try:
            data = json.loads(extract_json(user))
        except json.JSONDecodeError:
            data = {}
        if not isinstance(data, dict):
            data = {}
        claim_id = data.get("claim_id", "the claim")
        recommendation = data.get("recommendation", "refer")
        outcome = data.get("outcome") or "undetermined"
        indemnity = float(data.get("indemnity") or 0)
        band = data.get("fraud_band") or "unknown"
        return (
            f"Claim {claim_id} is {outcome}. "
            f"The recommendation is {recommendation} with indemnity {indemnity:.2f}. "
            f"Fraud band is {band}."
        )
