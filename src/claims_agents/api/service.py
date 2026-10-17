"""Application service around the compiled workflow.

PII is masked before ``invoke`` so the checkpointer input is already clean.
The graph masks again and still sees injection text, which is what forces review.
"""

from __future__ import annotations

import re
import threading
import uuid
from datetime import UTC, datetime
from typing import Any

from langgraph.types import Command

from claims_agents.api.schemas import ClaimResponse, UsageTotals
from claims_agents.errors import (
    ClaimNotFoundError,
    ClaimStateError,
    DuplicateClaimError,
    PolicyNotFoundError,
)
from claims_agents.graph.workflow import (
    build_graph,
    initial_state,
    read_interrupt,
    thread_config,
)
from claims_agents.guardrails.pii import redact_pii
from claims_agents.observability.usage import usage_totals
from claims_agents.runtime import Dependencies, build_checkpointer

_CLAIM_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{2,40}$")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _new_claim_id() -> str:
    return "CLM-" + uuid.uuid4().hex[:8].upper()


class ClaimService:
    def __init__(self, deps: Dependencies, graph: Any | None = None) -> None:
        self.deps = deps
        self.graph = graph or build_graph(deps, build_checkpointer(deps.settings))
        self._index: dict[str, dict[str, str]] = {}
        self._lock = threading.Lock()

    def submit(
        self,
        *,
        policy_id: str,
        documents: list[str],
        claim_id: str | None = None,
    ) -> ClaimResponse:
        if self.deps.corpus.get(policy_id) is None:
            raise PolicyNotFoundError(f"unknown policy {policy_id}")
        chosen = claim_id or _new_claim_id()
        if not _CLAIM_ID.match(chosen):
            raise ClaimStateError(f"invalid claim id {chosen}")
        prepared = [redact_pii(document)[0] for document in documents]
        with self._lock:
            if chosen in self._index:
                raise DuplicateClaimError(chosen)
            stamp = _now()
            self._index[chosen] = {
                "policy_id": policy_id,
                "created_at": stamp,
                "updated_at": stamp,
            }
            self.graph.invoke(
                initial_state(chosen, policy_id, prepared),
                thread_config(chosen),
            )
            self._index[chosen]["updated_at"] = _now()
            return self._present(chosen)

    def resume(
        self,
        claim_id: str,
        *,
        decision: str,
        reviewer: str,
        notes: str,
    ) -> ClaimResponse:
        with self._lock:
            if claim_id not in self._index:
                raise ClaimNotFoundError(claim_id)
            pending = read_interrupt(self.graph.get_state(thread_config(claim_id)))
            if pending is None:
                raise ClaimStateError("claim is not awaiting human review")
            self.graph.invoke(
                Command(
                    resume={"decision": decision, "reviewer": reviewer, "notes": notes},
                ),
                thread_config(claim_id),
            )
            self._index[claim_id]["updated_at"] = _now()
            return self._present(claim_id)

    def get(self, claim_id: str) -> ClaimResponse:
        with self._lock:
            if claim_id not in self._index:
                raise ClaimNotFoundError(claim_id)
            return self._present(claim_id)

    def list_claims(self) -> list[dict[str, str]]:
        with self._lock:
            rows = []
            for claim_id, meta in self._index.items():
                view = self._present(claim_id)
                rows.append(
                    {
                        "claim_id": claim_id,
                        "policy_id": meta["policy_id"],
                        "status": view.status,
                        "created_at": meta["created_at"],
                        "updated_at": meta["updated_at"],
                    }
                )
            return rows

    def _present(self, claim_id: str) -> ClaimResponse:
        snapshot = self.graph.get_state(thread_config(claim_id))
        values = dict(snapshot.values)
        pending = read_interrupt(snapshot)
        memo = values.get("memo") if isinstance(values.get("memo"), dict) else None
        if pending is not None:
            status = "awaiting_human"
        elif isinstance(memo, dict) and memo.get("workflow_status"):
            status = str(memo["workflow_status"])
        else:
            status = "in_review"
        meta = self._index[claim_id]
        usage = list(values.get("usage") or [])
        totals = usage_totals(usage)
        return ClaimResponse(
            claim_id=claim_id,
            policy_id=str(values.get("policy_id") or meta["policy_id"]),
            status=status,
            created_at=meta["created_at"],
            updated_at=meta["updated_at"],
            claim=values.get("claim") if isinstance(values.get("claim"), dict) else None,
            coverage=values.get("coverage") if isinstance(values.get("coverage"), dict) else None,
            fraud=values.get("fraud") if isinstance(values.get("fraud"), dict) else None,
            payout=values.get("payout") if isinstance(values.get("payout"), dict) else None,
            memo=memo,
            trace=list(values.get("trace") or []),
            usage=usage,
            usage_totals=UsageTotals.model_validate(totals),
            pending_review=pending,
            errors=list(values.get("errors") or []),
            escalation_reasons=list(values.get("escalation_reasons") or []),
        )
