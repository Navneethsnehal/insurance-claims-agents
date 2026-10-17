"""Wire settings, the form corpus, the fraud model, and the chat provider."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from claims_agents.config import Settings, get_settings
from claims_agents.errors import ClaimsError
from claims_agents.fraud.score import FraudScorer
from claims_agents.fraud.train import load_bundle
from claims_agents.llm.base import LLMClient
from claims_agents.llm.factory import build_llm
from claims_agents.observability.tracing import SpanLedger, TracingHooks
from claims_agents.retrieval.corpus import Corpus, load_corpus
from claims_agents.retrieval.hybrid import HybridRetriever


@dataclass
class Dependencies:
    settings: Settings
    llm: LLMClient
    corpus: Corpus
    retriever: HybridRetriever
    scorer: FraudScorer
    ledger: SpanLedger


def build_dependencies(settings: Settings | None = None) -> Dependencies:
    resolved = settings or get_settings()
    corpus = load_corpus(resolved.corpus_path)
    bundle = load_bundle(
        resolved.fraud_model_path,
        resolved.fraud_train_path,
        resolved.fraud_test_path,
    )
    baseline = np.asarray(bundle["baseline"], dtype=np.float64)
    return Dependencies(
        settings=resolved,
        llm=build_llm(resolved),
        corpus=corpus,
        retriever=HybridRetriever(corpus.clauses),
        scorer=FraudScorer(
            model=bundle["model"],
            baseline=baseline,
            high_threshold=resolved.fraud_hitl_threshold,
        ),
        ledger=SpanLedger(TracingHooks(otel_enabled=resolved.otel_enabled)),
    )


def build_checkpointer(settings: Settings) -> Any:
    """Return the LangGraph checkpointer.

    Memory is the supported local and CI runtime. Postgres is opt-in and
    requires the ``postgres`` extra plus ``CLAIMS_DATABASE_URL``.
    """
    if settings.checkpointer == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    if settings.checkpointer != "postgres":
        raise ClaimsError(f"unsupported checkpointer {settings.checkpointer}")
    if not settings.database_url:
        raise ClaimsError("CLAIMS_DATABASE_URL is required when CLAIMS_CHECKPOINTER=postgres")
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
    except ImportError as exc:
        raise ClaimsError(
            "postgres checkpointer requires pip install 'claims-agents[postgres]'"
        ) from exc
    saver = PostgresSaver.from_conn_string(settings.database_url)
    if hasattr(saver, "setup"):
        saver.setup()
    return saver
