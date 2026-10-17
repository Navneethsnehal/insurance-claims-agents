"""FastAPI application."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from claims_agents import __version__
from claims_agents.api.schemas import (
    ClaimResponse,
    HealthResponse,
    ResumeClaimRequest,
    SubmitClaimRequest,
)
from claims_agents.api.service import ClaimService
from claims_agents.config import get_settings
from claims_agents.errors import (
    ClaimNotFoundError,
    ClaimStateError,
    DuplicateClaimError,
    PolicyNotFoundError,
)
from claims_agents.observability.logging import configure_logging
from claims_agents.runtime import build_dependencies
from claims_agents.scenarios import scenarios


def create_app(service: ClaimService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings = get_settings()
        configure_logging(settings.log_level)
        if service is not None:
            app.state.service = service
        else:
            app.state.service = ClaimService(build_dependencies(settings))
        yield

    app = FastAPI(
        title="Claims Agents",
        version=__version__,
        summary="Supervisor-routed claims triage with coverage, fraud, and human review.",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        claim_service: ClaimService = app.state.service
        return HealthResponse(
            status="ok",
            llm_provider=claim_service.deps.settings.llm_provider,
            vector_store="local_hybrid",
            checkpointer=claim_service.deps.settings.checkpointer,
            fraud_model="loaded",
            policies=len(claim_service.deps.corpus.policies),
            clauses=len(claim_service.deps.corpus.clauses),
        )

    @app.get("/samples")
    async def list_samples() -> list[dict[str, str]]:
        return [
            {
                "claim_id": scenario.claim_id,
                "policy_id": scenario.policy_id,
                "claimant_name": scenario.claimant_name,
                "loss_type": scenario.loss_type,
                "book_outcome": scenario.expected_outcome,
                "document": scenario.document(),
            }
            for scenario in scenarios()
        ]

    @app.get("/claims")
    async def list_claims() -> list[dict[str, str]]:
        claim_service: ClaimService = app.state.service
        return await asyncio.to_thread(claim_service.list_claims)

    @app.post("/claims", response_model=ClaimResponse, status_code=201)
    async def submit_claim(body: SubmitClaimRequest) -> ClaimResponse:
        claim_service: ClaimService = app.state.service
        try:
            return await asyncio.to_thread(
                lambda: claim_service.submit(
                    policy_id=body.policy_id,
                    documents=body.documents,
                    claim_id=body.claim_id,
                )
            )
        except PolicyNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except DuplicateClaimError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ClaimStateError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/claims/{claim_id}", response_model=ClaimResponse)
    async def get_claim(claim_id: str) -> ClaimResponse:
        claim_service: ClaimService = app.state.service
        try:
            return await asyncio.to_thread(claim_service.get, claim_id)
        except ClaimNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/claims/{claim_id}/resume", response_model=ClaimResponse)
    async def resume_claim(claim_id: str, body: ResumeClaimRequest) -> ClaimResponse:
        claim_service: ClaimService = app.state.service
        try:
            return await asyncio.to_thread(
                lambda: claim_service.resume(
                    claim_id,
                    decision=body.decision,
                    reviewer=body.reviewer,
                    notes=body.notes,
                )
            )
        except ClaimNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ClaimStateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return app


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "claims_agents.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        factory=False,
    )


app = create_app()
