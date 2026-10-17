"""Runtime configuration.

Environment variables use the ``CLAIMS_`` prefix. The default provider is the
deterministic fake model so tests, evals, and CI never require an API key.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["fake", "openai", "anthropic"]
CheckpointerName = Literal["memory", "postgres"]


def default_data_dir() -> Path:
    """Locate the repository ``data`` directory in a checkout or image."""
    repo_data = Path(__file__).resolve().parents[2] / "data"
    packaged = Path(__file__).resolve().parent / "data"
    if repo_data.exists():
        return repo_data
    if packaged.exists():
        return packaged
    return repo_data


class Settings(BaseSettings):
    """Process configuration loaded from the environment."""

    model_config = SettingsConfigDict(
        env_prefix="CLAIMS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: ProviderName = "fake"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-3-5-sonnet-latest"
    fraud_hitl_threshold: float = 0.65
    value_hitl_threshold: float = 25_000.0
    confidence_escalation_threshold: float = 0.55
    retrieval_k: int = 5
    data_dir: Path | None = None
    log_level: str = "INFO"
    otel_enabled: bool = False
    checkpointer: CheckpointerName = "memory"
    database_url: str | None = None
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    aftermarket_sublimit: float = 1_000.0
    structured_output_retries: int = Field(default=2, ge=0, le=5)

    def resolved_data_dir(self) -> Path:
        if self.data_dir is not None:
            return self.data_dir
        return default_data_dir()

    @property
    def fraud_model_path(self) -> Path:
        return self.resolved_data_dir() / "models" / "fraud_gbm.joblib"

    @property
    def fraud_train_path(self) -> Path:
        return self.resolved_data_dir() / "fraud" / "train.csv"

    @property
    def fraud_test_path(self) -> Path:
        return self.resolved_data_dir() / "fraud" / "test.csv"

    @property
    def corpus_path(self) -> Path:
        return self.resolved_data_dir() / "policies" / "corpus.json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
