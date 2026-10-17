from pathlib import Path

import pytest

from claims_agents import __version__
from claims_agents.config import Settings, default_data_dir, get_settings
from claims_agents.errors import (
    ClaimNotFoundError,
    ClaimsError,
    ClaimStateError,
    DuplicateClaimError,
    ExtractionError,
    LLMConfigError,
    PolicyNotFoundError,
    StructuredOutputError,
)


def test_version_is_set() -> None:
    assert __version__ == "0.1.0"


def test_defaults_are_hermetic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLAIMS_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("CLAIMS_DATA_DIR", raising=False)
    get_settings.cache_clear()
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "fake"
    assert settings.fraud_hitl_threshold == 0.65
    assert settings.value_hitl_threshold == 25_000
    assert settings.corpus_path.name == "corpus.json"


def test_domain_errors_are_claims_errors() -> None:
    samples = [
        LLMConfigError("llm"),
        StructuredOutputError("json"),
        ExtractionError("parse"),
        PolicyNotFoundError("policy"),
        ClaimNotFoundError("id"),
        ClaimStateError("state"),
        DuplicateClaimError("dup"),
    ]
    for exc in samples:
        assert isinstance(exc, ClaimsError)


def test_data_dir_resolution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    packaged = tmp_path / "packaged"
    repo = tmp_path / "repo"
    explicit = Settings(_env_file=None, data_dir=repo)
    assert explicit.resolved_data_dir() == repo
    assert explicit.fraud_train_path == repo / "fraud" / "train.csv"
    assert explicit.fraud_test_path == repo / "fraud" / "test.csv"

    monkeypatch.setenv("CLAIMS_DATA_DIR", str(packaged))
    from_env = Settings(_env_file=None)
    assert from_env.resolved_data_dir() == packaged


def test_default_data_dir_prefers_existing_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_data = tmp_path / "data"
    packaged = tmp_path / "src" / "claims_agents" / "data"
    repo_data.mkdir(parents=True)
    packaged.mkdir(parents=True)
    config_file = tmp_path / "src" / "claims_agents" / "config.py"
    monkeypatch.setattr("claims_agents.config.__file__", str(config_file))

    assert default_data_dir() == repo_data
    repo_data.rmdir()
    assert default_data_dir() == packaged
    packaged.rmdir()
    assert default_data_dir() == repo_data


def test_env_overrides_provider(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CLAIMS_LLM_PROVIDER", "openai")
    monkeypatch.setenv("CLAIMS_OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("CLAIMS_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.llm_provider == "openai"
    assert settings.resolved_data_dir() == tmp_path
    assert settings.fraud_model_path == tmp_path / "models" / "fraud_gbm.joblib"
    get_settings.cache_clear()
