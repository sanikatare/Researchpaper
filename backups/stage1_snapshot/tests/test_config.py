"""Tests for configuration loading, validation, and credential masking."""

from __future__ import annotations

from config import (
    DEFAULT_MODEL_ID,
    MODALITIES,
    THINKING_LEVELS,
    ExperimentConfig,
    get_api_key,
    mask_api_key,
)


def test_default_experiment_config_valid() -> None:
    cfg = ExperimentConfig()
    assert cfg.model_id == DEFAULT_MODEL_ID
    assert tuple(cfg.thinking_levels) == THINKING_LEVELS
    assert tuple(cfg.modalities) == MODALITIES
    assert cfg.validate() == []


def test_config_rejects_non_hydrogen_model() -> None:
    cfg = ExperimentConfig(model_id="other-model")
    errs = cfg.validate()
    assert any("gemini-4-hydrogen" in e for e in errs)


def test_config_rejects_invalid_thinking_level() -> None:
    cfg = ExperimentConfig(thinking_levels=["LOW", "MINIMAL"])
    errs = cfg.validate()
    assert any("MINIMAL" in e for e in errs)


def test_safe_dict_contains_no_api_key() -> None:
    cfg = ExperimentConfig()
    d = cfg.to_safe_dict()
    assert "api_key" not in d
    assert "GEMINI_API_KEY" not in d


def test_mask_api_key_and_env_read(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyTestSecretKey1234")
    key = get_api_key()
    assert key == "AIzaSyTestSecretKey1234"
    masked = mask_api_key(key)
    assert masked.startswith("AIza")
    assert masked.endswith("1234")
    assert "TestSecretKey" not in masked
