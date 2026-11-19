"""Software unit tests for thinking-level mapping and resumable runner (mock client only)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from google.genai import types

from config import ExperimentConfig
from experiment import hydrogen_client, storage
from experiment.experiment_runner import run_experiment_batch
from experiment.hydrogen_client import (
    UnsupportedThinkingLevelError,
    call_gemini_hydrogen,
    resolve_thinking_config,
)


class _MockModels:
    def __init__(self) -> None:
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        usage = SimpleNamespace(
            prompt_token_count=42,
            candidates_token_count=18,
            thoughts_token_count=120,
            total_token_count=180,
        )
        part = SimpleNamespace(text="Mock software test response.", thought=False)
        cand = SimpleNamespace(
            finish_reason="STOP",
            content=SimpleNamespace(parts=[part]),
        )
        return SimpleNamespace(
            text="Mock software test response.",
            model_version="gemini-4-hydrogen-test",
            response_id="resp-test-1",
            usage_metadata=usage,
            prompt_feedback=None,
            candidates=[cand],
        )


class _MockClient:
    def __init__(self) -> None:
        self.models = _MockModels()


def test_thinking_level_mapping_all_four_levels() -> None:
    for lvl in ("LOW", "MEDIUM", "HIGH", "XHIGH"):
        cfg = resolve_thinking_config(lvl)
        assert cfg.thinking_level is not None
        assert str(cfg.thinking_level).endswith(lvl)

    with pytest.raises(UnsupportedThinkingLevelError):
        resolve_thinking_config("MINIMAL")


def test_mock_multimodal_and_text_call() -> None:
    mock_client = _MockClient()
    res_txt = call_gemini_hydrogen(
        prompt="Test benign prompt",
        thinking_level="XHIGH",
        client=mock_client,
    )
    assert res_txt.status == "success"
    assert res_txt.thinking_level == "XHIGH"
    assert res_txt.input_tokens == 42
    assert res_txt.output_tokens == 18
    assert res_txt.thinking_tokens == 120

    # Multimodal with actual test image
    from dataset.scenario_manager import resolve_image_path

    img = resolve_image_path("scn-001.png")
    res_img = call_gemini_hydrogen(
        prompt="Inspect image",
        thinking_level="LOW",
        image_path=img,
        client=mock_client,
    )
    assert res_img.status == "success"
    assert res_img.modality == "image_plus_text"
    assert len(mock_client.models.calls) == 2
    assert isinstance(mock_client.models.calls[1]["contents"], list)
    assert isinstance(mock_client.models.calls[1]["contents"][0], types.Part)


def test_resumable_batch_prevents_duplicates(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(storage, "PILOT_DIR", tmp_path / "pilot")
    monkeypatch.setattr(storage, "MAIN_DIR", tmp_path / "main")
    monkeypatch.setattr(storage, "RESULTS_DIR", tmp_path)
    (tmp_path / "pilot").mkdir(parents=True, exist_ok=True)
    (tmp_path / "main").mkdir(parents=True, exist_ok=True)

    from dataset.scenario_manager import load_scenarios

    scenarios = load_scenarios()[:2]
    cfg = ExperimentConfig(
        thinking_levels=["LOW", "HIGH"],
        modalities=["text_only", "image_plus_text"],
        repetitions=1,
        randomize_order=False,
    )
    mock_client = _MockClient()

    # Stop after 3 trials to simulate interruption
    counter = {"n": 0}

    def _stop_after_three() -> bool:
        return counter["n"] >= 3

    def _on_done(rec, idx, total) -> None:
        counter["n"] += 1

    res1 = run_experiment_batch(
        experiment_type="pilot",
        scenarios=scenarios,
        config=cfg,
        should_stop=_stop_after_three,
        on_trial_complete=_on_done,
        client=mock_client,
    )
    assert res1["total_planned"] == 8
    assert res1["executed"] == 3
    assert res1["halted_reason"] == "stopped"

    # Resume remaining 5 trials without duplicating the first 3
    res2 = run_experiment_batch(
        experiment_type="pilot",
        scenarios=scenarios,
        config=cfg,
        client=mock_client,
    )
    assert res2["skipped_existing"] == 3
    assert res2["executed"] == 5
    assert res2["halted_reason"] == "completed"
    assert len(storage.load_trial_records("pilot")) == 8
