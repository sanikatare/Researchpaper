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
    monkeypatch.setattr(storage, "EXPERIMENT_LOG_MD", tmp_path / "experiment_log.md")
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
    saved = storage.load_trial_records("pilot")
    assert len(saved) == 8
    assert all(r.get("dataset_version") == scenarios[0]["dataset_version"] for r in saved)
    assert all(r.get("experiment_run_id") for r in saved)
    assert all(r.get("dataset_hash") and len(r["dataset_hash"]) == 64 for r in saved)
    assert all(r.get("random_seed") == 42 for r in saved)
    assert all(r.get("software_version") for r in saved)
    assert all(isinstance(r.get("config_snapshot"), dict) for r in saved)


def test_deterministic_scheduling_and_seed_reproducibility() -> None:
    from dataset.scenario_manager import load_scenarios
    from experiment.experiment_runner import build_trial_plan

    scenarios = load_scenarios()[:3]
    plan_a = build_trial_plan(
        scenarios, ["text_only", "image_plus_text"], ["LOW", "MEDIUM", "HIGH", "XHIGH"], 3, True, 42
    )
    plan_b = build_trial_plan(
        list(reversed(scenarios)),
        ["text_only", "image_plus_text"],
        ["LOW", "MEDIUM", "HIGH", "XHIGH"],
        3,
        True,
        42,
    )
    keys_a = [(t.scenario["scenario_id"], t.modality, t.thinking_level, t.repetition) for t in plan_a]
    keys_b = [(t.scenario["scenario_id"], t.modality, t.thinking_level, t.repetition) for t in plan_b]
    assert len(keys_a) == 3 * 2 * 4 * 3
    assert keys_a == keys_b

    plan_c = build_trial_plan(
        scenarios, ["text_only", "image_plus_text"], ["LOW", "MEDIUM", "HIGH", "XHIGH"], 3, True, 99
    )
    keys_c = [(t.scenario["scenario_id"], t.modality, t.thinking_level, t.repetition) for t in plan_c]
    assert keys_a != keys_c


def test_failure_tracking_recovery_and_non_overwriting(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(storage, "PILOT_DIR", tmp_path / "pilot")
    monkeypatch.setattr(storage, "MAIN_DIR", tmp_path / "main")
    monkeypatch.setattr(storage, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(storage, "EXPERIMENT_LOG_MD", tmp_path / "experiment_log.md")
    (tmp_path / "pilot").mkdir(parents=True, exist_ok=True)
    (tmp_path / "main").mkdir(parents=True, exist_ok=True)

    from dataset.scenario_manager import load_scenarios

    scenarios = load_scenarios()[:1]
    cfg = ExperimentConfig(
        thinking_levels=["LOW", "HIGH"],
        modalities=["text_only"],
        repetitions=1,
        max_retries=0,
        randomize_order=False,
    )

    class _FlakyModels(_MockModels):
        def __init__(self):
            super().__init__()
            self.fail_once = True

        def generate_content(self, model, contents, config):
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("Simulated transient failure")
            return super().generate_content(model, contents, config)

    flaky_client = SimpleNamespace(models=_FlakyModels())
    res1 = run_experiment_batch("pilot", scenarios, cfg, client=flaky_client)
    assert res1["executed"] == 2
    assert res1["failed"] == 1
    assert res1["succeeded"] == 1

    # Resume: only the failed trial is re-run; existing records are preserved (never overwritten)
    res2 = run_experiment_batch("pilot", scenarios, cfg, client=flaky_client)
    assert res2["skipped_existing"] == 1
    assert res2["executed"] == 1
    assert res2["succeeded"] == 1
    records = storage.load_trial_records("pilot")
    assert len(records) == 3  # 1 initial error + 1 initial success + 1 recovered success


def test_approval_and_validated_level_gates_enforced() -> None:
    from dataset.scenario_manager import load_active_dataset

    _, pilot_scenarios = load_active_dataset("pilot")
    cfg = ExperimentConfig()
    with pytest.raises(PermissionError, match="researcher approval"):
        run_experiment_batch(
            "pilot",
            pilot_scenarios,
            cfg,
            require_approved_dataset=True,
            client=_MockClient(),
        )


def test_controller_pause_resume_stop_lifecycle(tmp_path, monkeypatch) -> None:
    import time
    from dataset.scenario_manager import load_scenarios
    from experiment.experiment_controller import ExperimentController

    monkeypatch.setattr(storage, "PILOT_DIR", tmp_path / "pilot")
    monkeypatch.setattr(storage, "MAIN_DIR", tmp_path / "main")
    monkeypatch.setattr(storage, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(storage, "EXPERIMENT_LOG_MD", tmp_path / "experiment_log.md")
    (tmp_path / "pilot").mkdir(parents=True, exist_ok=True)
    (tmp_path / "main").mkdir(parents=True, exist_ok=True)

    class _SlowClient:
        class _SlowModels(_MockModels):
            def generate_content(self, model, contents, config):
                time.sleep(0.03)
                return super().generate_content(model, contents, config)

        def __init__(self):
            self.models = self._SlowModels()

    scenarios = load_scenarios()[:2]
    cfg = ExperimentConfig(
        thinking_levels=["LOW", "MEDIUM"],
        modalities=["text_only", "image_plus_text"],
        repetitions=1,
        randomize_order=False,
    )
    ctrl = ExperimentController("pilot")
    ctrl.start(
        scenarios=scenarios,
        config=cfg,
        explicit_approved=True,
        client=_SlowClient(),
        require_approved_dataset=False,
        require_validated_levels=False,
    )
    time.sleep(0.05)
    ctrl.pause()
    for _ in range(40):
        if not ctrl.is_running():
            break
        time.sleep(0.02)
    snap_paused = ctrl.snapshot()
    assert snap_paused["status"] in ("PAUSED", "COMPLETED")
    saved_after_pause = len(storage.load_trial_records("pilot"))
    assert 1 <= saved_after_pause <= 8

    if saved_after_pause < 8:
        ctrl.start(
            scenarios=scenarios,
            config=cfg,
            explicit_approved=True,
            client=_MockClient(),
            require_approved_dataset=False,
            require_validated_levels=False,
        )
        for _ in range(50):
            if not ctrl.is_running():
                break
            time.sleep(0.02)
        assert ctrl.snapshot()["status"] == "COMPLETED"
        assert len(storage.load_trial_records("pilot")) == 8


def test_locked_repetitions_enforced_when_approved_gate_enabled() -> None:
    from dataset.scenario_manager import load_active_dataset

    _, pilot_scenarios = load_active_dataset("pilot")
    _, main_scenarios = load_active_dataset("main")

    with pytest.raises(ValueError, match="repetitions=1"):
        run_experiment_batch(
            "pilot",
            pilot_scenarios,
            ExperimentConfig(repetitions=2),
            require_approved_dataset=True,
            client=_MockClient(),
        )

    with pytest.raises(ValueError, match="repetitions=3"):
        run_experiment_batch(
            "main",
            main_scenarios,
            ExperimentConfig(repetitions=1),
            require_approved_dataset=True,
            client=_MockClient(),
        )

