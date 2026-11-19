"""Tests for append-only JSONL storage, deduplication, config drift protection, and CSV export."""

from __future__ import annotations

import pytest

from config import ExperimentConfig
from experiment import storage


def test_append_only_storage_and_resumability_dedup(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(storage, "PILOT_DIR", tmp_path / "pilot")
    monkeypatch.setattr(storage, "MAIN_DIR", tmp_path / "main")
    monkeypatch.setattr(storage, "RESULTS_DIR", tmp_path)
    (tmp_path / "pilot").mkdir(parents=True, exist_ok=True)
    (tmp_path / "main").mkdir(parents=True, exist_ok=True)

    rec1 = {
        "trial_id": "t1",
        "experiment_type": "pilot",
        "scenario_id": "SCN-001",
        "modality": "text_only",
        "thinking_level": "LOW",
        "repetition": 1,
        "status": "success",
        "response_text": "First",
    }
    rec2 = {
        "trial_id": "t2",
        "experiment_type": "pilot",
        "scenario_id": "SCN-001",
        "modality": "image_plus_text",
        "thinking_level": "XHIGH",
        "repetition": 1,
        "status": "error",
        "response_text": "",
    }
    storage.append_trial_record("pilot", rec1, also_mirror_unified=False)
    storage.append_trial_record("pilot", rec2, also_mirror_unified=False)

    loaded = storage.load_trial_records("pilot")
    assert len(loaded) == 2
    assert loaded[0]["trial_id"] == "t1"
    assert loaded[1]["trial_id"] == "t2"

    completed = storage.get_completed_conditions("pilot")
    assert ("SCN-001", "text_only", "LOW", 1) in completed
    # Failed trials must NOT be marked as completed
    assert ("SCN-001", "image_plus_text", "XHIGH", 1) not in completed

    out_csv = tmp_path / "pilot_export.csv"
    storage.export_records_to_csv("pilot", out_csv)
    assert out_csv.exists()


def test_config_drift_halts_execution(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(storage, "MAIN_DIR", tmp_path / "main")
    (tmp_path / "main").mkdir(parents=True, exist_ok=True)

    cfg1 = ExperimentConfig(max_output_tokens=2048)
    storage.verify_or_lock_config("main", cfg1, sdk_version="2.33.0")

    # Same config succeeds
    storage.verify_or_lock_config("main", cfg1, sdk_version="2.33.0")

    # Modified max_output_tokens raises ConfigurationDriftError
    cfg2 = ExperimentConfig(max_output_tokens=4096)
    with pytest.raises(storage.ConfigurationDriftError):
        storage.verify_or_lock_config("main", cfg2, sdk_version="2.33.0")

    # Modified SDK version raises ConfigurationDriftError
    with pytest.raises(storage.ConfigurationDriftError):
        storage.verify_or_lock_config("main", cfg1, sdk_version="2.34.0")
