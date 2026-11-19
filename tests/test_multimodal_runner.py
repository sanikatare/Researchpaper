"""Tests for multimodal payload validation, inline Part attachment, failure capture, and pilot isolation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from google.genai import types
from PIL import Image

from config import ExperimentConfig
from dataset.scenario_manager import load_active_dataset, resolve_image_path
from experiment import hydrogen_client, storage
from experiment.experiment_runner import (
    PlannedTrial,
    build_trial_plan,
    execute_single_planned_trial,
    run_experiment_batch,
)
from experiment.multimodal_runner import (
    prepare_modality_bundle,
    run_multimodal_trial_with_metadata,
)


class _InspectingModels:
    def __init__(self, raise_on_image: bool = False) -> None:
        self.calls = []
        self.raise_on_image = raise_on_image

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self.raise_on_image and isinstance(contents, list):
            raise RuntimeError("Simulated multimodal attachment API failure")
        usage = SimpleNamespace(
            prompt_token_count=64,
            candidates_token_count=22,
            thoughts_token_count=90,
            total_token_count=176,
        )
        part = SimpleNamespace(text="Controlled test response.", thought=False)
        cand = SimpleNamespace(finish_reason="STOP", content=SimpleNamespace(parts=[part]))
        return SimpleNamespace(
            text="Controlled test response.",
            model_version="gemini-4-hydrogen",
            response_id="resp-mm-1",
            usage_metadata=usage,
            prompt_feedback=None,
            candidates=[cand],
        )


def _sample_scenario():
    _, pilot_scenarios = load_active_dataset("pilot")
    assert len(pilot_scenarios) == 12
    return pilot_scenarios[0]


def test_text_only_has_no_image_and_image_plus_text_attaches_exact_image(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(storage, "PILOT_DIR", tmp_path / "pilot")
    monkeypatch.setattr(storage, "MAIN_DIR", tmp_path / "main")
    monkeypatch.setattr(storage, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(storage, "EXPERIMENT_LOG_MD", tmp_path / "experiment_log.md")
    (tmp_path / "pilot").mkdir(parents=True, exist_ok=True)
    (tmp_path / "main").mkdir(parents=True, exist_ok=True)

    scen = _sample_scenario()
    expected_img_path = resolve_image_path(scen["image_asset_reference"])
    assert expected_img_path is not None and expected_img_path.exists()
    expected_bytes = expected_img_path.read_bytes()

    mock_client = SimpleNamespace(models=_InspectingModels())
    cfg = ExperimentConfig(thinking_levels=["LOW"], modalities=["text_only", "image_plus_text"])

    # 1. Text-only trial
    txt_rec = execute_single_planned_trial(
        experiment_type="pilot",
        trial=PlannedTrial(scenario=scen, modality="text_only", thinking_level="LOW", repetition=1),
        config=cfg,
        client=mock_client,
    )
    assert len(mock_client.models.calls) == 1
    txt_contents = mock_client.models.calls[0]["contents"]
    assert isinstance(txt_contents, str)
    assert txt_contents == scen["text_only_prompt"]
    assert txt_rec["image_path"] is None
    assert txt_rec["image_asset_reference"] is None
    assert txt_rec["image_mime_type"] is None
    assert txt_rec["image_size_bytes"] is None
    assert txt_rec["attachment_verified"] is False
    assert txt_rec["dataset_version"] == scen["dataset_version"]

    # 2. Image+text trial
    img_rec = execute_single_planned_trial(
        experiment_type="pilot",
        trial=PlannedTrial(scenario=scen, modality="image_plus_text", thinking_level="LOW", repetition=1),
        config=cfg,
        client=mock_client,
    )
    assert len(mock_client.models.calls) == 2
    mm_contents = mock_client.models.calls[1]["contents"]
    assert isinstance(mm_contents, list)
    assert len(mm_contents) == 2
    img_part, txt_part = mm_contents
    assert isinstance(img_part, types.Part)
    assert img_part.inline_data is not None
    assert img_part.inline_data.mime_type == "image/png"
    assert img_part.inline_data.data == expected_bytes
    assert isinstance(txt_part, types.Part)
    assert txt_part.text == scen["image_text_prompt"]

    assert img_rec["image_asset_reference"] == scen["image_asset_reference"]
    assert img_rec["image_mime_type"] == "image/png"
    assert img_rec["image_size_bytes"] == len(expected_bytes)
    assert img_rec["attachment_verified"] is True
    assert img_rec["dataset_version"] == scen["dataset_version"]


def test_multimodal_rejects_corrupt_missing_mime_mismatch_and_placeholder_images(tmp_path) -> None:
    scen = dict(_sample_scenario())

    # Corrupt bytes
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not-a-valid-png")
    scen_corrupt = dict(scen, image_asset_reference=str(corrupt))
    with pytest.raises(Exception):
        prepare_modality_bundle(scen_corrupt, "image_plus_text")

    # Missing file
    scen_missing = dict(scen, image_asset_reference=str(tmp_path / "missing.png"))
    with pytest.raises(FileNotFoundError):
        prepare_modality_bundle(scen_missing, "image_plus_text")

    # Zero-byte file
    empty_png = tmp_path / "empty.png"
    empty_png.write_bytes(b"")
    scen_empty = dict(scen, image_asset_reference=str(empty_png))
    with pytest.raises(ValueError, match="empty"):
        prepare_modality_bundle(scen_empty, "image_plus_text")

    # PNG bytes saved with wrong extension (.jpg) -> MIME mismatch
    valid_png = tmp_path / "real.png"
    Image.new("RGB", (64, 64), color=(10, 20, 30)).save(valid_png, format="PNG")
    wrong_ext = tmp_path / "mislabeled.jpg"
    wrong_ext.write_bytes(valid_png.read_bytes())
    scen_mismatch = dict(scen, image_asset_reference=str(wrong_ext))
    with pytest.raises(ValueError, match="MIME mismatch"):
        prepare_modality_bundle(scen_mismatch, "image_plus_text")

    # Unsupported extension (.bmp)
    bmp_path = tmp_path / "unsupported.bmp"
    Image.new("RGB", (64, 64), color=(10, 20, 30)).save(bmp_path, format="BMP")
    scen_bmp = dict(scen, image_asset_reference=str(bmp_path))
    with pytest.raises(ValueError, match="Unsupported image file extension"):
        prepare_modality_bundle(scen_bmp, "image_plus_text")

    # Placeholder image file name rejected
    placeholder_path = tmp_path / "placeholder_card.png"
    placeholder_path.write_bytes(valid_png.read_bytes())
    scen_ph = dict(scen, image_asset_reference=str(placeholder_path))
    with pytest.raises(ValueError, match="Placeholder image"):
        prepare_modality_bundle(scen_ph, "image_plus_text")

    # Oversized file check (> 20 MB)
    huge_file = tmp_path / "huge.png"
    with open(huge_file, "wb") as fh:
        fh.truncate(hydrogen_client.MAX_IMAGE_BYTES + 1)
    scen_huge = dict(scen, image_asset_reference=str(huge_file))
    with pytest.raises(ValueError, match="exceeds 20 MB"):
        prepare_modality_bundle(scen_huge, "image_plus_text")


def test_image_api_error_captured_cleanly_and_pilot_isolation(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(storage, "PILOT_DIR", tmp_path / "pilot")
    monkeypatch.setattr(storage, "MAIN_DIR", tmp_path / "main")
    monkeypatch.setattr(storage, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(storage, "EXPERIMENT_LOG_MD", tmp_path / "experiment_log.md")
    (tmp_path / "pilot").mkdir(parents=True, exist_ok=True)
    (tmp_path / "main").mkdir(parents=True, exist_ok=True)

    _, pilot_scenarios = load_active_dataset("pilot")
    # Verify pilot factorial plan = 12 * 2 * 4 * 1 = 96 calls
    full_pilot_plan = build_trial_plan(
        scenarios=pilot_scenarios,
        modalities=["text_only", "image_plus_text"],
        thinking_levels=["LOW", "MEDIUM", "HIGH", "XHIGH"],
        repetitions=1,
        randomize_order=True,
        random_seed=42,
    )
    assert len(full_pilot_plan) == 96

    # Execute 1 scenario where image_plus_text raises an API error -> captured in pilot JSONL, main untouched
    mock_client = SimpleNamespace(models=_InspectingModels(raise_on_image=True))
    cfg = ExperimentConfig(
        thinking_levels=["LOW"],
        modalities=["text_only", "image_plus_text"],
        repetitions=1,
        max_retries=0,
        randomize_order=False,
    )
    summary = run_experiment_batch(
        experiment_type="pilot",
        scenarios=pilot_scenarios[:1],
        config=cfg,
        client=mock_client,
    )
    assert summary["succeeded"] == 1
    assert summary["failed"] == 1
    pilot_recs = storage.load_trial_records("pilot")
    main_recs = storage.load_trial_records("main")
    assert len(pilot_recs) == 2
    assert len(main_recs) == 0
    err_rec = [r for r in pilot_recs if r["modality"] == "image_plus_text"][0]
    assert err_rec["status"] == "error"
    assert "multimodal attachment" in (err_rec["error_message"] or "")
    assert err_rec["image_asset_reference"] == pilot_scenarios[0]["image_asset_reference"]
    assert err_rec["attachment_verified"] is True
