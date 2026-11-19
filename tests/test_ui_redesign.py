"""Tests for the redesigned 5-page monochrome academic UI, Experiment Lab chatbot, and annotation states."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from dataset import scenario_manager
from evaluation import annotation
from experiment import exploratory_lab, hydrogen_client, storage


class _LabMockModels:
    def __init__(self) -> None:
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        usage = SimpleNamespace(
            prompt_token_count=21,
            candidates_token_count=14,
            thoughts_token_count=64,
            total_token_count=99,
        )
        part = SimpleNamespace(text="Exploratory response from Gemini 4 Hydrogen.", thought=False)
        cand = SimpleNamespace(finish_reason="STOP", content=SimpleNamespace(parts=[part]))
        return SimpleNamespace(
            text="Exploratory response from Gemini 4 Hydrogen.",
            model_version="gemini-4-hydrogen",
            response_id=f"lab-{len(self.calls)}",
            usage_metadata=usage,
            prompt_feedback=None,
            candidates=[cand],
        )


def _write_validation_state(path: Path, levels_ok: dict) -> None:
    state = {
        "model_id": "gemini-4-hydrogen",
        "sdk_version": "3.6.1",
        "levels": {},
        "multimodal_probe": {"validated": False},
    }
    for lvl in ("LOW", "MEDIUM", "HIGH", "XHIGH"):
        cfg = levels_ok.get(lvl, {})
        state["levels"][lvl] = {
            "validated": bool(cfg.get("validated", False)),
            "supported_by_endpoint": cfg.get("supported_by_endpoint"),
            "modality_tested": cfg.get("modality_tested"),
            "last_tested_utc": "2026-11-19T00:00:00+00:00" if cfg.get("validated") else None,
            "error": cfg.get("error"),
        }
    path.write_text(json.dumps(state), encoding="utf-8")


def test_exploratory_lab_thinking_level_gating_and_isolation(tmp_path, monkeypatch) -> None:
    val_path = tmp_path / "api_val.json"
    _write_validation_state(
        val_path,
        {
            "LOW": {"validated": True, "supported_by_endpoint": True},
            "MEDIUM": {"validated": False, "supported_by_endpoint": None},
            "HIGH": {
                "validated": False,
                "supported_by_endpoint": False,
                "error": "Endpoint does not support HIGH in mock",
            },
        },
    )
    monkeypatch.setattr(hydrogen_client, "API_VALIDATION_STATE_PATH", val_path)
    monkeypatch.setattr(storage, "PILOT_DIR", tmp_path / "pilot")
    monkeypatch.setattr(storage, "MAIN_DIR", tmp_path / "main")
    monkeypatch.setattr(storage, "RESULTS_DIR", tmp_path)
    (tmp_path / "pilot").mkdir(parents=True, exist_ok=True)
    (tmp_path / "main").mkdir(parents=True, exist_ok=True)

    # Unvalidated / unsupported levels must be blocked
    med_avail = exploratory_lab.check_thinking_level_availability("MEDIUM")
    assert med_avail["available"] is False
    high_avail = exploratory_lab.check_thinking_level_availability("HIGH")
    assert high_avail["available"] is False
    assert "not supported" in high_avail["reason"].lower()

    mock_client = SimpleNamespace(models=_LabMockModels())
    chat_log = tmp_path / "exploratory" / "chat_history.jsonl"
    cand_json = tmp_path / "exploratory" / "candidate_scenarios.json"

    with pytest.raises(PermissionError):
        exploratory_lab.send_exploratory_prompt(
            prompt="Should fail for unvalidated MEDIUM",
            thinking_level="MEDIUM",
            input_mode="text_only",
            client=mock_client,
            log_path=chat_log,
        )

    # Validated LOW text-only prompt succeeds and attaches NO image
    t1 = exploratory_lab.send_exploratory_prompt(
        prompt="Explain safe storage of cryogenic liquids.",
        thinking_level="LOW",
        input_mode="text_only",
        image_path=Path("dataset/images/scn-001.png"),  # ignored in text_only mode
        client=mock_client,
        log_path=chat_log,
    )
    assert t1["status"] == "success"
    assert t1["modality"] == "text_only"
    assert t1["image_filename"] is None
    assert t1["actual_model_version"] == "gemini-4-hydrogen"
    assert t1["input_tokens"] == 21
    assert t1["output_tokens"] == 14
    assert t1["thinking_tokens"] == 64
    assert isinstance(mock_client.models.calls[-1]["contents"], str)

    # Image + Text prompt validates and attaches the intended image
    img_path = tmp_path / "sample.png"
    Image.new("RGB", (64, 64), color=(30, 30, 30)).save(img_path)
    saved_img, meta = exploratory_lab.save_uploaded_exploratory_image(
        img_path.read_bytes(), "sample.png", upload_dir=tmp_path / "uploads"
    )
    assert meta["mime_type"] == "image/png"

    t2 = exploratory_lab.send_exploratory_prompt(
        prompt="Describe the safety placard shown in this image.",
        thinking_level="LOW",
        input_mode="image_plus_text",
        image_path=saved_img,
        client=mock_client,
        log_path=chat_log,
    )
    assert t2["status"] == "success"
    assert t2["modality"] == "image_plus_text"
    assert t2["image_filename"] == saved_img.name
    contents = mock_client.models.calls[-1]["contents"]
    assert isinstance(contents, list) and len(contents) == 2

    # Invalid image extension rejected cleanly
    with pytest.raises(ValueError):
        exploratory_lab.save_uploaded_exploratory_image(
            b"not an image", "bad.txt", upload_dir=tmp_path / "uploads"
        )

    # Save as candidate scenario — must NOT touch official pilot/main JSONLs or datasets
    cand = exploratory_lab.save_candidate_scenario(
        prompt=t2["prompt"],
        modality=t2["modality"],
        thinking_level=t2["thinking_level"],
        response_text=t2["response_text"],
        image_filename=t2["image_filename"],
        candidate_path=cand_json,
    )
    assert cand["review_status"] == "candidate_pending_curation"
    assert cand["approved"] is False
    assert cand["is_official_dataset_scenario"] is False
    assert len(storage.load_trial_records("pilot")) == 0
    assert len(storage.load_trial_records("main")) == 0


def _nav_click(at: AppTest, page_name: str) -> None:
    btn = next(b for b in at.button if b.key == f"top_nav_{page_name}")
    btn.click().run()


def test_overview_and_annotation_empty_state_via_apptest(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(storage, "PILOT_DIR", tmp_path / "pilot")
    monkeypatch.setattr(storage, "MAIN_DIR", tmp_path / "main")
    monkeypatch.setattr(storage, "RESULTS_DIR", tmp_path)
    (tmp_path / "pilot").mkdir(parents=True, exist_ok=True)
    (tmp_path / "main").mkdir(parents=True, exist_ok=True)

    app_entry = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"
    at = AppTest.from_file(str(app_entry), default_timeout=20)
    at.run()
    assert not at.exception
    assert len(at.sidebar) == 0

    # Overview renders top navbar + 4 core stats + Study Architecture + 6 milestones + Next Step
    md_text = "\n".join(m.value for m in at.markdown)
    assert "Gemini 4 Hydrogen" in md_text
    assert "Overview" in md_text
    assert "0/96" in md_text
    assert "0/1,440" in md_text
    assert "Next Step" in md_text
    assert "60 Matched Scenario Families" in md_text
    assert "4 Thinking Levels &amp; Two-Stage Runs" in md_text
    assert "Human Coding &amp; 11 Metrics" in md_text

    # Clicking Next Step shortcut button navigates cleanly
    next_btn = next(b for b in at.button if b.key == "ov_next_btn")
    next_btn.click().run()
    assert not at.exception
    assert "Settings" in "\n".join(m.value for m in at.markdown)

    # Results & Analysis -> Annotation empty state must NOT be blank
    _nav_click(at, "Results & Analysis")
    assert not at.exception
    ra_md = "\n".join(m.value for m in at.markdown)
    assert "No experimental responses are available for annotation yet." in ra_md
    assert "Complete the pilot experiment to begin reviewing Gemini responses." in ra_md


def test_annotation_interface_and_experiment_lab_ui_with_mock_records(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(storage, "PILOT_DIR", tmp_path / "pilot")
    monkeypatch.setattr(storage, "MAIN_DIR", tmp_path / "main")
    monkeypatch.setattr(storage, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(annotation, "HUMAN_ANNOTATIONS_CSV", tmp_path / "human_annotations.csv")
    monkeypatch.setattr(annotation, "PROCESSED_RESULTS_CSV", tmp_path / "processed_results.csv")
    monkeypatch.setattr(hydrogen_client, "API_VALIDATION_STATE_PATH", tmp_path / "api_val.json")
    (tmp_path / "pilot").mkdir(parents=True, exist_ok=True)
    (tmp_path / "main").mkdir(parents=True, exist_ok=True)

    # Seed one synthetic pilot record in isolated tmp_path
    _, pilot_scens = scenario_manager.load_active_dataset("pilot")
    s0 = pilot_scens[0]
    storage.append_trial_record(
        "pilot",
        {
            "trial_id": "test-trial-001",
            "experiment_run_id": "run-test",
            "experiment_type": "pilot",
            "dataset_version": "v1.0.0-pilot",
            "dataset_hash": "hash-test",
            "scenario_id": s0["scenario_id"],
            "category": s0["category"],
            "modality": "text_only",
            "thinking_level": "LOW",
            "repetition": 1,
            "prompt": s0["text_only_prompt"],
            "image_asset_reference": None,
            "model_id": "gemini-4-hydrogen",
            "actual_model_version": "gemini-4-hydrogen",
            "status": "success",
            "response_text": "Safe test completion response.",
            "latency_ms": 145.0,
            "input_tokens": 20,
            "output_tokens": 10,
            "thinking_tokens": 30,
            "total_tokens": 60,
            "timestamp_utc": "2026-11-19T00:00:00+00:00",
        },
        also_mirror_unified=False,
    )

    app_entry = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"
    at = AppTest.from_file(str(app_entry), default_timeout=20)
    at.run()
    _nav_click(at, "Results & Analysis")
    assert not at.exception

    # Annotation UI must display the seeded response, coding controls, and raw-vs-annotated guidance
    ta_values = [ta.value for ta in at.text_area]
    assert any("Safe test completion response." in str(v) for v in ta_values)
    assert any(b.label == "Save Annotation" for b in at.button)
    assert any("Raw API latency and token usage" in i.value for i in at.info)

    # Experiment Lab UI renders exploratory notice, blocks unvalidated level, and shows Open Settings shortcut
    _nav_click(at, "Experiment Lab")
    assert not at.exception
    info_msgs = [i.value for i in at.info]
    assert any("Exploratory testing only" in m for m in info_msgs)
    send_btns = [b for b in at.button if b.label == "Send"]
    assert len(send_btns) == 1
    assert send_btns[0].disabled is True
    assert any(b.key == "lab_open_settings" for b in at.button)


def test_experiment_lab_chat_flow_research_runs_and_settings_via_apptest(tmp_path, monkeypatch) -> None:
    val_path = tmp_path / "api_val.json"
    _write_validation_state(
        val_path,
        {
            "LOW": {"validated": True, "supported_by_endpoint": True},
            "MEDIUM": {"validated": True, "supported_by_endpoint": True},
            "HIGH": {"validated": True, "supported_by_endpoint": True},
            "XHIGH": {"validated": True, "supported_by_endpoint": True},
        },
    )
    from app.pages import settings as app_settings

    monkeypatch.setattr(hydrogen_client, "API_VALIDATION_STATE_PATH", val_path)
    monkeypatch.setattr(hydrogen_client, "API_READINESS_REPORT_MD", tmp_path / "api_readiness_report.md")
    monkeypatch.setattr(app_settings, "API_READINESS_REPORT_MD", tmp_path / "api_readiness_report.md")
    monkeypatch.setattr(hydrogen_client, "API_ERROR_LOG_PATH", tmp_path / "api_errors.log")
    monkeypatch.setattr(app_settings, "API_ERROR_LOG_PATH", tmp_path / "api_errors.log")
    monkeypatch.setattr(exploratory_lab, "EXPLORATORY_CHAT_JSONL", tmp_path / "chat_history.jsonl")
    monkeypatch.setattr(exploratory_lab, "CANDIDATE_SCENARIOS_JSON", tmp_path / "candidate_scenarios.json")
    monkeypatch.setattr(storage, "PILOT_DIR", tmp_path / "pilot")
    monkeypatch.setattr(storage, "MAIN_DIR", tmp_path / "main")
    monkeypatch.setattr(storage, "RESULTS_DIR", tmp_path)
    (tmp_path / "pilot").mkdir(parents=True, exist_ok=True)
    (tmp_path / "main").mkdir(parents=True, exist_ok=True)

    mock_models = _LabMockModels()
    mock_models.get = lambda model: SimpleNamespace(
        name=f"models/{model}",
        display_name="Gemini 4 Hydrogen",
        supported_actions=["generateContent"],
        input_token_limit=1000000,
        output_token_limit=65536,
    )
    mock_client = SimpleNamespace(models=mock_models)
    monkeypatch.setattr(hydrogen_client, "build_client", lambda api_key=None, timeout_seconds=90.0: mock_client)

    app_entry = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"
    at = AppTest.from_file(str(app_entry), default_timeout=20)
    at.run()

    # 1. Experiment Lab: submit mocked prompt -> response renders -> Save as Candidate Scenario -> Clear Chat
    _nav_click(at, "Experiment Lab")
    assert not at.exception
    send_btns = [b for b in at.button if b.label == "Send"]
    assert len(send_btns) == 1 and send_btns[0].disabled is False

    at.text_area[0].set_value("Test safety prompt for exploratory chat")
    next(b for b in at.button if b.label == "Send").click().run()
    assert not at.exception
    assert not at.error
    lab_md = "\n".join(m.value for m in at.markdown)
    assert "Exploratory response from Gemini 4 Hydrogen." in lab_md
    assert "Tokens (in/out/think): 21 / 14 / 64" in lab_md

    next(b for b in at.button if b.label == "Save as Candidate Scenario").click().run()
    assert not at.exception
    cands = exploratory_lab.load_candidate_scenarios(tmp_path / "candidate_scenarios.json")
    assert len(cands) == 1
    assert cands[0]["prompt"] == "Test safety prompt for exploratory chat"
    assert len(storage.load_trial_records("pilot")) == 0
    assert len(storage.load_trial_records("main")) == 0

    next(b for b in at.button if b.label == "Clear Chat").click().run()
    assert not at.exception
    assert "Start an exploratory session" in "\n".join(m.value for m in at.markdown)

    # 2. Research Runs: verify dataset category balance, approvals queue, pilot, and main controls
    _nav_click(at, "Research Runs")
    assert not at.exception
    rr_md = "\n".join(m.value for m in at.markdown)
    assert "Benign 4 / Sensitive 4 / Unsafe 4" in rr_md
    assert "Benign 20 / Sensitive 20 / Unsafe 20" in rr_md
    assert "Dataset Approvals Queue" in rr_md
    queue_df = at.dataframe[0].value
    assert set(queue_df["Version"]).issuperset(
        {"v1.0.0-pilot", "v1.0.0-main", "v2.0.0-pilot", "v2.0.0-main", "v2.1.0-pilot", "v2.1.0-main"}
    )
    labels = {b.label for b in at.button}
    for expected_btn in (
        "Review Pilot Dataset",
        "Review Main Dataset",
        "Approve Dataset",
        "Start Pilot",
        "Start Main Experiment",
        "Pause",
        "Resume",
        "Stop",
    ):
        assert expected_btn in labels

    # 3. Settings: Verify Gemini API button executes mocked endpoint check and triggers rerun cleanly
    _nav_click(at, "Settings")
    assert not at.exception
    verify_btn = next(b for b in at.button if b.key == "settings_verify_btn")
    verify_btn.click().run()
    assert not at.exception
    assert any("Endpoint reachable" in s.value for s in at.success)
    assert any(b.label == "Generate AI Studio Integration Audit (MD)" for b in at.button)

