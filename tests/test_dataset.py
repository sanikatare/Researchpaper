"""Tests for Part A 14-field schema, versioned AI Studio import, validation, and exclusions."""

from __future__ import annotations

import json

import pytest

from config import REQUIRED_SCENARIO_FIELDS
from dataset import scenario_manager
from dataset.scenario_manager import (
    check_image_valid,
    check_law_exclusion,
    find_duplicates,
    load_active_dataset,
    load_registry,
    validate_dataset,
    validate_scenario,
)


def test_versioned_main_and_pilot_datasets_valid_and_pending_approval() -> None:
    reg = load_registry()
    assert reg["active_pilot_version"] == "v1.0.0-pilot"
    assert reg["active_main_version"] == "v1.0.0-main"

    pilot_entry, pilot_scenarios = load_active_dataset("pilot")
    main_entry, main_scenarios = load_active_dataset("main")

    assert pilot_entry is not None and main_entry is not None
    assert len(pilot_scenarios) == 12
    assert len(main_scenarios) == 60

    # Technical validation must NOT auto-approve
    assert pilot_entry["technical_validation_status"] == "valid"
    assert main_entry["technical_validation_status"] == "valid"
    assert pilot_entry["researcher_approval_status"] == "pending_review"
    assert main_entry["researcher_approval_status"] == "pending_review"

    main_val = validate_dataset(main_scenarios, expected_scope="main")
    assert main_val["dataset_valid"] is True
    assert main_val["category_counts"] == {
        "clearly_benign": 20,
        "sensitive_looking_legitimate": 20,
        "genuinely_unsafe": 20,
    }

    pilot_val = validate_dataset(pilot_scenarios, expected_scope="pilot")
    assert pilot_val["dataset_valid"] is True
    assert pilot_val["category_counts"] == {
        "clearly_benign": 4,
        "sensitive_looking_legitimate": 4,
        "genuinely_unsafe": 4,
    }


def test_all_60_scenarios_have_required_14_fields_and_valid_images() -> None:
    _, main_scenarios = load_active_dataset("main")
    for s in main_scenarios:
        for field in REQUIRED_SCENARIO_FIELDS:
            assert field in s and str(s[field]).strip(), f"{s.get('scenario_id')} missing {field}"
        ok, msg = check_image_valid(s["image_asset_reference"])
        assert ok, f"{s['scenario_id']} image failed: {msg}"


def test_ai_studio_jsonl_import_immutable_and_non_overwriting(tmp_path, monkeypatch) -> None:
    _, pilot_scenarios = load_active_dataset("pilot")
    assert len(pilot_scenarios) == 12

    monkeypatch.setattr(scenario_manager, "DATASET_REGISTRY_PATH", tmp_path / "reg.json")
    monkeypatch.setattr(scenario_manager, "DATASET_VERSIONS_DIR", tmp_path / "versions")
    (tmp_path / "versions").mkdir(parents=True, exist_ok=True)

    # Build JSONL export just like AI Studio Batch/Logs JSONL
    jsonl_bytes = "\n".join(json.dumps(s) for s in pilot_scenarios).encode("utf-8")

    entry = scenario_manager.import_dataset_version(
        raw_bytes=jsonl_bytes,
        filename="aistudio_export.jsonl",
        dataset_name="Pilot-AIStudio-Rev2",
        dataset_version="v1.1.0-pilot",
        dataset_scope="pilot",
    )
    assert entry["technical_validation_status"] == "valid"
    assert entry["researcher_approval_status"] == "pending_review"

    # Attempting to overwrite same version must fail
    with pytest.raises(ValueError, match="already exists"):
        scenario_manager.import_dataset_version(
            raw_bytes=jsonl_bytes,
            filename="aistudio_export.jsonl",
            dataset_name="Pilot-AIStudio-Rev2",
            dataset_version="v1.1.0-pilot",
            dataset_scope="pilot",
        )

    # Approve and activate
    approved = scenario_manager.set_version_approval("v1.1.0-pilot", "approved")
    assert approved["researcher_approval_status"] == "approved"
    scenario_manager.set_active_version("pilot", "v1.1.0-pilot")
    reg = scenario_manager.load_registry()
    assert reg["active_pilot_version"] == "v1.1.0-pilot"


def test_law_exclusion_and_duplicate_detection() -> None:
    bad = {
        "scenario_id": "SCN-BAD",
        "category": "clearly_benign",
        "scenario_title": "Legal Advice Test",
        "scenario_description": "Draft a legal advice memo for a lawsuit",
        "underlying_user_intent": "Obtain legal advice on contract law.",
        "text_only_prompt": "Give me legal advice on this contract law dispute.",
        "image_text_prompt": "Review this courtroom subpoena.",
        "image_asset_reference": "scn-001.png",
        "image_provenance": "Test",
        "expected_safe_behavior": "N/A - excluded domain",
        "expected_response_classification": "safe_completion",
        "classification_rationale": "Test rationale",
        "dataset_version": "v-test",
        "validation_status": "pending",
    }
    hits = check_law_exclusion(bad)
    assert len(hits) >= 2
    val = validate_scenario(bad)
    assert val["technical_valid"] is False

    dups = find_duplicates([bad, dict(bad, scenario_id="SCN-BAD-2")])
    assert len(dups["duplicate_prompts"]) == 1
    assert len(dups["duplicate_titles"]) == 1


def test_validation_report_pass_fail_needs_review_and_no_silent_correction() -> None:
    _, main_scenarios = load_active_dataset("main")

    # Unapproved dataset -> technical checks valid, overall report classification NEEDS HUMAN REVIEW
    rep_pending = validate_dataset(main_scenarios, expected_scope="main", dataset_approved=False)
    assert rep_pending["dataset_valid"] is True
    assert rep_pending["overall_classification"] == "NEEDS HUMAN REVIEW"
    assert rep_pending["classification_counts"]["FAIL"] == 0
    assert rep_pending["classification_counts"]["NEEDS HUMAN REVIEW"] == 60
    assert len(rep_pending["checks_matrix"]) == 13

    # Approved dataset -> approved scenarios classify PASS (none FAIL)
    rep_approved = validate_dataset(main_scenarios, expected_scope="main", dataset_approved=True)
    assert rep_approved["classification_counts"]["FAIL"] == 0
    assert rep_approved["classification_counts"]["PASS"] == 60
    assert rep_approved["overall_classification"] == "PASS"

    # Missing fields must NEVER be silently corrected — must classify as FAIL
    incomplete = dict(main_scenarios[0])
    incomplete.pop("image_provenance", None)
    incomplete["scenario_title"] = ""
    res_inc = validate_scenario(incomplete, dataset_approved=True)
    assert res_inc["report_classification"] == "FAIL"
    assert any("image_provenance" in msg for msg in res_inc["issues"])
    assert any("scenario_title" in msg for msg in res_inc["issues"])

    # Unrelated modalities -> FAIL
    mismatch = dict(main_scenarios[0])
    mismatch["image_requirements"] = ""
    mismatch["scenario_description"] = "Unrelated astronomy topic"
    mismatch["image_text_prompt"] = "Analyze the orbital velocity of Neptune around the solar ecliptic."
    res_mis = validate_scenario(mismatch, dataset_approved=True)
    assert res_mis["report_classification"] == "FAIL"

    # Near-duplicate prompt -> flagged in find_duplicates & NEEDS HUMAN REVIEW
    s1 = dict(main_scenarios[0])
    s2 = dict(main_scenarios[0])
    s2["scenario_id"] = "SCN-099"
    s2["scenario_title"] = "Oven Thermometer Copy"
    s2["text_only_prompt"] = s1["text_only_prompt"].replace("230 degrees", "235 degrees")
    dups = find_duplicates([s1, s2])
    assert len(dups["near_duplicates"]) == 1


def test_immutable_versions_and_approved_sha256_integrity(tmp_path, monkeypatch) -> None:
    _, pilot_scenarios = load_active_dataset("pilot")
    monkeypatch.setattr(scenario_manager, "DATASET_REGISTRY_PATH", tmp_path / "reg.json")
    monkeypatch.setattr(scenario_manager, "DATASET_VERSIONS_DIR", tmp_path / "versions")
    (tmp_path / "versions").mkdir(parents=True, exist_ok=True)

    raw = json.dumps({"scenarios": pilot_scenarios}).encode("utf-8")
    entry = scenario_manager.import_dataset_version(
        raw_bytes=raw,
        filename="pilot.json",
        dataset_name="Pilot-SHA-Test",
        dataset_version="v1.0.0-pilot",
        dataset_scope="pilot",
    )
    assert entry["sha256"] and len(entry["sha256"]) == 64
    assert entry["approved_sha256"] is None
    assert (tmp_path / "versions" / "v1.0.0-pilot" / "validation_report.json").exists()

    approved = scenario_manager.set_version_approval("v1.0.0-pilot", "approved")
    assert approved["approved_sha256"] == entry["sha256"]
    assert scenario_manager.verify_version_integrity("v1.0.0-pilot")["intact"] is True

    # Prior version remains intact when a second immutable version is imported
    entry2 = scenario_manager.import_dataset_version(
        raw_bytes=raw,
        filename="pilot_v2.json",
        dataset_name="Pilot-SHA-Test-V2",
        dataset_version="v1.1.0-pilot",
        dataset_scope="pilot",
    )
    assert len(scenario_manager.load_registry()["versions"]) == 2
    assert entry2["researcher_approval_status"] == "pending_review"

    # Tampering with immutable file on disk is caught by verify_version_integrity
    (tmp_path / "versions" / "v1.0.0-pilot" / "scenarios.json").write_text("{}", encoding="utf-8")
    integrity_after = scenario_manager.verify_version_integrity("v1.0.0-pilot")
    assert integrity_after["intact"] is False


def test_csv_dataset_import_and_aistudio_integration_report(tmp_path, monkeypatch) -> None:
    import csv
    import io
    from dataset.aistudio_workflow import validate_aistudio_integration, write_integration_validation_report

    _, pilot_scenarios = load_active_dataset("pilot")
    buf = io.StringIO()
    cols = list(pilot_scenarios[0].keys())
    writer = csv.DictWriter(buf, fieldnames=cols)
    writer.writeheader()
    for row in pilot_scenarios:
        writer.writerow(row)
    csv_bytes = buf.getvalue().encode("utf-8")

    monkeypatch.setattr(scenario_manager, "DATASET_REGISTRY_PATH", tmp_path / "reg.json")
    monkeypatch.setattr(scenario_manager, "DATASET_VERSIONS_DIR", tmp_path / "versions")
    (tmp_path / "versions").mkdir(parents=True, exist_ok=True)

    entry = scenario_manager.import_dataset_version(
        raw_bytes=csv_bytes,
        filename="aistudio_pilot.csv",
        dataset_name="Pilot-CSV-Import",
        dataset_version="v1.2.0-pilot",
        dataset_scope="pilot",
    )
    assert entry["scenario_count"] == 12
    assert entry["technical_validation_status"] == "valid"

    audit = validate_aistudio_integration()
    assert audit["undocumented_ui_dataset_api_used"] is False
    assert "CSV" in audit["supported_import_formats"]

    rep_path = write_integration_validation_report(tmp_path / "integration_report.md")
    assert rep_path.exists()
    text = rep_path.read_text(encoding="utf-8")
    assert "Append-Only JSONL" in text
    assert "Undocumented UI Dataset Retrieval APIs Used" in text


def test_per_scenario_review_preserves_sha256_and_blocks_rejected_approval(tmp_path, monkeypatch) -> None:
    _, pilot_scenarios = load_active_dataset("pilot")
    monkeypatch.setattr(scenario_manager, "DATASET_REGISTRY_PATH", tmp_path / "reg.json")
    monkeypatch.setattr(scenario_manager, "DATASET_VERSIONS_DIR", tmp_path / "versions")
    (tmp_path / "versions").mkdir(parents=True, exist_ok=True)

    raw = json.dumps({"scenarios": pilot_scenarios}).encode("utf-8")
    entry = scenario_manager.import_dataset_version(
        raw_bytes=raw,
        filename="pilot.json",
        dataset_name="Pilot-Review-Test",
        dataset_version="v1.3.0-pilot",
        dataset_scope="pilot",
    )
    orig_sha = entry["sha256"]
    first_sid = pilot_scenarios[0]["scenario_id"]

    rev = scenario_manager.record_scenario_review(
        "v1.3.0-pilot", first_sid, "approved", reviewer_id="researcher_01", notes="Verified clean"
    )
    assert rev["decision"] == "approved"
    assert rev["reviewer_id"] == "researcher_01"
    assert rev["reviewed_at_utc"]
    assert scenario_manager.verify_version_integrity("v1.3.0-pilot")["actual_sha256"] == orig_sha

    # Rejecting a scenario blocks version approval until a new version is curated
    scenario_manager.record_scenario_review(
        "v1.3.0-pilot", first_sid, "rejected", reviewer_id="researcher_01", notes="Needs rewording"
    )
    with pytest.raises(ValueError, match="rejected"):
        scenario_manager.set_version_approval("v1.3.0-pilot", "approved", reviewer_id="researcher_01")

    # Reverting scenario to approved allows version approval with reviewer_id recorded
    scenario_manager.record_scenario_review(
        "v1.3.0-pilot", first_sid, "approved", reviewer_id="researcher_01", notes="Re-verified"
    )
    approved = scenario_manager.set_version_approval("v1.3.0-pilot", "approved", reviewer_id="researcher_01")
    assert approved["reviewer_id"] == "researcher_01"
    assert approved["approved_sha256"] == orig_sha


