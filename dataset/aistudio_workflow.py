"""Google AI Studio Logs & Datasets export/import interoperability helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import (
    DEFAULT_MODEL_ID,
    MAIN_RAW_JSONL,
    PILOT_RAW_JSONL,
    RESEARCH_DIR,
    SOFTWARE_VERSION,
    get_api_key,
)
from dataset.scenario_manager import load_registry, verify_version_integrity
from experiment.hydrogen_client import get_sdk_version, load_validation_state
from experiment.storage import load_trial_records


def format_observations_for_aistudio_jsonl(
    experiment_type: Optional[str] = None,
) -> str:
    """Convert raw experimental observations into Google AI Studio Batch/Dataset JSONL lines."""
    records = load_trial_records(experiment_type)
    lines: List[str] = []
    for rec in records:
        entry: Dict[str, Any] = {
            "request": {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": rec.get("prompt", "")}],
                    }
                ],
                "generationConfig": {
                    "thinkingConfig": {
                        "thinkingLevel": rec.get("thinking_level", "HIGH")
                    }
                },
            },
            "response": {
                "text": rec.get("response_text", ""),
                "modelVersion": rec.get("actual_model_version"),
                "finishReason": rec.get("finish_reason"),
            },
            "metadata": {
                "experiment_run_id": rec.get("experiment_run_id"),
                "trial_id": rec.get("trial_id"),
                "experiment_type": rec.get("experiment_type"),
                "dataset_version": rec.get("dataset_version"),
                "dataset_hash": rec.get("dataset_hash"),
                "scenario_id": rec.get("scenario_id"),
                "category": rec.get("category"),
                "modality": rec.get("modality"),
                "thinking_level": rec.get("thinking_level"),
                "repetition": rec.get("repetition"),
                "status": rec.get("status"),
                "latency_ms": rec.get("latency_ms"),
                "input_tokens": rec.get("input_tokens"),
                "output_tokens": rec.get("output_tokens"),
                "thinking_tokens": rec.get("thinking_tokens"),
            },
        }
        lines.append(json.dumps(entry, ensure_ascii=False))
    return "\n".join(lines) + ("\n" if lines else "")


def write_aistudio_export_jsonl(
    output_path: Path, experiment_type: Optional[str] = None
) -> Path:
    """Write AI Studio-compatible JSONL export file for secondary curation/audit."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = format_observations_for_aistudio_jsonl(experiment_type)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def validate_aistudio_integration() -> Dict[str, Any]:
    """Audit real Google AI Studio workflow separation, formats, and logging observability honestly."""
    registry = load_registry()
    val_state = load_validation_state()
    api_key_present = bool(get_api_key())
    levels = val_state.get("levels", {})
    live_calls_recorded = any(bool(v.get("last_tested_utc")) for v in levels.values())
    validated_levels = [lvl for lvl, info in levels.items() if info.get("validated")]

    pilot_obs = load_trial_records("pilot")
    main_obs = load_trial_records("main")

    # Never claim AI Studio UI logging is active without live verified requests + UI confirmation
    if not api_key_present:
        logging_status = "UNVERIFIED (GEMINI_API_KEY not configured in runtime)"
    elif not live_calls_recorded:
        logging_status = "UNVERIFIED (no live GenerateContent requests sent yet)"
    else:
        logging_status = (
            "REQUIRES VISUAL CONFIRMATION IN AI STUDIO LOGS UI "
            "(live SDK calls executed locally; confirm project logging toggle in Google AI Studio)"
        )

    versions_integrity = [
        {
            "dataset_version": v["dataset_version"],
            "scope": v["dataset_scope"],
            "approval_status": v["researcher_approval_status"],
            "intact": verify_version_integrity(v["dataset_version"])["intact"],
            "sha256": v.get("sha256"),
        }
        for v in registry.get("versions", [])
    ]

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "software_version": SOFTWARE_VERSION,
        "sdk_version": get_sdk_version(),
        "model_id": DEFAULT_MODEL_ID,
        "api_key_present": api_key_present,
        "validated_thinking_levels": validated_levels,
        "supported_import_formats": ["JSON", "JSONL", "CSV"],
        "supported_observation_exports": ["Append-Only JSONL", "CSV", "AI Studio JSONL"],
        "scenario_vs_observation_separation": {
            "scenario_datasets_dir": "dataset/versions/",
            "pilot_observation_archive": str(PILOT_RAW_JSONL),
            "main_observation_archive": str(MAIN_RAW_JSONL),
            "pilot_observations_count": len(pilot_obs),
            "main_observations_count": len(main_obs),
        },
        "undocumented_ui_dataset_api_used": False,
        "ai_studio_logging_status": logging_status,
        "version_integrity": versions_integrity,
    }


def write_integration_validation_report(
    output_path: Path = RESEARCH_DIR / "aistudio_integration_validation_report.md",
) -> Path:
    """Write factual Google AI Studio integration & runner readiness validation report."""
    audit = validate_aistudio_integration()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Google AI Studio Integration & Experiment Workflow Validation Report",
        "",
        f"- **Generated (UTC):** {audit['generated_at_utc']}",
        f"- **Software Version:** `{audit['software_version']}`",
        f"- **Google GenAI SDK:** `{audit['sdk_version']}`",
        f"- **Target Model:** `{audit['model_id']}`",
        "",
        "## 1. Separation of Scenario Datasets vs. Experimental Observations",
        "- **Scenario Datasets (Versioned Inputs):** `dataset/versions/<version>/scenarios.json` & `.jsonl` (read-only inside Streamlit; curated in Google AI Studio and imported via JSON, JSONL, or CSV).",
        f"- **Pilot Observation Archive (Authoritative Append-Only JSONL):** `{audit['scenario_vs_observation_separation']['pilot_observation_archive']}` (current records: `{audit['scenario_vs_observation_separation']['pilot_observations_count']}`).",
        f"- **Main Observation Archive (Authoritative Append-Only JSONL):** `{audit['scenario_vs_observation_separation']['main_observation_archive']}` (current records: `{audit['scenario_vs_observation_separation']['main_observations_count']}`).",
        "",
        "## 2. Google AI Studio API & Logging Verification Status",
        f"- **`GEMINI_API_KEY` Configured:** `{audit['api_key_present']}`",
        f"- **Validated Thinking Levels (Live Endpoint):** `{audit['validated_thinking_levels'] or 'None yet'}`",
        f"- **Google AI Studio Project Logging Status:** **{audit['ai_studio_logging_status']}**",
        "- **Undocumented UI Dataset Retrieval APIs Used:** `False` (the official `google-genai` SDK does not expose retrieval of arbitrary UI-named AI Studio Datasets; scenario datasets are imported explicitly as versioned JSON/JSONL/CSV files).",
        "",
        "## 3. Immutable Scenario Version Integrity",
    ]
    for v in audit["version_integrity"]:
        lines.append(
            f"- `{v['dataset_version']}` (`{v['scope']}`): approval=`{v['approval_status']}`, "
            f"sha256_intact=`{v['intact']}`, sha256=`{v['sha256']}`"
        )
    lines.extend(
        [
            "",
            "## 4. Next Verification Steps",
            "1. Configure `GEMINI_API_KEY` and enable **Project Logging** in Google AI Studio.",
            "2. Run `.venv/bin/python validate_thinking_levels.py --include-multimodal` to validate `LOW`, `MEDIUM`, `HIGH`, and `XHIGH` against `models/gemini-4-hydrogen`.",
            "3. Confirm the resulting probe requests appear in the Google AI Studio **Logs** UI.",
            "4. Grant explicit Researcher Approval to `v1.0.0-pilot` before running the 96-call Pilot, and wait for pilot review before executing the 1,440-call Main experiment.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
