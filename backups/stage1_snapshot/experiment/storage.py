"""Append-only JSONL storage, resumability checks, and configuration locking."""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from config import (
    EXPERIMENT_LOG_MD,
    MAIN_DIR,
    PILOT_DIR,
    RESULTS_DIR,
    RUNTIME_CONFIG_PATH,
    ExperimentConfig,
    ensure_directories,
)

_WRITE_LOCK = threading.Lock()


class ConfigurationDriftError(RuntimeError):
    """Raised when model, SDK version, or core config changes mid-experiment."""


def get_experiment_dir(experiment_type: str) -> Path:
    """Return isolated directory for 'pilot' or 'main' experiment."""
    ensure_directories()
    exp = experiment_type.strip().lower()
    if exp == "pilot":
        return PILOT_DIR
    if exp == "main":
        return MAIN_DIR
    raise ValueError(f"Unknown experiment_type '{experiment_type}'. Use 'pilot' or 'main'.")


def get_raw_jsonl_path(experiment_type: str) -> Path:
    return get_experiment_dir(experiment_type) / "raw_responses.jsonl"


def get_state_path(experiment_type: str) -> Path:
    return get_experiment_dir(experiment_type) / "experiment_state.json"


def get_config_snapshot_path(experiment_type: str) -> Path:
    return get_experiment_dir(experiment_type) / "config_snapshot.json"


def compute_config_fingerprint(config: ExperimentConfig, sdk_version: str) -> str:
    """Deterministic fingerprint of critical experimental parameters."""
    payload = {
        "model_id": config.model_id,
        "sdk_version": sdk_version,
        "api_version": config.api_version,
        "max_output_tokens": config.max_output_tokens,
        "include_thoughts": config.include_thoughts,
        "system_instruction": config.system_instruction.strip(),
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_or_lock_config(
    experiment_type: str, config: ExperimentConfig, sdk_version: str
) -> Dict[str, Any]:
    """Lock config on first run or verify no mid-experiment drift occurred."""
    snap_path = get_config_snapshot_path(experiment_type)
    current_fp = compute_config_fingerprint(config, sdk_version)
    snapshot = {
        "experiment_type": experiment_type,
        "model_id": config.model_id,
        "sdk_version": sdk_version,
        "api_version": config.api_version,
        "max_output_tokens": config.max_output_tokens,
        "include_thoughts": config.include_thoughts,
        "system_instruction": config.system_instruction.strip(),
        "fingerprint": current_fp,
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if not snap_path.exists():
        snap_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        return snapshot

    existing = json.loads(snap_path.read_text(encoding="utf-8"))
    if existing.get("fingerprint") != current_fp:
        diffs: List[str] = []
        for field in (
            "model_id",
            "sdk_version",
            "api_version",
            "max_output_tokens",
            "include_thoughts",
            "system_instruction",
        ):
            if existing.get(field) != snapshot.get(field):
                diffs.append(
                    f"{field}: locked={existing.get(field)!r} vs current={snapshot.get(field)!r}"
                )
        raise ConfigurationDriftError(
            "Experiment configuration or API/SDK version changed during "
            f"{experiment_type} experiment ({'; '.join(diffs)}). Execution halted to "
            "preserve experimental integrity."
        )
    return existing


def append_trial_record(
    experiment_type: str,
    record: Dict[str, Any],
    also_mirror_unified: bool = True,
) -> None:
    """Append one trial record to append-only JSONL without overwriting existing records."""
    ensure_directories()
    target_path = get_raw_jsonl_path(experiment_type)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _WRITE_LOCK:
        with target_path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
        if also_mirror_unified:
            unified_path = RESULTS_DIR / "raw_responses.jsonl"
            with unified_path.open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()


def load_trial_records(experiment_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Read raw trial records from pilot, main, or both."""
    records: List[Dict[str, Any]] = []
    if experiment_type is None:
        for exp in ("pilot", "main"):
            records.extend(load_trial_records(exp))
        return records

    path = get_raw_jsonl_path(experiment_type)
    if not path.exists():
        return records

    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError:
                # Preserve visibility of corrupt lines without crashing reader
                records.append(
                    {
                        "trial_id": f"corrupt-{experiment_type}-{line_no}",
                        "experiment_type": experiment_type,
                        "status": "corrupt_record",
                        "raw_line": stripped,
                    }
                )
    return records


def condition_key(
    scenario_id: str, modality: str, thinking_level: str, repetition: int
) -> Tuple[str, str, str, int]:
    """Canonical key uniquely identifying an experimental condition."""
    return (
        str(scenario_id),
        str(modality),
        str(thinking_level).upper(),
        int(repetition),
    )


def get_completed_conditions(
    experiment_type: str, include_blocked_as_completed: bool = True
) -> Set[Tuple[str, str, str, int]]:
    """Identify already completed experimental conditions to prevent duplicates on resume."""
    valid_statuses = {"success", "blocked"} if include_blocked_as_completed else {"success"}
    completed: Set[Tuple[str, str, str, int]] = set()
    for rec in load_trial_records(experiment_type):
        if rec.get("status") in valid_statuses:
            try:
                completed.add(
                    condition_key(
                        rec["scenario_id"],
                        rec["modality"],
                        rec["thinking_level"],
                        int(rec["repetition"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return completed


def save_runtime_config(config: ExperimentConfig) -> None:
    """Persist runtime non-secret experiment settings."""
    ensure_directories()
    RUNTIME_CONFIG_PATH.write_text(
        json.dumps(config.to_safe_dict(), indent=2), encoding="utf-8"
    )


def load_runtime_config() -> ExperimentConfig:
    """Load persisted runtime settings or return defaults."""
    ensure_directories()
    if not RUNTIME_CONFIG_PATH.exists():
        return ExperimentConfig()
    try:
        data = json.loads(RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))
        valid_fields = ExperimentConfig.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return ExperimentConfig(**filtered)
    except Exception:
        return ExperimentConfig()


def load_experiment_state(experiment_type: str) -> Dict[str, Any]:
    """Read persisted execution state for pilot or main experiment."""
    path = get_state_path(experiment_type)
    if not path.exists():
        return {
            "experiment_type": experiment_type,
            "status": "IDLE",
            "approved": False,
            "total_planned": 0,
            "completed_count": 0,
            "failed_count": 0,
            "skipped_existing": 0,
            "last_updated_utc": None,
            "history": [],
        }
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "experiment_type": experiment_type,
            "status": "IDLE",
            "approved": False,
            "total_planned": 0,
            "completed_count": 0,
            "failed_count": 0,
            "skipped_existing": 0,
            "last_updated_utc": None,
            "history": [],
        }


def save_experiment_state(experiment_type: str, state: Dict[str, Any]) -> None:
    """Persist execution state JSON atomically."""
    path = get_state_path(experiment_type)
    state["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def append_experiment_log(message: str) -> None:
    """Append timestamped entry to research/experiment_log.md."""
    ensure_directories()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    if not EXPERIMENT_LOG_MD.exists():
        EXPERIMENT_LOG_MD.write_text(
            "# Experiment Execution Log\n\n", encoding="utf-8"
        )
    with _WRITE_LOCK:
        with EXPERIMENT_LOG_MD.open("a", encoding="utf-8") as fh:
            fh.write(f"- **[{ts}]** {message}\n")


def export_records_to_csv(experiment_type: Optional[str], output_csv: Path) -> Path:
    """Export raw JSONL records to flat CSV without mutating raw JSONL."""
    records = load_trial_records(experiment_type)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        pd.DataFrame().to_csv(output_csv, index=False)
        return output_csv
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    return output_csv
