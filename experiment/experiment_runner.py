"""Trial plan builder and resumable experiment runner."""

from __future__ import annotations

import hashlib
import json
import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import (
    MAIN_TARGET_REPETITIONS,
    PILOT_TARGET_REPETITIONS,
    SOFTWARE_VERSION,
    ExperimentConfig,
)
from dataset.scenario_manager import get_version_entry, verify_version_integrity
from experiment.hydrogen_client import (
    TrialCallResult,
    get_sdk_version,
    load_validation_state,
)
from experiment.multimodal_runner import (
    MultimodalPayload,
    run_multimodal_trial_with_metadata,
)
from experiment.storage import (
    append_experiment_log,
    append_trial_record,
    compute_config_fingerprint,
    condition_key,
    get_completed_conditions,
    verify_or_lock_config,
)


@dataclass
class PlannedTrial:
    """Single scheduled experimental condition."""

    scenario: Dict[str, Any]
    modality: str
    thinking_level: str
    repetition: int


def compute_scenarios_sha256(scenarios: List[Dict[str, Any]]) -> str:
    """Compute canonical SHA-256 hash for a list of scenarios."""
    ordered = sorted(scenarios, key=lambda s: str(s.get("scenario_id", "")))
    encoded = json.dumps(ordered, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_trial_plan(
    scenarios: List[Dict[str, Any]],
    modalities: List[str],
    thinking_levels: List[str],
    repetitions: int,
    randomize_order: bool = True,
    random_seed: int = 42,
) -> List[PlannedTrial]:
    """Generate deterministic full-factorial schedule across scenarios, modalities, levels, reps."""
    ordered_scenarios = sorted(scenarios, key=lambda s: str(s.get("scenario_id", "")))
    plan: List[PlannedTrial] = []
    for rep in range(1, repetitions + 1):
        for scenario in ordered_scenarios:
            for modality in modalities:
                for level in thinking_levels:
                    plan.append(
                        PlannedTrial(
                            scenario=scenario,
                            modality=modality,
                            thinking_level=level.upper(),
                            repetition=rep,
                        )
                    )
    if randomize_order:
        rng = random.Random(random_seed)
        rng.shuffle(plan)
    return plan


def filter_pending_trials(
    experiment_type: str,
    plan: List[PlannedTrial],
    include_blocked_as_completed: bool = True,
) -> Tuple[List[PlannedTrial], int]:
    """Filter out conditions that already completed successfully to prevent duplicates."""
    completed = get_completed_conditions(
        experiment_type, include_blocked_as_completed=include_blocked_as_completed
    )
    pending: List[PlannedTrial] = []
    skipped = 0
    for trial in plan:
        key = condition_key(
            trial.scenario["scenario_id"],
            trial.modality,
            trial.thinking_level,
            trial.repetition,
        )
        if key in completed:
            skipped += 1
        else:
            pending.append(trial)
    return pending, skipped


def execute_single_planned_trial(
    experiment_type: str,
    trial: PlannedTrial,
    config: ExperimentConfig,
    experiment_run_id: Optional[str] = None,
    dataset_hash: Optional[str] = None,
    config_fingerprint: Optional[str] = None,
    api_key: Optional[str] = None,
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run one planned trial, append raw record with full provenance to JSONL, and return it."""
    scenario = trial.scenario
    sid = str(scenario["scenario_id"])
    category = str(scenario["category"])
    run_id = experiment_run_id or f"{experiment_type}-exp-{uuid.uuid4().hex[:10]}"
    trial_id = (
        f"{experiment_type}-{sid}-{trial.modality}-{trial.thinking_level}"
        f"-r{trial.repetition}-{uuid.uuid4().hex[:8]}"
    )

    bundle: Optional[MultimodalPayload] = None
    try:
        bundle, call_res = run_multimodal_trial_with_metadata(
            scenario=scenario,
            modality=trial.modality,
            thinking_level=trial.thinking_level,
            config=config,
            api_key=api_key,
            client=client,
        )
        prompt = bundle.prompt
        img_str = str(bundle.image_path) if bundle.image_path else None
    except Exception as exc:
        # Never silently discard failed trials
        prompt = str(
            scenario.get(
                "text_only_prompt"
                if trial.modality == "text_only"
                else "image_text_prompt",
                scenario.get("text_prompt", ""),
            )
        )
        img_str = scenario.get("image_asset_reference") or scenario.get("image_path")
        call_res = TrialCallResult(
            model_id=config.model_id,
            actual_model_version=None,
            thinking_level=trial.thinking_level,
            modality=trial.modality,
            status="error",
            response_text="",
            thought_summary=None,
            finish_reason=None,
            block_reason=None,
            latency_ms=0.0,
            input_tokens=None,
            output_tokens=None,
            thinking_tokens=None,
            total_tokens=None,
            response_id=None,
            error_message=str(exc),
            error_code=type(exc).__name__,
            retry_count=0,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            sdk_version=get_sdk_version(),
        )

    sdk_ver = call_res.sdk_version or get_sdk_version()
    is_mm = trial.modality == "image_plus_text"
    asset_ref = (
        bundle.image_asset_reference
        if bundle is not None
        else (scenario.get("image_asset_reference") or scenario.get("image_path"))
    )
    record: Dict[str, Any] = {
        "experiment_run_id": run_id,
        "trial_id": trial_id,
        "experiment_type": experiment_type,
        "dataset_version": (bundle.dataset_version if bundle is not None else scenario.get("dataset_version")),
        "dataset_hash": dataset_hash,
        "scenario_id": sid,
        "category": category,
        "expected_classification": scenario.get("expected_response_classification")
        or scenario.get("expected_classification"),
        "modality": trial.modality,
        "thinking_level": trial.thinking_level,
        "repetition": trial.repetition,
        "random_seed": config.random_seed,
        "randomize_order": config.randomize_order,
        "prompt": prompt,
        "image_path": img_str if is_mm else None,
        "image_asset_reference": asset_ref if is_mm else None,
        "image_mime_type": bundle.image_mime_type if (bundle is not None and is_mm) else None,
        "image_size_bytes": bundle.image_size_bytes if (bundle is not None and is_mm) else None,
        "attachment_verified": bool(bundle.attachment_verified and is_mm) if bundle is not None else False,
        "semantic_match_status": bundle.semantic_match_status if bundle is not None else "UNCHECKED",
        "response_text": call_res.response_text,
        "thought_summary": call_res.thought_summary,
        "finish_reason": call_res.finish_reason,
        "block_reason": call_res.block_reason,
        "latency_ms": call_res.latency_ms,
        "input_tokens": call_res.input_tokens,
        "output_tokens": call_res.output_tokens,
        "thinking_tokens": call_res.thinking_tokens,
        "total_tokens": call_res.total_tokens,
        "model_id": call_res.model_id,
        "actual_model_version": call_res.actual_model_version,
        "response_id": call_res.response_id,
        "software_version": SOFTWARE_VERSION,
        "sdk_version": sdk_ver,
        "config_fingerprint": config_fingerprint
        or compute_config_fingerprint(config, sdk_ver),
        "config_snapshot": config.to_safe_dict(),
        "timestamp_utc": call_res.timestamp_utc,
        "status": call_res.status,
        "error_message": call_res.error_message,
        "error_code": call_res.error_code,
        "retry_count": call_res.retry_count,
    }
    append_trial_record(experiment_type, record, also_mirror_unified=True)
    return record


def run_experiment_batch(
    experiment_type: str,
    scenarios: List[Dict[str, Any]],
    config: ExperimentConfig,
    should_pause: Optional[Callable[[], bool]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    on_trial_complete: Optional[Callable[[Dict[str, Any], int, int], None]] = None,
    api_key: Optional[str] = None,
    client: Optional[Any] = None,
    dataset_version: Optional[str] = None,
    dataset_hash: Optional[str] = None,
    experiment_run_id: Optional[str] = None,
    require_approved_dataset: bool = False,
    require_validated_levels: bool = False,
) -> Dict[str, Any]:
    """Execute pending trials synchronously with config locking, resumability, and pause/stop hooks."""
    errors = config.validate()
    if errors:
        raise ValueError("Invalid configuration: " + "; ".join(errors))
    if not scenarios:
        raise ValueError("Cannot execute experiment with zero scenarios.")

    resolved_ver = dataset_version or (scenarios[0].get("dataset_version") if scenarios else None)
    resolved_hash = dataset_hash

    if require_approved_dataset:
        if experiment_type == "pilot" and config.repetitions != PILOT_TARGET_REPETITIONS:
            raise ValueError(
                f"Pilot experiment requires repetitions={PILOT_TARGET_REPETITIONS} (96-call design), "
                f"got {config.repetitions}."
            )
        if experiment_type == "main" and config.repetitions != MAIN_TARGET_REPETITIONS:
            raise ValueError(
                f"Main experiment requires repetitions={MAIN_TARGET_REPETITIONS} (1,440-call design), "
                f"got {config.repetitions}."
            )
        if not resolved_ver:
            raise PermissionError("Approved versioned scenario dataset is required.")
        entry = get_version_entry(str(resolved_ver))
        if entry is None:
            raise PermissionError(f"Dataset version '{resolved_ver}' is not registered.")
        if entry.get("researcher_approval_status") != "approved":
            raise PermissionError(
                f"Dataset version '{resolved_ver}' is '{entry.get('researcher_approval_status')}' "
                "and requires explicit researcher approval before running."
            )
        integrity = verify_version_integrity(str(resolved_ver))
        if not integrity["intact"]:
            raise PermissionError(
                integrity.get("error") or "Dataset SHA-256 integrity verification failed."
            )
        resolved_hash = entry.get("approved_sha256") or entry.get("sha256")
    elif not resolved_hash and resolved_ver:
        entry = get_version_entry(str(resolved_ver))
        if entry is not None:
            resolved_hash = entry.get("approved_sha256") or entry.get("sha256")

    if not resolved_hash:
        resolved_hash = compute_scenarios_sha256(scenarios)

    if require_validated_levels:
        val_state = load_validation_state()
        levels_state = val_state.get("levels", {})
        unvalidated = [
            lvl
            for lvl in config.thinking_levels
            if not (
                levels_state.get(lvl, {}).get("validated")
                and levels_state.get(lvl, {}).get("supported_by_endpoint") is True
            )
        ]
        if unvalidated:
            raise RuntimeError(
                "Thinking level(s) not validated against the live endpoint: "
                + ", ".join(unvalidated)
            )

    sdk_ver = get_sdk_version()
    locked_snap = verify_or_lock_config(
        experiment_type,
        config,
        sdk_ver,
        dataset_version=str(resolved_ver) if resolved_ver else None,
        dataset_hash=resolved_hash,
    )
    config_fp = locked_snap["fingerprint"]

    run_id = experiment_run_id or (
        f"{experiment_type}-exp-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    )

    plan = build_trial_plan(
        scenarios=scenarios,
        modalities=config.modalities,
        thinking_levels=config.thinking_levels,
        repetitions=config.repetitions,
        randomize_order=config.randomize_order,
        random_seed=config.random_seed,
    )
    pending, skipped_existing = filter_pending_trials(experiment_type, plan)
    total_planned = len(plan)

    append_experiment_log(
        f"Started/resumed `{experiment_type}` run `{run_id}` (dataset={resolved_ver}, "
        f"sha256={resolved_hash[:12]}..., seed={config.random_seed}): "
        f"total_planned={total_planned}, skipped_completed={skipped_existing}, pending={len(pending)}."
    )

    executed = 0
    succeeded = 0
    failed = 0
    halted_reason = "completed"

    for idx, trial in enumerate(pending, start=1):
        if should_stop and should_stop():
            halted_reason = "stopped"
            break
        if should_pause and should_pause():
            halted_reason = "paused"
            break

        rec = execute_single_planned_trial(
            experiment_type=experiment_type,
            trial=trial,
            config=config,
            experiment_run_id=run_id,
            dataset_hash=resolved_hash,
            config_fingerprint=config_fp,
            api_key=api_key,
            client=client,
        )
        executed += 1
        if rec["status"] in ("success", "blocked"):
            succeeded += 1
        else:
            failed += 1

        if on_trial_complete:
            on_trial_complete(rec, idx, len(pending))

    append_experiment_log(
        f"Finished `{experiment_type}` run `{run_id}` ({halted_reason}): executed={executed}, "
        f"succeeded={succeeded}, failed={failed}, skipped_existing={skipped_existing}."
    )

    return {
        "experiment_run_id": run_id,
        "experiment_type": experiment_type,
        "dataset_version": resolved_ver,
        "dataset_hash": resolved_hash,
        "random_seed": config.random_seed,
        "software_version": SOFTWARE_VERSION,
        "sdk_version": sdk_ver,
        "config_fingerprint": config_fp,
        "halted_reason": halted_reason,
        "total_planned": total_planned,
        "skipped_existing": skipped_existing,
        "pending_at_start": len(pending),
        "executed": executed,
        "succeeded": succeeded,
        "failed": failed,
    }
