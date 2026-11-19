"""Trial plan builder and resumable experiment runner."""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from config import ExperimentConfig
from experiment.hydrogen_client import TrialCallResult, get_sdk_version
from experiment.multimodal_runner import run_multimodal_trial
from experiment.storage import (
    append_experiment_log,
    append_trial_record,
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


def build_trial_plan(
    scenarios: List[Dict[str, Any]],
    modalities: List[str],
    thinking_levels: List[str],
    repetitions: int,
    randomize_order: bool = True,
    random_seed: int = 42,
) -> List[PlannedTrial]:
    """Generate full factorial trial schedule across scenarios, modalities, levels, reps."""
    plan: List[PlannedTrial] = []
    for rep in range(1, repetitions + 1):
        for scenario in scenarios:
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
    api_key: Optional[str] = None,
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run one planned trial, append raw record to JSONL, and return the record."""
    scenario = trial.scenario
    sid = str(scenario["scenario_id"])
    category = str(scenario["category"])
    trial_id = (
        f"{experiment_type}-{sid}-{trial.modality}-{trial.thinking_level}"
        f"-r{trial.repetition}-{uuid.uuid4().hex[:8]}"
    )

    try:
        prompt, img_str, call_res = run_multimodal_trial(
            scenario=scenario,
            modality=trial.modality,
            thinking_level=trial.thinking_level,
            config=config,
            api_key=api_key,
            client=client,
        )
    except Exception as exc:
        # Never silently discard failed trials
        prompt = str(
            scenario.get(
                "text_prompt" if trial.modality == "text_only" else "image_text_prompt",
                "",
            )
        )
        img_str = scenario.get("image_path")
        from datetime import datetime, timezone

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

    record: Dict[str, Any] = {
        "trial_id": trial_id,
        "experiment_type": experiment_type,
        "scenario_id": sid,
        "category": category,
        "expected_classification": scenario.get("expected_classification"),
        "modality": trial.modality,
        "thinking_level": trial.thinking_level,
        "repetition": trial.repetition,
        "prompt": prompt,
        "image_path": img_str if trial.modality == "image_plus_text" else None,
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
        "sdk_version": call_res.sdk_version,
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
) -> Dict[str, Any]:
    """Execute pending trials synchronously with config locking, resumability, and pause/stop hooks."""
    errors = config.validate()
    if errors:
        raise ValueError("Invalid configuration: " + "; ".join(errors))

    sdk_ver = get_sdk_version()
    verify_or_lock_config(experiment_type, config, sdk_ver)

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
        f"Started/resumed `{experiment_type}` batch: total_planned={total_planned}, "
        f"skipped_completed={skipped_existing}, pending={len(pending)}."
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
        f"Finished `{experiment_type}` batch ({halted_reason}): executed={executed}, "
        f"succeeded={succeeded}, failed={failed}, skipped_existing={skipped_existing}."
    )

    return {
        "experiment_type": experiment_type,
        "halted_reason": halted_reason,
        "total_planned": total_planned,
        "skipped_existing": skipped_existing,
        "pending_at_start": len(pending),
        "executed": executed,
        "succeeded": succeeded,
        "failed": failed,
    }
