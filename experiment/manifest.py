"""Experiment run manifest, raw-to-processed traceability verification, and main execution readiness/reporting."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from config import (
    ANNOTATION_READINESS_MD,
    DEFAULT_MODEL_ID,
    EXPERIMENT_MANIFEST_JSON,
    MAIN_REPORT_MD,
    MAIN_TARGET_CALLS,
    MAIN_TARGET_REPETITIONS,
    MAIN_TARGET_SCENARIOS,
    MISSING_FAILED_REPORT_MD,
    MODALITIES,
    RESEARCH_DIR,
    SOFTWARE_VERSION,
    THINKING_LEVELS,
    ExperimentConfig,
    get_api_key,
)
from dataset.scenario_manager import load_active_dataset, verify_version_integrity
from evaluation.annotation import build_processed_results, load_annotations, resolve_final_annotations
from experiment.hydrogen_client import (
    get_sdk_version,
    is_api_fully_validated,
    load_validation_state,
)
from experiment.storage import (
    compute_config_fingerprint,
    condition_key,
    load_runtime_config,
    load_trial_records,
)


def build_experiment_manifest(
    experiment_type: Optional[str] = None,
    save_path: Path = EXPERIMENT_MANIFEST_JSON,
    config: Optional[ExperimentConfig] = None,
) -> Dict[str, Any]:
    """Generate experiment run manifest capturing dataset hash, model, config, seed, and execution counts."""
    cfg = config or load_runtime_config()
    sdk_ver = get_sdk_version()
    pilot_entry, _ = load_active_dataset("pilot")
    main_entry, _ = load_active_dataset("main")

    records = load_trial_records(experiment_type)
    df_ann = load_annotations()
    resolved = resolve_final_annotations(df_ann)

    pilot_recs = [r for r in records if r.get("experiment_type") == "pilot"]
    main_recs = [r for r in records if r.get("experiment_type") == "main"]

    def _counts(recs: List[Dict[str, Any]]) -> Dict[str, int]:
        succ = sum(1 for r in recs if r.get("status") == "success")
        blk = sum(1 for r in recs if r.get("status") == "blocked")
        err = sum(1 for r in recs if r.get("status") == "error")
        uniq_tids = len({str(r.get("trial_id")) for r in recs if r.get("trial_id")})
        uniq_cond = len(
            {
                condition_key(
                    str(r.get("scenario_id")),
                    str(r.get("modality")),
                    str(r.get("thinking_level")),
                    int(r.get("repetition", 1)),
                )
                for r in recs
                if r.get("status") in ("success", "blocked")
            }
        )
        return {
            "total_records": len(recs),
            "succeeded": succ,
            "blocked": blk,
            "failed": err,
            "unique_trial_ids": uniq_tids,
            "unique_completed_conditions": uniq_cond,
        }

    target_entry = pilot_entry if experiment_type == "pilot" else main_entry
    ds_ver = target_entry.get("dataset_version") if target_entry else None
    ds_hash = (
        (target_entry.get("approved_sha256") or target_entry.get("sha256"))
        if target_entry
        else None
    )

    manifest: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": cfg.model_id,
        "software_version": SOFTWARE_VERSION,
        "sdk_version": sdk_ver,
        "random_seed": cfg.random_seed,
        "randomize_order": cfg.randomize_order,
        "config_fingerprint": compute_config_fingerprint(cfg, sdk_ver),
        "configuration": cfg.to_safe_dict(),
        "datasets": {
            "pilot": {
                "dataset_version": pilot_entry.get("dataset_version") if pilot_entry else None,
                "sha256": pilot_entry.get("sha256") if pilot_entry else None,
                "approved_sha256": pilot_entry.get("approved_sha256") if pilot_entry else None,
                "researcher_approval_status": pilot_entry.get("researcher_approval_status") if pilot_entry else None,
                "scenario_count": pilot_entry.get("scenario_count") if pilot_entry else 0,
            },
            "main": {
                "dataset_version": main_entry.get("dataset_version") if main_entry else None,
                "sha256": main_entry.get("sha256") if main_entry else None,
                "approved_sha256": main_entry.get("approved_sha256") if main_entry else None,
                "researcher_approval_status": main_entry.get("researcher_approval_status") if main_entry else None,
                "scenario_count": main_entry.get("scenario_count") if main_entry else 0,
            },
        },
        "active_scope_dataset_version": ds_ver,
        "active_scope_dataset_hash": ds_hash,
        "execution_counts": {
            "overall": _counts(records),
            "pilot": _counts(pilot_recs),
            "main": _counts(main_recs),
            "annotation_entries": len(df_ann),
            "annotated_resolved_trials": int(resolved["resolved_label"].notna().sum()) if not resolved.empty else 0,
        },
    }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify_raw_traceability(
    df_processed: Optional[pd.DataFrame] = None,
    experiment_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify 100% traceability from processed observations back to immutable raw JSONL records."""
    raw_records = load_trial_records(experiment_type)
    raw_by_tid = {str(r.get("trial_id")): r for r in raw_records if r.get("trial_id")}

    # Check duplicate completed conditions in raw JSONL
    completed_keys: Dict[str, List[str]] = {}
    for r in raw_records:
        if r.get("status") in ("success", "blocked"):
            ck = condition_key(
                str(r.get("scenario_id")),
                str(r.get("modality")),
                str(r.get("thinking_level")),
                int(r.get("repetition", 1)),
            )
            scoped_ck = f"{r.get('experiment_type')}|{ck}"
            completed_keys.setdefault(scoped_ck, []).append(str(r.get("trial_id")))
    duplicate_conditions = {k: v for k, v in completed_keys.items() if len(v) > 1}

    if df_processed is None:
        df_processed = build_processed_results(experiment_type, save_csv=False)

    unmatched_processed: List[str] = []
    mismatched_fields: List[str] = []
    if not df_processed.empty:
        for _, row in df_processed.iterrows():
            tid = str(row.get("trial_id", ""))
            raw = raw_by_tid.get(tid)
            if raw is None:
                unmatched_processed.append(tid)
                continue
            for col in ("scenario_id", "modality", "thinking_level", "status"):
                if str(row.get(col)) != str(raw.get(col)):
                    mismatched_fields.append(f"{tid}:{col}")

    return {
        "raw_count": len(raw_records),
        "processed_count": len(df_processed),
        "all_processed_traced_to_raw": len(unmatched_processed) == 0 and len(mismatched_fields) == 0,
        "unmatched_trial_ids": unmatched_processed,
        "mismatched_fields": mismatched_fields,
        "duplicate_completed_conditions": duplicate_conditions,
        "zero_duplicate_completed_conditions": len(duplicate_conditions) == 0,
    }


def verify_main_experiment_readiness(
    config: Optional[ExperimentConfig] = None,
    explicit_approved: bool = False,
) -> Dict[str, Any]:
    """Audit all 9 pre-execution prerequisites for the 1,440-call main experiment."""
    cfg = config or load_runtime_config()
    main_entry, main_scenarios = load_active_dataset("main")
    val_state = load_validation_state()
    levels_state = val_state.get("levels", {})

    ds_approved = bool(
        main_entry and main_entry.get("researcher_approval_status") == "approved"
    )
    ds_hash_ok = False
    if main_entry:
        integrity = verify_version_integrity(main_entry["dataset_version"])
        ds_hash_ok = bool(integrity.get("intact"))

    validated_levels = [
        lvl
        for lvl in THINKING_LEVELS
        if levels_state.get(lvl, {}).get("validated")
        and levels_state.get(lvl, {}).get("supported_by_endpoint") is True
    ]

    checks = [
        {
            "check": "Approved Main Dataset Version & SHA-256",
            "passed": ds_approved and ds_hash_ok and len(main_scenarios) == MAIN_TARGET_SCENARIOS,
            "detail": (
                f"Version={main_entry.get('dataset_version') if main_entry else None}, "
                f"Approval={main_entry.get('researcher_approval_status') if main_entry else None}, "
                f"Scenarios={len(main_scenarios)}/{MAIN_TARGET_SCENARIOS}, HashIntact={ds_hash_ok}"
            ),
        },
        {
            "check": "Exclusive Model Identifier",
            "passed": cfg.model_id == DEFAULT_MODEL_ID,
            "detail": f"Configured model_id={cfg.model_id}",
        },
        {
            "check": "All 4 Thinking Levels Endpoint-Validated",
            "passed": len(validated_levels) == len(THINKING_LEVELS) and is_api_fully_validated(),
            "detail": f"Validated: {validated_levels} (requires {list(THINKING_LEVELS)})",
        },
        {
            "check": "Both Input Modalities Configured",
            "passed": set(cfg.modalities) == set(MODALITIES),
            "detail": f"Modalities={cfg.modalities}",
        },
        {
            "check": "Confirmatory Repetitions & 1,440 Target",
            "passed": len(main_scenarios) * len(MODALITIES) * len(THINKING_LEVELS) * MAIN_TARGET_REPETITIONS == MAIN_TARGET_CALLS,
            "detail": f"Target design={len(main_scenarios)} × 2 × 4 × {MAIN_TARGET_REPETITIONS} = {MAIN_TARGET_CALLS} calls",
        },
        {
            "check": "Valid Runtime Configuration & Random Seed",
            "passed": len(cfg.validate()) == 0 and isinstance(cfg.random_seed, int),
            "detail": f"Errors={cfg.validate() or 'None'}, random_seed={cfg.random_seed}",
        },
        {
            "check": "API Key Present & Connectivity Prerequisite",
            "passed": bool(get_api_key()),
            "detail": "GEMINI_API_KEY configured" if get_api_key() else "GEMINI_API_KEY missing from runtime environment",
        },
        {
            "check": "Explicit Researcher Execution Approval",
            "passed": bool(explicit_approved),
            "detail": "Approved" if explicit_approved else "Awaiting explicit researcher approval",
        },
    ]
    return {
        "ready": all(c["passed"] for c in checks),
        "checks": checks,
    }


def generate_main_post_run_reports(
    save_dir: Path = RESEARCH_DIR,
    manifest_path: Path = EXPERIMENT_MANIFEST_JSON,
    explicit_approved: bool = False,
) -> Dict[str, Path]:
    """Generate the 5 post-execution main experiment reports/manifests from actual main state (no fabrication)."""
    save_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    manifest = build_experiment_manifest("main", save_path=manifest_path)
    readiness = verify_main_experiment_readiness(explicit_approved=explicit_approved)
    main_records = load_trial_records("main")
    df_main = pd.DataFrame(main_records) if main_records else pd.DataFrame()
    counts = manifest["execution_counts"]["main"]

    # 1. Main Experiment Completion Report
    main_status = "COMPLETED" if counts["unique_completed_conditions"] >= MAIN_TARGET_CALLS else (
        "IN PROGRESS / PARTIAL" if counts["total_records"] > 0 else "BLOCKED / NOT EXECUTED"
    )
    comp_lines = [
        "# Main Experiment Completion Report",
        "",
        f"**Generated:** {ts}",
        f"**Execution Status:** `{main_status}`",
        f"**Target Calls:** {MAIN_TARGET_CALLS} (60 scenarios × 2 modalities × 4 thinking levels × 3 repetitions)",
        f"**Recorded Trials:** total={counts['total_records']}, succeeded={counts['succeeded']}, blocked={counts['blocked']}, failed={counts['failed']}, unique_completed={counts['unique_completed_conditions']}",
        "",
        "## Pre-Execution Gate Verification",
    ]
    for chk in readiness["checks"]:
        mark = "PASS" if chk["passed"] else "BLOCKED"
        comp_lines.append(f"- **[{mark}] {chk['check']}:** {chk['detail']}")
    comp_path = save_dir / MAIN_REPORT_MD.name
    comp_path.write_text("\n".join(comp_lines) + "\n", encoding="utf-8")

    # 2. Missing and Failed Trial Report
    mf_lines = [
        "# Missing and Failed Trial Report (Main Experiment)",
        "",
        f"**Generated:** {ts}",
        f"**Total Main Records:** {counts['total_records']}",
        f"**Missing Target Conditions:** {max(0, MAIN_TARGET_CALLS - counts['unique_completed_conditions'])} / {MAIN_TARGET_CALLS}",
        f"**Failed API Records:** {counts['failed']}",
        "",
    ]
    if not df_main.empty and "status" in df_main.columns:
        errs = df_main[df_main["status"] == "error"]
        if not errs.empty:
            mf_lines.append("## Recorded API Errors")
            for _, r in errs.head(50).iterrows():
                mf_lines.append(
                    f"- `{r.get('trial_id')}` ({r.get('scenario_id')}, {r.get('modality')}, {r.get('thinking_level')}): {r.get('error_code')} — {r.get('error_message')}"
                )
        else:
            mf_lines.append("No failed API trials recorded.")
    else:
        mf_lines.append("No main experiment trials have been executed yet.")
    mf_path = save_dir / MISSING_FAILED_REPORT_MD.name
    mf_path.write_text("\n".join(mf_lines) + "\n", encoding="utf-8")

    # 3. Annotation Readiness Report
    evaluable_api = counts["succeeded"] + counts["blocked"]
    df_ann = load_annotations()
    main_ann = df_ann[df_ann["experiment_type"] == "main"] if not df_ann.empty else df_ann
    ar_lines = [
        "# Annotation Readiness Report (Main Experiment)",
        "",
        f"**Generated:** {ts}",
        f"**Evaluable Completed API Trials Available for Coding:** {evaluable_api} / {MAIN_TARGET_CALLS}",
        f"**Existing Main Annotator Entries:** {len(main_ann)}",
        f"**Readiness Status:** {'READY FOR DUAL HUMAN CODING' if evaluable_api > 0 else 'WAITING FOR MAIN EXPERIMENT EXECUTION'}",
        "",
        "- Dual-coder CSV storage: `results/human_annotations.csv` (strictly isolated from raw JSONL).",
    ]
    ar_path = save_dir / ANNOTATION_READINESS_MD.name
    ar_path.write_text("\n".join(ar_lines) + "\n", encoding="utf-8")

    return {
        "main_completion_report": comp_path,
        "missing_failed_report": mf_path,
        "annotation_readiness_report": ar_path,
        "manifest_json": manifest_path,
    }
