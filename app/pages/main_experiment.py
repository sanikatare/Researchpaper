"""PAGE 5 — Main Experiment Control (60-Scenario Confirmatory Study)."""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from app.components import render_metric_card, render_page_header
from config import MAIN_TARGET_CALLS, MODALITIES, THINKING_LEVELS
from dataset.scenario_manager import load_active_dataset
from experiment.experiment_controller import get_controller
from experiment.experiment_runner import build_trial_plan, filter_pending_trials
from experiment.hydrogen_client import is_api_fully_validated
from experiment.manifest import (
    generate_main_post_run_reports,
    verify_main_experiment_readiness,
)
from experiment.storage import (
    get_config_snapshot_path,
    load_experiment_state,
    load_runtime_config,
    load_trial_records,
)


def render() -> None:
    render_page_header(
        "Main Experiment",
        "Confirmatory Full-Scale Execution (60 × 2 × 4 × 3 = 1,440 API Calls) — Append-Only & Resumable",
    )

    main_entry, main_scenarios = load_active_dataset("main")
    if main_entry is None:
        st.error("No active confirmatory main dataset version selected. Configure one in **3. Dataset Status**.")
        return

    is_approved = main_entry.get("researcher_approval_status") == "approved"
    if not is_approved:
        st.warning(
            f"Active main dataset `{main_entry['dataset_version']}` is `{main_entry.get('researcher_approval_status')}`. "
            "Approve it in **3. Dataset Status** before starting the main experiment."
        )

    if not is_api_fully_validated():
        st.warning("All 4 thinking levels must pass validation in **2. API Configuration** first.")

    cfg = load_runtime_config()
    ctrl = get_controller("main")
    snap = ctrl.snapshot()
    main_records = load_trial_records("main")
    main_state = load_experiment_state("main")

    st.subheader("Experimental Configuration")
    col1, col2 = st.columns(2)
    with col1:
        st.text_input(
            "Selected Dataset Version",
            value=f"{main_entry['dataset_name']} ({main_entry['dataset_version']})",
            disabled=True,
        )
        selected_modalities = st.multiselect(
            "Modalities", options=list(MODALITIES), default=list(MODALITIES)
        )
        selected_levels = st.multiselect(
            "Thinking Levels", options=list(THINKING_LEVELS), default=list(THINKING_LEVELS)
        )
    with col2:
        reps = st.number_input("Repetitions per Condition", min_value=1, max_value=5, value=3, step=1)
        randomize_order = st.checkbox("Randomized Execution Order", value=True)
        max_out = st.number_input(
            "Output Token Limit", min_value=64, max_value=65536, value=int(cfg.max_output_tokens), step=256
        )
        timeout_sec = st.number_input(
            "Request Timeout (seconds)", min_value=5.0, max_value=600.0, value=float(cfg.timeout_seconds), step=5.0
        )

    main_cfg = load_runtime_config()
    main_cfg.modalities = selected_modalities or list(MODALITIES)
    main_cfg.thinking_levels = selected_levels or list(THINKING_LEVELS)
    main_cfg.repetitions = int(reps)
    main_cfg.randomize_order = bool(randomize_order)
    main_cfg.max_output_tokens = int(max_out)
    main_cfg.timeout_seconds = float(timeout_sec)

    plan = build_trial_plan(
        scenarios=main_scenarios,
        modalities=main_cfg.modalities,
        thinking_levels=main_cfg.thinking_levels,
        repetitions=main_cfg.repetitions,
        randomize_order=main_cfg.randomize_order,
        random_seed=main_cfg.random_seed,
    )
    pending, skipped = filter_pending_trials("main", plan)
    err_cnt = sum(1 for r in main_records if r.get("status") == "error")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Scenarios", str(len(main_scenarios)), f"Expected calls: {len(plan)} (target {MAIN_TARGET_CALLS})", accent_color="#4285F4")
    with m2:
        render_metric_card("Completed (Resumable)", str(skipped), "Skipped if resuming", accent_color="#34A853")
    with m3:
        render_metric_card("Pending Calls", str(len(pending)), "Queued for execution", accent_color="#FBBC04")
    with m4:
        render_metric_card("Failed Trials", str(err_cnt), "Logged in main JSONL", accent_color="#EA4335")

    st.caption(
        "Estimated inference cost: Omitted because verified unit pricing for Gemini 4 Hydrogen "
        "is not bundled; actual token counts are captured per call from API `usage_metadata`."
    )

    snap_file = get_config_snapshot_path("main")
    if snap_file.exists():
        st.info(
            f"Configuration lock active (`{snap_file.name}`). Execution halts automatically if "
            "model identifier, SDK version, or core generation config drifts mid-experiment."
        )

    st.divider()
    st.subheader("Explicit Researcher Confirmation")
    with st.container(border=True):
        st.markdown(
            f"""
            - **Model:** `{main_cfg.model_id}`
            - **Dataset Version:** `{main_entry['dataset_version']}` (`{len(main_scenarios)}` scenarios)
            - **Design:** `{len(main_cfg.modalities)}` modalities × `{len(main_cfg.thinking_levels)}` thinking levels × `{main_cfg.repetitions}` repetitions
            - **Pending Calls:** `{len(pending)}` of `{len(plan)}`
            - **Storage:** Append-only `results/main/raw_responses.jsonl` (tagged with `dataset_version`)
            """
        )
        explicit_approval = st.checkbox(
            "I confirm pilot review is complete and explicitly authorize executing the Main Experiment.",
            value=False,
        )

    b1, b2, b3, b4 = st.columns(4)
    is_running = snap["is_alive"]
    can_start = (
        not is_running
        and explicit_approval
        and is_approved
        and is_api_fully_validated()
        and len(pending) > 0
    )
    with b1:
        if st.button("Start / Resume Main Experiment", type="primary", disabled=not can_start, use_container_width=True):
            ctrl.start(
                scenarios=main_scenarios,
                config=main_cfg,
                explicit_approved=True,
                dataset_version=main_entry.get("dataset_version"),
                dataset_hash=main_entry.get("approved_sha256") or main_entry.get("sha256"),
                require_approved_dataset=True,
                require_validated_levels=True,
            )
            st.rerun()
    with b2:
        if st.button("Pause", disabled=not is_running, use_container_width=True):
            ctrl.pause()
    with b3:
        if st.button("Stop", disabled=not is_running, use_container_width=True):
            ctrl.stop()
    with b4:
        if st.button("Refresh", use_container_width=True):
            st.rerun()

    done_now = skipped + snap["succeeded_in_run"] if is_running else skipped
    st.progress(min(1.0, done_now / max(1, len(plan))))
    st.write(
        f"**State:** `{snap['status']}` | Run Progress: `{snap['executed_in_run']}/{snap['pending_at_start']}` | "
        f"Succeeded: `{snap['succeeded_in_run']}` | Failed: `{snap['failed_in_run']}`"
    )
    if snap["last_error"]:
        st.error(f"Execution Halted / Error: {snap['last_error']}")

    if snap["logs"]:
        st.subheader("Live Execution Log")
        st.text_area("Main Log", value="\n".join(snap["logs"][:60]), height=180, disabled=True)

    if is_running:
        time.sleep(2.0)
        st.rerun()

    st.subheader("Pre-Execution Readiness Verification")
    readiness = verify_main_experiment_readiness(main_cfg, explicit_approved=explicit_approval)
    st.dataframe(pd.DataFrame(readiness["checks"]), use_container_width=True, hide_index=True)

    st.subheader("Execution History")
    history = main_state.get("history", [])
    if history:
        st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)
    else:
        st.caption("No main experiment runs logged yet.")

    if st.button("Generate Main Manifest & Post-Execution Status Reports"):
        paths = generate_main_post_run_reports(explicit_approved=explicit_approval)
        st.success(
            "Generated `results/experiment_manifest.json`, `research/main_experiment_report.md`, "
            "`research/missing_failed_trials_report.md`, and `research/annotation_readiness_report.md`."
        )
        st.caption(f"Files written: {', '.join(p.name for p in paths.values())}")
