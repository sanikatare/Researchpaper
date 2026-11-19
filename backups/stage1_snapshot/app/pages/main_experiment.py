"""PAGE 5 — Main Experiment Control with Explicit Researcher Approval Gate."""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from app.components import render_metric_card, render_page_header
from config import MAIN_TARGET_CALLS, MODALITIES, THINKING_LEVELS
from dataset.scenario_manager import get_main_scenarios, load_scenarios
from experiment.experiment_controller import get_controller
from experiment.experiment_runner import build_trial_plan, filter_pending_trials
from experiment.hydrogen_client import is_api_fully_validated
from experiment.storage import (
    get_config_snapshot_path,
    load_experiment_state,
    load_runtime_config,
    load_trial_records,
)


def render() -> None:
    render_page_header(
        "Main Experiment",
        "Controlled Full-Scale Execution (Target: 60 × 2 × 4 × 3 = 1,440 API Calls) — Append-Only & Resumable",
    )

    scenarios = get_main_scenarios(load_scenarios(), require_approved=True)
    cfg = load_runtime_config()
    ctrl = get_controller("main")
    snap = ctrl.snapshot()
    main_records = load_trial_records("main")
    main_state = load_experiment_state("main")

    if not is_api_fully_validated():
        st.warning("API configuration is not fully validated for all 4 thinking levels.")

    if len(scenarios) < 60:
        st.info(
            f"Currently {len(scenarios)} researcher-approved scenarios are available "
            "(full main target: 60 scenarios = 20 per category)."
        )

    st.subheader("Main Experiment Parameters")
    col1, col2 = st.columns(2)
    with col1:
        avail_ids = [s["scenario_id"] for s in scenarios]
        selected_ids = st.multiselect(
            "Approved Scenarios", options=avail_ids, default=avail_ids
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

    selected_scenarios = [s for s in scenarios if s["scenario_id"] in selected_ids]

    main_cfg = load_runtime_config()
    main_cfg.modalities = selected_modalities or list(MODALITIES)
    main_cfg.thinking_levels = selected_levels or list(THINKING_LEVELS)
    main_cfg.repetitions = int(reps)
    main_cfg.randomize_order = bool(randomize_order)
    main_cfg.max_output_tokens = int(max_out)
    main_cfg.timeout_seconds = float(timeout_sec)

    plan = build_trial_plan(
        scenarios=selected_scenarios,
        modalities=main_cfg.modalities,
        thinking_levels=main_cfg.thinking_levels,
        repetitions=main_cfg.repetitions,
        randomize_order=main_cfg.randomize_order,
        random_seed=main_cfg.random_seed,
    )
    pending, skipped = filter_pending_trials("main", plan)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Estimated Total Calls", str(len(plan)), f"Initial target: {MAIN_TARGET_CALLS}")
    with m2:
        render_metric_card("Completed (Resumable)", str(skipped), "Prevented from re-running")
    with m3:
        render_metric_card("Remaining Pending", str(len(pending)), "Queued for execution")
    with m4:
        err_cnt = sum(1 for r in main_records if r.get("status") == "error")
        render_metric_card("Failed Trials", str(err_cnt), "Logged in main JSONL")

    st.caption(
        "Estimated inference cost: Omitted unless verified official unit pricing for "
        "Gemini 4 Hydrogen is explicitly supplied; no speculative pricing is fabricated."
    )

    snap_file = get_config_snapshot_path("main")
    if snap_file.exists():
        st.info(
            f"Configuration lock active (`{snap_file.name}`). Execution will automatically halt "
            "if model identifier, SDK version, or token/instruction settings change mid-experiment."
        )

    st.divider()
    st.subheader("Explicit Researcher Confirmation Screen")
    with st.container(border=True):
        st.markdown(
            f"""
            **Pre-Execution Confirmation Checklist**
            - **Model:** `{main_cfg.model_id}`
            - **Scenarios:** `{len(selected_scenarios)}` researcher-approved families
            - **Conditions:** `{len(main_cfg.modalities)}` modalities × `{len(main_cfg.thinking_levels)}` thinking levels × `{main_cfg.repetitions}` repetitions
            - **Pending API Calls:** `{len(pending)}` (Total planned: `{len(plan)}`)
            - **Storage Mode:** Append-only `results/main/raw_responses.jsonl` (raw observations never overwritten)
            """
        )
        explicit_approval = st.checkbox(
            "I have reviewed the pilot readiness and dataset configuration and explicitly approve executing the Main Experiment.",
            value=False,
        )

    b1, b2, b3, b4 = st.columns(4)
    is_running = snap["is_alive"]
    with b1:
        if st.button(
            "Start / Resume Main Experiment",
            type="primary",
            disabled=is_running or not explicit_approval or len(pending) == 0 or not is_api_fully_validated(),
            use_container_width=True,
        ):
            ctrl.start(
                scenarios=selected_scenarios,
                config=main_cfg,
                explicit_approved=True,
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

    denom = max(1, len(plan))
    done_now = skipped + snap["succeeded_in_run"] if is_running else skipped
    st.progress(min(1.0, done_now / denom))
    st.write(
        f"**State:** `{snap['status']}` | Run Progress: `{snap['executed_in_run']}/{snap['pending_at_start']}` | "
        f"Succeeded: `{snap['succeeded_in_run']}` | Failed: `{snap['failed_in_run']}`"
    )
    if snap["last_error"]:
        st.error(f"Execution Halted / Error: {snap['last_error']}")

    if snap["logs"]:
        st.subheader("Live Execution & Error Monitor")
        st.text_area("Main Log", value="\n".join(snap["logs"][:60]), height=180, disabled=True)

    if is_running:
        time.sleep(2.0)
        st.rerun()

    st.subheader("Execution History")
    history = main_state.get("history", [])
    if history:
        st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)
    else:
        st.caption("No main experiment execution events recorded yet.")
