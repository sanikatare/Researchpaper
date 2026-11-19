"""PAGE 4 — Pilot Experiment Control & Report Generation."""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from app.components import render_metric_card, render_page_header
from config import CATEGORY_LABELS, MODALITIES, PILOT_TARGET_CALLS, THINKING_LEVELS
from dataset.scenario_manager import get_pilot_scenarios, load_scenarios
from evaluation.annotation import build_processed_results
from evaluation.metrics import generate_pilot_report
from experiment.experiment_controller import get_controller
from experiment.experiment_runner import build_trial_plan, filter_pending_trials
from experiment.hydrogen_client import is_api_fully_validated
from experiment.storage import load_runtime_config, load_trial_records


def render() -> None:
    render_page_header(
        "Pilot Experiment",
        "Dedicated 12-Scenario Pilot Execution (Target: 12 × 2 × 4 × 1 = 96 Calls) & Readiness Evaluation",
    )

    all_scenarios = load_scenarios()
    pilot_approved = get_pilot_scenarios(all_scenarios, require_approved=True)
    pilot_all = get_pilot_scenarios(all_scenarios, require_approved=False)

    if not is_api_fully_validated():
        st.warning(
            "All 4 thinking levels have not been verified in **2. API Configuration** yet. "
            "Validate the API before running live pilot calls."
        )

    if len(pilot_all) != 12:
        st.info(f"Currently {len(pilot_all)} scenarios are flagged `in_pilot=True` (target: 12).")
    if len(pilot_approved) < len(pilot_all):
        st.warning(
            f"{len(pilot_approved)}/{len(pilot_all)} pilot scenarios are researcher-approved. "
            "Review and approve scenarios in **3. Dataset Management** before running."
        )

    cfg = load_runtime_config()
    ctrl = get_controller("pilot")
    snap = ctrl.snapshot()
    pilot_records = load_trial_records("pilot")

    # Configuration controls
    st.subheader("Pilot Configuration")
    col_l, col_r = st.columns(2)
    with col_l:
        avail_ids = [s["scenario_id"] for s in pilot_approved]
        selected_ids = st.multiselect(
            "Selected Pilot Scenarios (Researcher-Approved)",
            options=avail_ids,
            default=avail_ids,
        )
        selected_modalities = st.multiselect(
            "Modalities",
            options=list(MODALITIES),
            default=list(MODALITIES),
        )
    with col_r:
        selected_levels = st.multiselect(
            "Thinking Levels",
            options=list(THINKING_LEVELS),
            default=list(THINKING_LEVELS),
        )
        pilot_reps = st.number_input(
            "Repetitions per Condition", min_value=1, max_value=3, value=1, step=1
        )
        randomize_pilot = st.checkbox("Randomize Execution Order", value=True)

    selected_scenarios = [s for s in pilot_approved if s["scenario_id"] in selected_ids]

    # Category distribution
    cat_dist = {
        CATEGORY_LABELS[c]: sum(1 for s in selected_scenarios if s.get("category") == c)
        for c in CATEGORY_LABELS
    }
    st.caption(f"Category distribution: {cat_dist}")

    pilot_cfg = load_runtime_config()
    pilot_cfg.modalities = selected_modalities or list(MODALITIES)
    pilot_cfg.thinking_levels = selected_levels or list(THINKING_LEVELS)
    pilot_cfg.repetitions = int(pilot_reps)
    pilot_cfg.randomize_order = bool(randomize_pilot)

    plan = build_trial_plan(
        scenarios=selected_scenarios,
        modalities=pilot_cfg.modalities,
        thinking_levels=pilot_cfg.thinking_levels,
        repetitions=pilot_cfg.repetitions,
        randomize_order=pilot_cfg.randomize_order,
        random_seed=pilot_cfg.random_seed,
    )
    pending, skipped = filter_pending_trials("pilot", plan)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Target Pilot Calls", str(len(plan)), f"Initial design: {PILOT_TARGET_CALLS}")
    with m2:
        render_metric_card("Already Completed", str(skipped), "Skipped on resume")
    with m3:
        render_metric_card("Pending Calls", str(len(pending)), "Remaining to run")
    with m4:
        err_cnt = sum(1 for r in pilot_records if r.get("status") == "error")
        render_metric_card("Recorded Errors", str(err_cnt), "Preserved in pilot JSONL")

    with st.expander("Preview Selected Pilot Prompts"):
        if selected_scenarios:
            prev_id = st.selectbox("Preview Scenario", options=[s["scenario_id"] for s in selected_scenarios])
            prev_s = next(s for s in selected_scenarios if s["scenario_id"] == prev_id)
            st.markdown(f"**Text-Only Prompt:** {prev_s['text_prompt']}")
            st.markdown(f"**Image + Text Prompt:** {prev_s['image_text_prompt']}")
            st.markdown(f"**Expected Safe Behavior:** {prev_s['expected_safe_behavior']}")

    # Execution Controls
    b1, b2, b3, b4 = st.columns(4)
    is_running = snap["is_alive"]
    with b1:
        if st.button(
            "Start / Resume Pilot",
            type="primary",
            disabled=is_running or len(pending) == 0 or not is_api_fully_validated(),
            use_container_width=True,
        ):
            ctrl.start(scenarios=selected_scenarios, config=pilot_cfg, explicit_approved=True)
            st.rerun()
    with b2:
        if st.button("Pause Pilot", disabled=not is_running, use_container_width=True):
            ctrl.pause()
    with b3:
        if st.button("Stop Pilot", disabled=not is_running, use_container_width=True):
            ctrl.stop()
    with b4:
        if st.button("Refresh Status", use_container_width=True):
            st.rerun()

    # Live progress
    total_done_now = skipped + snap["succeeded_in_run"] if is_running else skipped
    denom = max(1, len(plan))
    st.progress(min(1.0, total_done_now / denom))
    st.write(
        f"**Runner Status:** `{snap['status']}` | Executed in active run: "
        f"`{snap['executed_in_run']}/{snap['pending_at_start']}` | "
        f"Succeeded: `{snap['succeeded_in_run']}` | Failed: `{snap['failed_in_run']}`"
    )
    if snap["last_error"]:
        st.error(f"Runner Error: {snap['last_error']}")

    if snap["logs"]:
        st.subheader("Live Execution Log")
        st.text_area("Pilot Log", value="\n".join(snap["logs"][:50]), height=180, disabled=True)

    if is_running:
        time.sleep(2.0)
        st.rerun()

    st.divider()
    st.subheader("Recorded Pilot Trials & Pilot Report")
    if pilot_records:
        df_pilot = pd.DataFrame(pilot_records)
        cols = [
            c
            for c in (
                "trial_id",
                "scenario_id",
                "category",
                "modality",
                "thinking_level",
                "repetition",
                "status",
                "latency_ms",
                "input_tokens",
                "output_tokens",
                "thinking_tokens",
                "model_id",
                "actual_model_version",
                "error_message",
            )
            if c in df_pilot.columns
        ]
        st.dataframe(df_pilot[cols], use_container_width=True, hide_index=True)

    if st.button("Generate Pilot Evaluation Report"):
        df_proc = build_processed_results("pilot", save_csv=False)
        report_md = generate_pilot_report(df_proc, all_scenarios)
        st.success("Saved pilot report to `research/pilot_report.md`.")
        st.markdown(report_md)
