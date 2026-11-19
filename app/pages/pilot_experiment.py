"""PAGE 4 — Pilot Experiment Control (Isolated 12-Scenario Pilot Version)."""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

import json

from app.components import render_metric_card, render_page_header
from config import (
    CATEGORY_LABELS,
    MODALITIES,
    PILOT_MANIFEST_JSON,
    PILOT_TARGET_CALLS,
    PILOT_TARGET_REPETITIONS,
    THINKING_LEVELS,
)
from dataset.scenario_manager import load_active_dataset
from evaluation.annotation import build_processed_results
from evaluation.metrics import generate_pilot_report
from experiment.experiment_controller import get_controller
from experiment.experiment_runner import build_trial_plan, filter_pending_trials
from experiment.hydrogen_client import is_api_fully_validated
from experiment.manifest import build_experiment_manifest
from experiment.storage import load_runtime_config, load_trial_records


def render() -> None:
    render_page_header(
        "Pilot Experiment",
        "Isolated 12-Scenario Pilot Execution (Target: 12 × 2 × 4 × 1 = 96 API Calls)",
    )

    pilot_entry, pilot_scenarios = load_active_dataset("pilot")
    if pilot_entry is None:
        st.error("No active pilot dataset version selected. Configure one in **3. Dataset Status**.")
        return

    is_approved = pilot_entry.get("researcher_approval_status") == "approved"
    if not is_approved:
        st.warning(
            f"Active pilot dataset `{pilot_entry['dataset_version']}` is currently "
            f"`{pilot_entry.get('researcher_approval_status')}`. Grant Researcher Approval in "
            "**3. Dataset Status** before executing pilot trials."
        )

    if not is_api_fully_validated():
        st.warning(
            "Verify all four thinking levels in **2. API Configuration** before running pilot calls."
        )

    ctrl = get_controller("pilot")
    snap = ctrl.snapshot()
    pilot_records = load_trial_records("pilot")

    st.subheader("Pilot Configuration")
    c_cfg1, c_cfg2 = st.columns(2)
    with c_cfg1:
        st.text_input(
            "Active Pilot Dataset Version",
            value=f"{pilot_entry['dataset_name']} ({pilot_entry['dataset_version']})",
            disabled=True,
        )
        selected_modalities = st.multiselect(
            "Modalities", options=list(MODALITIES), default=list(MODALITIES)
        )
    with c_cfg2:
        selected_levels = st.multiselect(
            "Thinking Levels", options=list(THINKING_LEVELS), default=list(THINKING_LEVELS)
        )
        pilot_reps = st.number_input(
            "Repetitions (Fixed Pilot Protocol: 1)",
            min_value=PILOT_TARGET_REPETITIONS,
            max_value=PILOT_TARGET_REPETITIONS,
            value=PILOT_TARGET_REPETITIONS,
            step=1,
            disabled=True,
        )
        randomize_pilot = st.checkbox("Randomize Trial Order", value=True)

    pilot_cfg = load_runtime_config()
    pilot_cfg.modalities = selected_modalities or list(MODALITIES)
    pilot_cfg.thinking_levels = selected_levels or list(THINKING_LEVELS)
    pilot_cfg.repetitions = PILOT_TARGET_REPETITIONS
    pilot_cfg.randomize_order = bool(randomize_pilot)

    plan = build_trial_plan(
        scenarios=pilot_scenarios,
        modalities=pilot_cfg.modalities,
        thinking_levels=pilot_cfg.thinking_levels,
        repetitions=pilot_cfg.repetitions,
        randomize_order=pilot_cfg.randomize_order,
        random_seed=pilot_cfg.random_seed,
    )
    pending, skipped = filter_pending_trials("pilot", plan)
    err_cnt = sum(1 for r in pilot_records if r.get("status") == "error")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Pilot Scenarios", str(len(pilot_scenarios)), f"Expected calls: {len(plan)} (target {PILOT_TARGET_CALLS})", accent_color="#4285F4")
    with m2:
        render_metric_card("Completed Trials", str(skipped), "Preserved in pilot JSONL", accent_color="#34A853")
    with m3:
        render_metric_card("Pending Trials", str(len(pending)), "Remaining to execute", accent_color="#FBBC04")
    with m4:
        render_metric_card("Failed Trials", str(err_cnt), "Logged without silent discard", accent_color="#EA4335")

    cat_dist = {
        CATEGORY_LABELS[c]: sum(1 for s in pilot_scenarios if s.get("category") == c)
        for c in CATEGORY_LABELS
    }
    st.caption(f"Pilot category balance: {cat_dist}")

    b1, b2, b3, b4 = st.columns(4)
    is_running = snap["is_alive"]
    can_start = (
        not is_running
        and is_approved
        and is_api_fully_validated()
        and len(pending) > 0
    )
    with b1:
        if st.button("Start / Resume Pilot", type="primary", disabled=not can_start, use_container_width=True):
            ctrl.start(
                scenarios=pilot_scenarios,
                config=pilot_cfg,
                explicit_approved=True,
                dataset_version=pilot_entry.get("dataset_version"),
                dataset_hash=pilot_entry.get("approved_sha256") or pilot_entry.get("sha256"),
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

    total_done = skipped + snap["succeeded_in_run"] if is_running else skipped
    st.progress(min(1.0, total_done / max(1, len(plan))))
    st.write(
        f"**Execution Status:** `{snap['status']}` | Active Run: `{snap['executed_in_run']}/{snap['pending_at_start']}` | "
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
    if pilot_records:
        st.subheader("Recorded Pilot Trials")
        df_pilot = pd.DataFrame(pilot_records)
        cols = [
            c
            for c in (
                "trial_id",
                "dataset_version",
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
                "actual_model_version",
                "error_message",
            )
            if c in df_pilot.columns
        ]
        st.dataframe(df_pilot[cols], use_container_width=True, hide_index=True)

    pr1, pr2 = st.columns(2)
    with pr1:
        if st.button("Generate Pilot Evaluation Report", use_container_width=True):
            df_proc = build_processed_results("pilot", save_csv=False)
            report_md = generate_pilot_report(df_proc, pilot_scenarios)
            build_experiment_manifest("pilot", save_path=PILOT_MANIFEST_JSON, config=pilot_cfg)
            st.success("Generated `research/pilot_report.md` and `results/pilot/pilot_experiment_manifest.json`.")
            st.markdown(report_md)
    with pr2:
        if st.button("Generate Pilot Manifest (JSON)", use_container_width=True):
            p_man = build_experiment_manifest("pilot", save_path=PILOT_MANIFEST_JSON, config=pilot_cfg)
            st.success("Saved `results/pilot/pilot_experiment_manifest.json`.")
            st.download_button(
                "Download Pilot Manifest (JSON)",
                data=json.dumps(p_man, indent=2),
                file_name="pilot_experiment_manifest.json",
                mime="application/json",
                use_container_width=True,
            )
