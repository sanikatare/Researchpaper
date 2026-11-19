"""PAGE 1 — Research Overview."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components import render_metric_card, render_page_header
from config import (
    CATEGORY_LABELS,
    DEFAULT_MODEL_ID,
    MAIN_TARGET_CALLS,
    MODALITIES,
    MODALITY_LABELS,
    PILOT_TARGET_CALLS,
    RESEARCH_OBJECTIVE,
    RESEARCH_TITLE,
    THINKING_LEVELS,
)
from dataset.scenario_manager import load_scenarios, validate_dataset
from evaluation.annotation import load_annotations, resolve_final_annotations
from experiment.hydrogen_client import is_api_fully_validated, load_validation_state
from experiment.storage import load_experiment_state, load_trial_records


def render() -> None:
    render_page_header(
        "Research Overview",
        "Experiment Control Panel and Reproducible Study Status",
    )

    st.markdown(f"### {RESEARCH_TITLE}")
    st.markdown(f"**Research Objective:** {RESEARCH_OBJECTIVE}")

    api_validated = is_api_fully_validated()
    if not api_validated:
        st.warning(
            "API configuration has not been validated across all four thinking levels "
            "(LOW, MEDIUM, HIGH, XHIGH). Visit **2. API Configuration** to run real "
            "connection verification before launching experiments."
        )

    scenarios = load_scenarios()
    val_summary = validate_dataset(scenarios)
    pilot_records = load_trial_records("pilot")
    main_records = load_trial_records("main")
    all_records = pilot_records + main_records

    completed_calls = sum(
        1 for r in all_records if r.get("status") in ("success", "blocked")
    )
    failed_calls = sum(1 for r in all_records if r.get("status") == "error")

    df_ann = load_annotations()
    df_resolved = resolve_final_annotations(df_ann)
    annotated_count = (
        int(df_resolved["resolved_label"].notna().sum())
        if not df_resolved.empty
        else 0
    )

    pilot_state = load_experiment_state("pilot")
    main_state = load_experiment_state("main")

    pilot_completed = sum(
        1 for r in pilot_records if r.get("status") in ("success", "blocked")
    )
    main_completed = sum(
        1 for r in main_records if r.get("status") in ("success", "blocked")
    )

    if main_state.get("status") == "RUNNING":
        overall_status = "Main Experiment Running"
    elif pilot_state.get("status") == "RUNNING":
        overall_status = "Pilot Experiment Running"
    elif main_completed >= MAIN_TARGET_CALLS:
        overall_status = "Main Experiment Data Collected"
    elif pilot_completed >= PILOT_TARGET_CALLS:
        overall_status = "Pilot Completed (Awaiting Main Approval)"
    elif api_validated:
        overall_status = "API Validated / Ready for Pilot"
    else:
        overall_status = "Setup & Validation Phase"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Experimental Model", DEFAULT_MODEL_ID, "Single-model design")
    with c2:
        render_metric_card("Current Status", overall_status)
    with c3:
        render_metric_card(
            "Scenario Families",
            str(val_summary["total_scenarios"]),
            f"{val_summary['researcher_approved_count']} researcher-approved",
        )
    with c4:
        render_metric_card(
            "Annotated Responses",
            str(annotated_count),
            f"{len(df_ann)} total coder entries",
        )

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        render_metric_card(
            "Completed API Calls",
            str(completed_calls),
            f"Pilot: {pilot_completed} | Main: {main_completed}",
        )
    with c6:
        render_metric_card(
            "Failed API Calls",
            str(failed_calls),
            "Preserved in raw JSONL",
        )
    with c7:
        render_metric_card(
            "Pilot Completion",
            f"{pilot_completed} / {PILOT_TARGET_CALLS}",
            f"State: {pilot_state.get('status', 'IDLE')}",
        )
    with c8:
        render_metric_card(
            "Main Completion",
            f"{main_completed} / {MAIN_TARGET_CALLS}",
            f"State: {main_state.get('status', 'IDLE')}",
        )

    st.subheader("Research Phase Progress")
    phases = [
        (
            "Phase 1: API & Thinking-Level Validation (4 levels)",
            1.0 if api_validated else sum(
                1
                for v in load_validation_state().get("levels", {}).values()
                if v.get("validated")
            ) / 4.0,
        ),
        (
            "Phase 2: Matched Scenario Dataset Approval (60 scenarios)",
            min(1.0, val_summary["researcher_approved_count"] / 60.0),
        ),
        (
            "Phase 3: Pilot Experiment (96 calls)",
            min(1.0, pilot_completed / float(PILOT_TARGET_CALLS)),
        ),
        (
            "Phase 4: Main Experiment (1,440 calls)",
            min(1.0, main_completed / float(MAIN_TARGET_CALLS)),
        ),
        (
            "Phase 5: Human Annotation Coverage",
            min(1.0, annotated_count / float(max(1, completed_calls)))
            if completed_calls > 0
            else 0.0,
        ),
    ]
    for label, frac in phases:
        st.write(f"**{label}** — {frac * 100:.1f}%")
        st.progress(frac)

    st.subheader("Experimental Design Matrix")
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**Thinking Levels Evaluated**")
        val_levels = load_validation_state().get("levels", {})
        lvl_rows = []
        for lvl in THINKING_LEVELS:
            info = val_levels.get(lvl, {})
            lvl_rows.append(
                {
                    "Thinking Level": lvl,
                    "API Validated": "Yes" if info.get("validated") else "No",
                    "Last Tested (UTC)": info.get("last_tested_utc") or "Not tested",
                }
            )
        st.dataframe(pd.DataFrame(lvl_rows), use_container_width=True, hide_index=True)

    with col_right:
        st.markdown("**Modalities & Scenario Categories**")
        mod_rows = [
            {"Modality Key": m, "Label": MODALITY_LABELS[m]} for m in MODALITIES
        ]
        st.dataframe(pd.DataFrame(mod_rows), use_container_width=True, hide_index=True)
        cat_rows = [
            {
                "Category": CATEGORY_LABELS[k],
                "Total": val_summary["category_counts"].get(k, 0),
                "Approved": val_summary["approved_by_category"].get(k, 0),
            }
            for k in CATEGORY_LABELS
        ]
        st.dataframe(pd.DataFrame(cat_rows), use_container_width=True, hide_index=True)
