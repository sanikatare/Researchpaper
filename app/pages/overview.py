"""PAGE 1 — Research Overview (Google-Inspired Light Theme)."""

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
from dataset.scenario_manager import load_active_dataset, load_registry
from evaluation.annotation import load_annotations, resolve_final_annotations
from experiment.hydrogen_client import (
    get_api_key,
    is_api_fully_validated,
    load_validation_state,
)
from experiment.storage import load_experiment_state, load_trial_records


def render() -> None:
    render_page_header(
        "Research Overview",
        "Controlled Empirical Workbench for Gemini 4 Hydrogen Cross-Modal Safety Evaluation",
    )

    st.markdown(f"### {RESEARCH_TITLE}")
    st.markdown(f"**Research Objective:** {RESEARCH_OBJECTIVE}")

    api_validated = is_api_fully_validated()
    api_key_present = bool(get_api_key())
    if not api_validated:
        st.warning(
            "API configuration has not been validated across all four thinking levels "
            "(LOW, MEDIUM, HIGH, XHIGH). Visit **2. API Configuration** to verify live connectivity."
        )

    pilot_entry, pilot_scenarios = load_active_dataset("pilot")
    main_entry, main_scenarios = load_active_dataset("main")
    reg = load_registry()

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

    pilot_completed = sum(
        1 for r in pilot_records if r.get("status") in ("success", "blocked")
    )
    main_completed = sum(
        1 for r in main_records if r.get("status") in ("success", "blocked")
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card(
            "Experimental Model",
            DEFAULT_MODEL_ID,
            "Exclusive model under study",
            accent_color="#4285F4",
        )
    with c2:
        conn_label = (
            "Validated (4/4)"
            if api_validated
            else ("Key Set (Unverified)" if api_key_present else "No API Key")
        )
        conn_color = "#34A853" if api_validated else "#FBBC04"
        render_metric_card(
            "API Connection Status",
            conn_label,
            "Read from environment / Secrets",
            accent_color=conn_color,
        )
    with c3:
        render_metric_card(
            "Current Dataset Version",
            main_entry["dataset_version"] if main_entry else "None",
            f"Pilot: {pilot_entry['dataset_version'] if pilot_entry else 'None'}",
            accent_color="#4285F4",
        )
    with c4:
        render_metric_card(
            "Total Scenario Families",
            str(len(main_scenarios)),
            f"Approval: {main_entry.get('researcher_approval_status', 'N/A') if main_entry else 'N/A'}",
            accent_color="#34A853"
            if main_entry and main_entry.get("researcher_approval_status") == "approved"
            else "#FBBC04",
        )

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        render_metric_card(
            "Total Completed Calls",
            str(completed_calls),
            f"Pilot: {pilot_completed} | Main: {main_completed}",
            accent_color="#4285F4",
        )
    with c6:
        render_metric_card(
            "Failed Calls",
            str(failed_calls),
            "Logged in append-only JSONL",
            accent_color="#EA4335" if failed_calls > 0 else "#DADCE0",
        )
    with c7:
        render_metric_card(
            "Annotated Responses",
            str(annotated_count),
            f"{len(df_ann)} independent coder entries",
            accent_color="#34A853",
        )
    with c8:
        pilot_state = load_experiment_state("pilot")
        main_state = load_experiment_state("main")
        render_metric_card(
            "Registered Versions",
            str(len(reg.get("versions", []))),
            f"Pilot: {pilot_state.get('status', 'IDLE')} | Main: {main_state.get('status', 'IDLE')}",
        )

    st.subheader("Research Phase Progress")
    val_levels = load_validation_state().get("levels", {})
    validated_lvl_count = sum(1 for v in val_levels.values() if v.get("validated"))

    p1 = validated_lvl_count / 4.0
    p2 = (
        1.0
        if (
            main_entry
            and main_entry.get("researcher_approval_status") == "approved"
            and pilot_entry
            and pilot_entry.get("researcher_approval_status") == "approved"
        )
        else (0.5 if (main_entry or pilot_entry) else 0.0)
    )
    p3 = min(1.0, pilot_completed / float(PILOT_TARGET_CALLS))
    p4 = min(1.0, main_completed / float(MAIN_TARGET_CALLS))
    p5 = (
        min(1.0, annotated_count / float(completed_calls))
        if completed_calls > 0
        else 0.0
    )

    for label, frac in [
        ("1. Thinking-Level API Verification (4 levels)", p1),
        ("2. AI Studio Dataset Approval (Pilot & Main)", p2),
        (f"3. Pilot Experiment Progress ({pilot_completed}/{PILOT_TARGET_CALLS})", p3),
        (f"4. Main Experiment Progress ({main_completed}/{MAIN_TARGET_CALLS})", p4),
        (f"5. Human Response Annotation ({annotated_count}/{completed_calls})", p5),
    ]:
        st.write(f"**{label}** — {frac * 100:.1f}%")
        st.progress(frac)

    st.subheader("Thinking-Level Availability & Experimental Design")
    col_l, col_r = st.columns(2)
    with col_l:
        lvl_rows = []
        for lvl in THINKING_LEVELS:
            info = val_levels.get(lvl, {})
            lvl_rows.append(
                {
                    "Thinking Level": lvl,
                    "Availability": "Validated" if info.get("validated") else "Not Validated",
                    "Observed Model Version": info.get("actual_model_version") or "N/A",
                    "Last Tested (UTC)": info.get("last_tested_utc") or "Never",
                }
            )
        st.dataframe(pd.DataFrame(lvl_rows), use_container_width=True, hide_index=True)

    with col_r:
        cat_counts = main_entry.get("category_counts", {}) if main_entry else {}
        design_rows = [
            {
                "Category": CATEGORY_LABELS[k],
                "Main Dataset Count": cat_counts.get(k, 0),
                "Expected Behavior": (
                    "Appropriate Refusal" if k == "genuinely_unsafe" else "Safe Completion"
                ),
            }
            for k in CATEGORY_LABELS
        ]
        st.dataframe(pd.DataFrame(design_rows), use_container_width=True, hide_index=True)
        st.caption(
            f"Modalities: {', '.join(MODALITY_LABELS[m] for m in MODALITIES)} | "
            "Google AI Studio serves as the exclusive dataset authoring & curation environment."
        )
