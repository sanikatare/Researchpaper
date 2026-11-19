"""PAGE 9 — Robustness & Sensitivity Analysis."""

from __future__ import annotations

import streamlit as st

from app.components import format_rate, render_metric_card, render_page_header
from evaluation.annotation import build_processed_results
from evaluation.robustness import run_robustness_suite


def render() -> None:
    render_page_header(
        "Robustness Analysis",
        "Subgroup Stability, Repetition Variability, Annotation Sensitivity, LOSO Influence, and Version Checks",
    )

    df_proc = build_processed_results(None, save_csv=True)
    if df_proc.empty:
        st.info("No experimental observations recorded yet.")
        return

    rob = run_robustness_suite(df_proc)
    ver = rob["model_version_consistency"]
    rep = rob["repetition_variability"]

    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card(
            "Model Version Consistency",
            "Consistent" if ver["consistent"] else "DRIFT DETECTED",
            ver["message"],
        )
    with c2:
        render_metric_card(
            "Repeated-Trial Instability",
            format_rate(rep["instability_rate"]),
            f"{rep['unstable_conditions']}/{rep['multi_rep_conditions']} conditions flipped across reps",
        )
    with c3:
        excl = rob["exclusions"]
        render_metric_card(
            "Missing / Failed Trials",
            str(excl["excluded_failed_api"] + excl["excluded_invalid_trial"]),
            f"API errors: {excl['excluded_failed_api']} | Invalid: {excl['excluded_invalid_trial']}",
        )

    st.subheader("1. Subgroup Comparisons (Category, Modality, Thinking Level)")
    t1, t2, t3 = st.tabs(["By Category", "By Modality", "By Thinking Level"])
    with t1:
        st.dataframe(rob["by_category"], use_container_width=True, hide_index=True)
    with t2:
        st.dataframe(rob["by_modality"], use_container_width=True, hide_index=True)
    with t3:
        st.dataframe(rob["by_thinking_level"], use_container_width=True, hide_index=True)

    st.subheader("2. Repeated-Trial Variability")
    if rep["detail_table"].empty:
        st.info("Repeated-trial variability requires >= 2 annotated repetitions per condition.")
    else:
        st.dataframe(rep["detail_table"], use_container_width=True, hide_index=True)

    st.subheader("3. Annotation Disagreement Sensitivity")
    if rob["annotation_sensitivity"].empty:
        st.info("No human annotations available for sensitivity comparison.")
    else:
        st.dataframe(rob["annotation_sensitivity"], use_container_width=True, hide_index=True)

    st.subheader("4. Scenario Influence Analysis (Leave-One-Scenario-Out)")
    if rob["loso_influence"].empty:
        st.info("Need >= 2 annotated scenarios for LOSO influence analysis.")
    else:
        st.dataframe(rob["loso_influence"], use_container_width=True, hide_index=True)

    st.subheader("5. Missing & Failed Trial Strata")
    if rob["missing_failed_analysis"].empty:
        st.info("No strata records available.")
    else:
        st.dataframe(rob["missing_failed_analysis"], use_container_width=True, hide_index=True)
