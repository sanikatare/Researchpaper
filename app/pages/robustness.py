"""PAGE 9 — Robustness & Sensitivity Analysis."""

from __future__ import annotations

import streamlit as st

from app.components import format_rate, render_metric_card, render_page_header
from config import FIGURES_DIR, ROBUSTNESS_REPORT_MD
from evaluation.annotation import build_processed_results
from evaluation.robustness import (
    generate_robustness_figures,
    generate_robustness_report,
    run_robustness_suite,
)


def render() -> None:
    render_page_header(
        "Robustness Analysis",
        "Subgroup Stability, Repetition Variability, Annotation Sensitivity, LOSO Influence, Dataset Versions, and Findings Audit",
    )

    df_proc = build_processed_results(None, save_csv=True)
    if df_proc.empty:
        st.info("No experimental observations recorded yet.")
        return

    rob = run_robustness_suite(df_proc, save_outputs=False)
    ver = rob["model_version_consistency"]
    rep = rob["repetition_variability"]
    ds_ver = rob["dataset_version_comparison"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card(
            "Model Version",
            "Consistent" if ver["consistent"] else "DRIFT DETECTED",
            ver["message"],
        )
    with c2:
        render_metric_card(
            "Dataset Versions",
            str(len(ds_ver["versions_observed"])),
            "Multiple versions" if ds_ver["multiple_versions"] else "Single version",
        )
    with c3:
        render_metric_card(
            "Repeated-Trial Instability",
            format_rate(rep["instability_rate"]),
            f"{rep['unstable_conditions']}/{rep['multi_rep_conditions']} conditions flipped across reps",
        )
    with c4:
        excl = rob["exclusions"]
        render_metric_card(
            "Missing / Failed Trials",
            str(excl["excluded_failed_api"] + excl["excluded_invalid_trial"]),
            f"API errors: {excl['excluded_failed_api']} | Invalid: {excl['excluded_invalid_trial']}",
        )

    st.subheader("Robust vs. Sensitive Findings Summary")
    st.dataframe(rob["robustness_findings"], use_container_width=True, hide_index=True)

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

    st.subheader("6. Dataset-Version Differences")
    if ds_ver["summary_table"].empty:
        st.info(ds_ver["message"])
    else:
        st.caption(ds_ver["message"])
        st.dataframe(ds_ver["summary_table"], use_container_width=True, hide_index=True)

    if st.button("Generate Robustness Report, CSV & PNG Figures", type="primary"):
        run_robustness_suite(df_proc, save_outputs=True)
        figs = generate_robustness_figures(rob, FIGURES_DIR)
        rep_md = generate_robustness_report(rob, ROBUSTNESS_REPORT_MD)
        st.success(
            f"Saved `research/robustness_report.md`, `results/robustness_summary.csv`, "
            f"and {len(figs)} PNG figure(s)."
        )
        st.markdown(rep_md)
