"""PAGE 8 — Statistical Analysis."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components import render_page_header
from evaluation.agreement import compute_cohens_kappa
from evaluation.annotation import build_processed_results, load_annotations
from evaluation.statistical_analysis import run_full_statistical_analysis


def render() -> None:
    render_page_header(
        "Statistical Analysis",
        "Descriptive 95% CIs, Paired McNemar & Wilcoxon Tests, Clustered Bootstrap, and Mixed-Effects / GEE Logistic Regression",
    )

    df_proc = build_processed_results(None, save_csv=True)
    if df_proc.empty:
        st.info("No experimental observations recorded yet.")
        return

    exp_scope = st.selectbox("Analysis Scope", options=["All", "pilot", "main"], index=0)
    if exp_scope != "All":
        df_proc = df_proc[df_proc["experiment_type"] == exp_scope].copy()

    stats_res = run_full_statistical_analysis(df_proc, save_csv=True)
    excl = stats_res["exclusions"]
    st.caption(
        f"Total records: {excl['total_records']} | Evaluable: {excl['evaluable_trials']} | "
        f"Excluded (API error={excl['excluded_failed_api']}, Unannotated={excl['excluded_unannotated_or_unresolved']}, "
        f"Invalid={excl['excluded_invalid_trial']})"
    )

    st.subheader("1. Descriptive Statistics & Wilson 95% Confidence Intervals")
    desc_df = stats_res["descriptive_table"]
    if desc_df.empty:
        st.info("Insufficient annotated data for descriptive table.")
    else:
        st.dataframe(desc_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Descriptive Statistics CSV",
            data=desc_df.to_csv(index=False),
            file_name="descriptive_statistical_results.csv",
            mime="text/csv",
        )

    st.subheader("2. Paired Comparisons (McNemar's Test & Holm-Bonferroni Correction)")
    mc_df = stats_res["mcnemar_tests"]
    if mc_df.empty:
        st.info("Insufficient paired annotated data for McNemar tests.")
    else:
        st.dataframe(mc_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download McNemar Results CSV",
            data=mc_df.to_csv(index=False),
            file_name="mcnemar_paired_tests.csv",
            mime="text/csv",
        )

    st.subheader("3. Paired Continuous Comparisons (Wilcoxon Signed-Rank & Cohen's dz)")
    cont_df = stats_res["paired_continuous_tests"]
    if cont_df.empty:
        st.info("Insufficient paired observations for continuous tests.")
    else:
        st.dataframe(cont_df, use_container_width=True, hide_index=True)

    st.subheader("4. Scenario-Clustered Bootstrap 95% Confidence Intervals")
    boot_df = stats_res["clustered_bootstrap"]
    if boot_df.empty:
        st.info("Need at least 3 distinct annotated scenarios for clustered bootstrap CIs.")
    else:
        st.dataframe(boot_df, use_container_width=True, hide_index=True)

    st.subheader("5. Scenario-Clustered Logistic Models (Fixed: Thinking Level, Modality, Category | Cluster: Scenario)")
    models = stats_res["logistic_models"]
    if not models:
        st.info("Insufficient annotated data to fit regression models.")
    else:
        for mod in models:
            with st.expander(f"{mod['outcome']} — Status: `{mod['status']}`", expanded=True):
                st.write(
                    f"Observations: {mod['n_obs']} | Scenarios: {mod['n_scenarios']} | "
                    f"Positives: {mod['positives']}"
                )
                st.caption(mod["message"])
                coef_df = mod["coefficients"]
                if not coef_df.empty:
                    st.dataframe(coef_df, use_container_width=True, hide_index=True)

    st.subheader("6. Inter-Annotator Agreement (Cohen's Kappa)")
    kappa = compute_cohens_kappa(load_annotations())
    st.write(
        f"Paired trials: {kappa['paired_count']} | Status: `{kappa['status']}` | "
        f"Kappa: `{kappa['kappa']}`"
    )
