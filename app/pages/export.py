"""PAGE 10 — Export Results, Google AI Studio JSONL Logs, and Research Files."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from app.components import render_page_header
from app.pages.results_dashboard import generate_png_figures
from config import (
    AISTUDIO_WORKFLOW_MD,
    DATASET_REGISTRY_PATH,
    EXPERIMENT_LOG_MD,
    EXPERIMENT_MANIFEST_JSON,
    FIGURES_DIR,
    HUMAN_ANNOTATIONS_CSV,
    IMPLEMENTATION_REPORT_MD,
    INTEGRATION_REPORT_MD,
    MAIN_RAW_JSONL,
    MATCHED_SCENARIOS_PATH,
    PILOT_RAW_JSONL,
    PILOT_REPORT_MD,
    PROCESSED_RESULTS_CSV,
    RESULTS_DIR,
    ROBUSTNESS_REPORT_MD,
    RUNTIME_CONFIG_PATH,
    STATISTICAL_RESULTS_CSV,
    UNIFIED_RAW_JSONL,
)
from dataset.aistudio_workflow import format_observations_for_aistudio_jsonl
from evaluation.agreement import compute_cohens_kappa
from evaluation.annotation import build_processed_results, load_annotations
from evaluation.robustness import run_robustness_suite
from evaluation.statistical_analysis import (
    generate_statistical_figures,
    run_full_statistical_analysis,
)
from experiment.manifest import build_experiment_manifest
from experiment.storage import export_records_to_csv


def _download_file_button(label: str, path: Path, mime: str, key: str) -> None:
    if path.exists() and path.stat().st_size > 0:
        st.download_button(
            label=label,
            data=path.read_bytes(),
            file_name=path.name,
            mime=mime,
            key=key,
            use_container_width=True,
        )
    else:
        st.button(
            f"{label} (Not Generated Yet)",
            disabled=True,
            key=f"dis_{key}",
            use_container_width=True,
        )


def render() -> None:
    render_page_header(
        "Export Results",
        "Download Versioned Datasets, Immutable Raw Observations, AI Studio JSONL Logs, Statistics, and Figures",
    )

    if st.button("Refresh All Processed CSVs, Stats, Robustness, Manifest & PNG Figures", type="primary"):
        export_records_to_csv("pilot", RESULTS_DIR / "pilot_results.csv")
        export_records_to_csv("main", RESULTS_DIR / "main_results.csv")
        df_proc = build_processed_results(None, save_csv=True)
        stats_res = run_full_statistical_analysis(df_proc, save_csv=True)
        generate_statistical_figures(stats_res, FIGURES_DIR)
        run_robustness_suite(df_proc, save_outputs=True)
        generate_png_figures(df_proc, FIGURES_DIR)
        build_experiment_manifest(None, save_path=EXPERIMENT_MANIFEST_JSON)
        st.success("Refreshed CSV exports, statistical tables, robustness report, manifest, and PNG figures.")

    st.subheader("Raw Observations & Processed Datasets")
    c1, c2, c3 = st.columns(3)
    with c1:
        _download_file_button("Raw Observations (Unified JSONL)", UNIFIED_RAW_JSONL, "application/jsonl", "dl_raw_all")
        _download_file_button("Pilot Raw JSONL", PILOT_RAW_JSONL, "application/jsonl", "dl_raw_pilot")
        _download_file_button("Main Raw JSONL", MAIN_RAW_JSONL, "application/jsonl", "dl_raw_main")
        _download_file_button("Experiment Manifest (JSON)", EXPERIMENT_MANIFEST_JSON, "application/json", "dl_manifest")
    with c2:
        export_records_to_csv("pilot", RESULTS_DIR / "pilot_results.csv")
        export_records_to_csv("main", RESULTS_DIR / "main_results.csv")
        _download_file_button("Pilot Results CSV", RESULTS_DIR / "pilot_results.csv", "text/csv", "dl_pilot_csv")
        _download_file_button("Main Results CSV", RESULTS_DIR / "main_results.csv", "text/csv", "dl_main_csv")
        _download_file_button("Human Annotations CSV", HUMAN_ANNOTATIONS_CSV, "text/csv", "dl_ann_csv")
        _download_file_button("Robustness Summary CSV", RESULTS_DIR / "robustness_summary.csv", "text/csv", "dl_rob_csv")
    with c3:
        _download_file_button("Processed Data CSV", PROCESSED_RESULTS_CSV, "text/csv", "dl_proc_csv")
        _download_file_button("Statistical Results CSV", STATISTICAL_RESULTS_CSV, "text/csv", "dl_stat_csv")
        _download_file_button("Statistical McNemar CSV", RESULTS_DIR / "statistical_mcnemar.csv", "text/csv", "dl_mcn_csv")
        _download_file_button("Statistical Power CSV", RESULTS_DIR / "statistical_power.csv", "text/csv", "dl_pwr_csv")
        kappa_json = json.dumps(compute_cohens_kappa(load_annotations()), indent=2)
        st.download_button(
            "Cohen's Kappa Results (JSON)",
            data=kappa_json,
            file_name="cohens_kappa_results.json",
            mime="application/json",
            use_container_width=True,
        )

    st.subheader("Google AI Studio Dataset & Log Interoperability")
    a1, a2, a3 = st.columns(3)
    with a1:
        _download_file_button("Matched Scenarios JSON", MATCHED_SCENARIOS_PATH, "application/json", "dl_dataset")
        _download_file_button("Dataset Version Registry JSON", DATASET_REGISTRY_PATH, "application/json", "dl_reg")
    with a2:
        ai_jsonl = format_observations_for_aistudio_jsonl(None)
        st.download_button(
            "Export Observations (AI Studio JSONL)",
            data=ai_jsonl or "",
            file_name="aistudio_observations_export.jsonl",
            mime="application/jsonl",
            disabled=not bool(ai_jsonl),
            use_container_width=True,
        )
        _download_file_button("Experiment Config JSON", RUNTIME_CONFIG_PATH, "application/json", "dl_cfg")
    with a3:
        _download_file_button("Experiment Log (MD)", EXPERIMENT_LOG_MD, "text/markdown", "dl_log")
        _download_file_button("Pilot Report (MD)", PILOT_REPORT_MD, "text/markdown", "dl_pilot_rep")
        _download_file_button("Robustness Report (MD)", ROBUSTNESS_REPORT_MD, "text/markdown", "dl_rob_rep")
        _download_file_button("Integration Test Report (MD)", INTEGRATION_REPORT_MD, "text/markdown", "dl_int_rep")
        _download_file_button("AI Studio Workflow Doc (MD)", AISTUDIO_WORKFLOW_MD, "text/markdown", "dl_ais_doc")
        _download_file_button("Implementation Report (MD)", IMPLEMENTATION_REPORT_MD, "text/markdown", "dl_impl_rep")

    st.subheader("Generated Figures (PNG)")
    pngs = sorted(FIGURES_DIR.glob("*.png")) if FIGURES_DIR.exists() else []
    if not pngs:
        st.caption("No PNG figures generated yet.")
    else:
        for p in pngs:
            _download_file_button(f"Download {p.name}", p, "image/png", f"dl_fig_{p.name}")
