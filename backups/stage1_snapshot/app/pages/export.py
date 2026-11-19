"""PAGE 10 — Export & Research Artifact Downloads."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.components import render_page_header
from app.pages.results_dashboard import generate_png_figures
from config import (
    EXPERIMENT_LOG_MD,
    FIGURES_DIR,
    HUMAN_ANNOTATIONS_CSV,
    IMPLEMENTATION_REPORT_MD,
    MAIN_RAW_JSONL,
    MATCHED_SCENARIOS_PATH,
    PILOT_RAW_JSONL,
    PILOT_REPORT_MD,
    PROCESSED_RESULTS_CSV,
    RESULTS_DIR,
    RUNTIME_CONFIG_PATH,
    STATISTICAL_RESULTS_CSV,
    UNIFIED_RAW_JSONL,
)
from evaluation.agreement import compute_cohens_kappa
from evaluation.annotation import build_processed_results, load_annotations
from evaluation.statistical_analysis import run_full_statistical_analysis
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
        "Export & Research Files",
        "Download Raw Observations (Append-Only), Processed Datasets, Statistical Tables, Figures, and Reports",
    )

    if st.button("Refresh All Processed Tables & PNG Figures", type="primary"):
        export_records_to_csv("pilot", RESULTS_DIR / "pilot_results.csv")
        export_records_to_csv("main", RESULTS_DIR / "main_results.csv")
        df_proc = build_processed_results(None, save_csv=True)
        run_full_statistical_analysis(df_proc, save_csv=True)
        generate_png_figures(df_proc, FIGURES_DIR)
        st.success("Refreshed CSV exports, statistical tables, and PNG figures.")

    st.subheader("Raw & Processed Data Files")
    c1, c2, c3 = st.columns(3)
    with c1:
        _download_file_button("Raw JSONL (Unified)", UNIFIED_RAW_JSONL, "application/jsonl", "dl_raw_all")
        _download_file_button("Raw JSONL (Pilot)", PILOT_RAW_JSONL, "application/jsonl", "dl_raw_pilot")
        _download_file_button("Raw JSONL (Main)", MAIN_RAW_JSONL, "application/jsonl", "dl_raw_main")
    with c2:
        export_records_to_csv("pilot", RESULTS_DIR / "pilot_results.csv")
        export_records_to_csv("main", RESULTS_DIR / "main_results.csv")
        _download_file_button("Pilot CSV", RESULTS_DIR / "pilot_results.csv", "text/csv", "dl_pilot_csv")
        _download_file_button("Main Experiment CSV", RESULTS_DIR / "main_results.csv", "text/csv", "dl_main_csv")
        _download_file_button("Human Annotations CSV", HUMAN_ANNOTATIONS_CSV, "text/csv", "dl_ann_csv")
    with c3:
        _download_file_button("Processed Results CSV", PROCESSED_RESULTS_CSV, "text/csv", "dl_proc_csv")
        _download_file_button("Statistical Results CSV", STATISTICAL_RESULTS_CSV, "text/csv", "dl_stat_csv")
        kappa_json = json.dumps(compute_cohens_kappa(load_annotations()), indent=2)
        st.download_button(
            "Cohen's Kappa Results (JSON)",
            data=kappa_json,
            file_name="cohens_kappa_results.json",
            mime="application/json",
            use_container_width=True,
        )

    st.subheader("Dataset, Configuration, and Research Logs")
    d1, d2, d3 = st.columns(3)
    with d1:
        _download_file_button("Matched Scenarios Dataset (JSON)", MATCHED_SCENARIOS_PATH, "application/json", "dl_dataset")
        _download_file_button("Experiment Configuration (JSON)", RUNTIME_CONFIG_PATH, "application/json", "dl_cfg")
    with d2:
        _download_file_button("Experiment Log (MD)", EXPERIMENT_LOG_MD, "text/markdown", "dl_log")
        _download_file_button("Pilot Report (MD)", PILOT_REPORT_MD, "text/markdown", "dl_pilot_rep")
    with d3:
        _download_file_button("Implementation Report (MD)", IMPLEMENTATION_REPORT_MD, "text/markdown", "dl_impl_rep")

    st.subheader("Generated Figures (PNG)")
    pngs = sorted(FIGURES_DIR.glob("*.png")) if FIGURES_DIR.exists() else []
    if not pngs:
        st.caption("No PNG figures generated yet. Click Refresh above after collecting and annotating trials.")
    else:
        for p in pngs:
            _download_file_button(f"Download {p.name}", p, "image/png", f"dl_fig_{p.name}")
