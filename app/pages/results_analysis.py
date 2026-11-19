"""Results & Analysis — Annotation, Safety Metrics, Comparisons, Statistics, Robustness & Export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st

from app.components import (
    MONOCHROME_CHART_SEQUENCE,
    format_number,
    format_rate,
    render_empty_state,
    render_metric_card,
    render_page_header,
    style_plotly_monochrome,
)
from config import (
    AISTUDIO_WORKFLOW_MD,
    ANNOTATION_LABELS,
    ANNOTATION_LABEL_DISPLAY,
    API_READINESS_REPORT_MD,
    CANDIDATE_SCENARIOS_JSON,
    CATEGORY_LABELS,
    DATASET_REGISTRY_PATH,
    EXPERIMENT_LOG_MD,
    EXPERIMENT_MANIFEST_JSON,
    EXPLORATORY_CHAT_JSONL,
    FIGURES_DIR,
    HUMAN_ANNOTATIONS_CSV,
    IMPLEMENTATION_REPORT_MD,
    INTEGRATION_REPORT_MD,
    LEGITIMATE_CATEGORIES,
    MAIN_MANIFEST_JSON,
    MAIN_RAW_JSONL,
    MAIN_REPORT_MD,
    MATCHED_SCENARIOS_PATH,
    MISSING_FAILED_REPORT_MD,
    MODALITIES,
    MODALITY_LABELS,
    PILOT_MANIFEST_JSON,
    PILOT_RAW_JSONL,
    PILOT_REPORT_MD,
    PROCESSED_RESULTS_CSV,
    RESULTS_DIR,
    ROBUSTNESS_REPORT_MD,
    RUNTIME_CONFIG_PATH,
    SCENARIO_CATEGORIES,
    STATISTICAL_RESULTS_CSV,
    THINKING_LEVELS,
    UNIFIED_RAW_JSONL,
)
from dataset.aistudio_workflow import format_observations_for_aistudio_jsonl
from dataset.scenario_manager import load_active_dataset, load_scenarios, resolve_image_path
from evaluation.agreement import compute_cohens_kappa
from evaluation.annotation import (
    build_processed_results,
    list_disagreements,
    load_annotations,
    save_annotation,
)
from evaluation.metrics import compute_core_metrics
from evaluation.robustness import (
    generate_robustness_figures,
    generate_robustness_report,
    run_robustness_suite,
)
from evaluation.statistical_analysis import (
    generate_statistical_figures,
    run_full_statistical_analysis,
)
from experiment.manifest import build_experiment_manifest
from experiment.storage import export_records_to_csv, load_trial_records

_METRIC_DESCRIPTIONS: Dict[str, str] = {
    "M1": "Share of genuinely unsafe prompts that the model appropriately refused.",
    "M2": "Share of safe or sensitive-looking legitimate requests that were unnecessarily refused.",
    "M3": "Share of genuinely unsafe prompts where the model complied unsafely.",
    "M4": "Share of legitimate requests fulfilled safely and helpfully.",
    "M5": "Share of legitimate requests fulfilled only partially with excessive caveats.",
    "M6": "Share of matched scenario pairs where text-only and image-plus-text yielded different safety decisions.",
    "M7": "End-to-end API response time in milliseconds.",
    "M8": "Prompt tokens reported by the Gemini API usage metadata.",
    "M9": "Visible response tokens reported by the Gemini API usage metadata.",
    "M10": "Internal reasoning tokens reported by the Gemini API usage metadata.",
    "M11": "Human-rated response helpfulness on legitimate requests (1 = not useful, 5 = fully useful).",
}


def _ordered_levels(levels: List[str]) -> List[str]:
    order = {lvl: i for i, lvl in enumerate(THINKING_LEVELS)}
    return sorted(levels, key=lambda x: order.get(x, 99))


def generate_png_figures(df_proc: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> List[Path]:
    """Generate clean monochrome academic PNG figures from actual observations."""
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []
    if df_proc.empty:
        return generated

    by_lvl = compute_core_metrics(df_proc, group_by=["thinking_level"])["grouped"]
    if not by_lvl.empty:
        by_lvl["order"] = by_lvl["thinking_level"].map(
            {l: i for i, l in enumerate(THINKING_LEVELS)}
        )
        by_lvl = by_lvl.sort_values("order")

        plt.rcParams.update(
            {
                "figure.facecolor": "#FFFFFF",
                "axes.facecolor": "#FFFFFF",
                "axes.edgecolor": "#E5E5E5",
                "axes.labelcolor": "#171717",
                "xtick.color": "#737373",
                "ytick.color": "#737373",
                "text.color": "#171717",
            }
        )

        # Chart A: Safety performance
        fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=200)
        x = range(len(by_lvl))
        w = 0.25
        for offset, col, lbl, color in [
            (-w, "correct_refusal_rate", "Correct Refusal", "#171717"),
            (0.0, "over_refusal_rate", "Over-Refusal", "#737373"),
            (w, "unsafe_compliance_rate", "Unsafe Compliance", "#BDBDBD"),
        ]:
            vals = [v * 100 if pd.notna(v) else 0.0 for v in by_lvl[col]]
            ax.bar([i + offset for i in x], vals, width=w, label=lbl, color=color, edgecolor="#171717", linewidth=0.6)
        ax.set_xticks(list(x))
        ax.set_xticklabels(by_lvl["thinking_level"])
        ax.set_ylabel("Rate (%)")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", color="#F3F3F1", linestyle="-", linewidth=1)
        ax.set_axisbelow(True)
        ax.set_title("Safety Performance Across Thinking Levels")
        ax.legend(frameon=True, edgecolor="#E5E5E5")
        fig.tight_layout()
        p_a = out_dir / "chart_a_safety_performance.png"
        fig.savefig(p_a)
        plt.close(fig)
        generated.append(p_a)

        # Chart B: Safe completion
        fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=200)
        for offset, col, lbl, color in [
            (-0.18, "safe_completion_rate", "Safe Completion", "#171717"),
            (0.18, "partial_safe_completion_rate", "Partial Safe Completion", "#8C8C8C"),
        ]:
            vals = [v * 100 if pd.notna(v) else 0.0 for v in by_lvl[col]]
            ax.bar([i + offset for i in x], vals, width=0.35, label=lbl, color=color, edgecolor="#171717", linewidth=0.6)
        ax.set_xticks(list(x))
        ax.set_xticklabels(by_lvl["thinking_level"])
        ax.set_ylabel("Rate (%)")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", color="#F3F3F1", linestyle="-", linewidth=1)
        ax.set_axisbelow(True)
        ax.set_title("Safe Completion on Legitimate Requests")
        ax.legend(frameon=True, edgecolor="#E5E5E5")
        fig.tight_layout()
        p_b = out_dir / "chart_b_safe_completion.png"
        fig.savefig(p_b)
        plt.close(fig)
        generated.append(p_b)

    return generated


def _render_single_trial_coder(
    row: pd.Series,
    scenarios_by_id: dict,
    df_ann: pd.DataFrame,
    annotator_id: str,
    blind_mode: bool = False,
) -> None:
    trial_id = str(row["trial_id"])
    sid = str(row["scenario_id"])
    cat = str(row["category"])
    scen = scenarios_by_id.get(sid, {})

    c_meta, c_resp = st.columns([1, 1.35])
    with c_meta:
        st.markdown(f"**Scenario {sid}** — {scen.get('scenario_title', '')}")
        st.caption(
            f"Category: {CATEGORY_LABELS.get(cat, cat)} · "
            f"Modality: {MODALITY_LABELS.get(str(row['modality']), row['modality'])} · "
            f"Thinking Level: {row['thinking_level']} (Rep {row['repetition']})"
        )
        st.markdown("**Prompt**")
        st.write(str(row["prompt"]))

        if row["modality"] == "image_plus_text":
            img_ref = (
                row.get("image_asset_reference")
                or row.get("image_path")
                or scen.get("image_asset_reference")
                or scen.get("image_path")
            )
            img_path = resolve_image_path(img_ref)
            if img_path and img_path.exists():
                st.image(str(img_path), caption=str(img_path.name), use_container_width=True)
            else:
                st.warning("Associated image not found.")

        if not blind_mode:
            with st.expander("Expected Safe Behavior (Rubric)", expanded=False):
                st.write(scen.get("expected_safe_behavior", "Not specified"))

    with c_resp:
        st.markdown("**Gemini 4 Hydrogen Response**")
        st.text_area(
            "Model Output",
            value=str(row.get("response_text") or "[Empty / Blocked Response]"),
            height=220,
            disabled=True,
            key=f"out_{annotator_id}_{trial_id}",
            label_visibility="collapsed",
        )
        if pd.notna(row.get("block_reason")) and row.get("block_reason"):
            st.caption(f"API block reason: `{row.get('block_reason')}`")

        existing = df_ann[
            (df_ann["trial_id"].astype(str) == trial_id)
            & (df_ann["annotator_id"].astype(str) == annotator_id)
        ]
        prev_label = (
            str(existing.iloc[0]["annotation_label"])
            if not existing.empty
            else ANNOTATION_LABELS[0]
        )
        prev_use = (
            int(existing.iloc[0]["usefulness_score"])
            if (not existing.empty and pd.notna(existing.iloc[0]["usefulness_score"]))
            else 3
        )
        prev_notes = (
            str(existing.iloc[0]["notes"])
            if (not existing.empty and pd.notna(existing.iloc[0]["notes"]))
            else ""
        )
        prev_pseudo = (
            str(existing.iloc[0]["coder_pseudonym"])
            if (
                not existing.empty
                and "coder_pseudonym" in existing.columns
                and pd.notna(existing.iloc[0]["coder_pseudonym"])
            )
            else ""
        )

        with st.form(f"ann_form_{annotator_id}_{trial_id}"):
            coder_pseudonym = st.text_input(
                "Pseudonymous Annotator ID (optional)",
                value=prev_pseudo,
                placeholder="e.g., R1_Anon",
            )
            selected_label = st.radio(
                "Annotation classification",
                options=list(ANNOTATION_LABELS),
                index=list(ANNOTATION_LABELS).index(prev_label)
                if prev_label in ANNOTATION_LABELS
                else 0,
                format_func=lambda k: ANNOTATION_LABEL_DISPLAY[k],
            )
            is_legit = cat in LEGITIMATE_CATEGORIES
            usefulness = None
            if is_legit:
                usefulness = st.slider(
                    "Usefulness score (1 = Not useful, 5 = Fully useful)",
                    min_value=1,
                    max_value=5,
                    value=prev_use,
                )
            else:
                st.caption("Usefulness score applies only to legitimate requests.")

            notes = st.text_area("Notes", value=prev_notes, height=70)
            submitted = st.form_submit_button("Save Annotation", type="primary")
            if submitted:
                save_annotation(
                    trial_id=trial_id,
                    experiment_type=str(row["experiment_type"]),
                    scenario_id=sid,
                    category=cat,
                    modality=str(row["modality"]),
                    thinking_level=str(row["thinking_level"]),
                    repetition=int(row["repetition"]),
                    annotator_id=annotator_id,
                    annotation_label=selected_label,
                    usefulness_score=usefulness if selected_label != "invalid_trial" else None,
                    notes=notes,
                    coder_pseudonym=coder_pseudonym,
                )
                build_processed_results(None, save_csv=True)
                st.success(f"Saved {annotator_id} annotation for {trial_id}.")
                st.rerun()


def _render_annotation_tab(records: List[Dict[str, Any]]) -> None:
    if not records:
        render_empty_state(
            "No experimental responses are available for annotation yet.",
            "Complete the pilot experiment to begin reviewing Gemini responses.",
            "Dual independent human annotation, blind coding, disagreement adjudication, and Cohen's kappa agreement will appear here automatically once trials are recorded.",
        )
        return

    _, pilot_scens = load_active_dataset("pilot")
    _, main_scens = load_active_dataset("main")
    scenarios_by_id = {
        s["scenario_id"]: s for s in (load_scenarios() + pilot_scens + main_scens)
    }
    df_trials = pd.DataFrame(records)
    df_ann = load_annotations()
    kappa_res = compute_cohens_kappa(df_ann)
    disagreements = list_disagreements(df_ann)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Recorded Trials", str(len(df_trials)))
    with m2:
        a1_cnt = int((df_ann["annotator_id"] == "annotator_1").sum()) if not df_ann.empty else 0
        a2_cnt = int((df_ann["annotator_id"] == "annotator_2").sum()) if not df_ann.empty else 0
        render_metric_card("Coder Entries", f"A1: {a1_cnt} · A2: {a2_cnt}")
    with m3:
        kappa_str = f"{kappa_res['kappa']:.3f}" if kappa_res["kappa"] is not None else "Unavailable"
        render_metric_card(
            "Cohen's Kappa",
            kappa_str,
            f"Paired: {kappa_res['paired_count']}" if kappa_res["paired_count"] else kappa_res["message"],
        )
    with m4:
        unresolved = (
            int((disagreements["adjudicator_label"].isna()).sum())
            if not disagreements.empty
            else 0
        )
        render_metric_card("Pending Adjudication", str(unresolved))

    mode = st.radio(
        "Annotation Workflow",
        options=["Blind Human Coding", "Disagreement Adjudication", "Inter-Annotator Agreement"],
        horizontal=True,
        key="ra_ann_mode",
    )

    if mode == "Blind Human Coding":
        c_f1, c_f2, c_f3, c_f4 = st.columns(4)
        with c_f1:
            annotator_id = st.selectbox(
                "Annotator",
                options=["annotator_1", "annotator_2"],
                format_func=lambda x: "Annotator 1" if x == "annotator_1" else "Annotator 2",
            )
        with c_f2:
            exp_filter = st.selectbox("Run scope", options=["All", "pilot", "main"], index=0)
        with c_f3:
            only_unannotated = st.checkbox("Only unannotated", value=True)
        with c_f4:
            blind_mode = st.checkbox("Blind mode (hide rubric)", value=True)

        filtered_trials = df_trials[df_trials["status"].isin(["success", "blocked"])].copy()
        if exp_filter != "All":
            filtered_trials = filtered_trials[filtered_trials["experiment_type"] == exp_filter]

        coded_ids = set(
            df_ann[df_ann["annotator_id"] == annotator_id]["trial_id"].astype(str)
        )
        if only_unannotated:
            filtered_trials = filtered_trials[
                ~filtered_trials["trial_id"].astype(str).isin(coded_ids)
            ]

        if filtered_trials.empty:
            st.info("No remaining unannotated responses for this coder and filter.")
        else:
            trial_ids = filtered_trials["trial_id"].astype(str).tolist()
            if "ann_idx" not in st.session_state:
                st.session_state.ann_idx = 0
            st.session_state.ann_idx = min(st.session_state.ann_idx, len(trial_ids) - 1)

            nav1, nav2, nav3 = st.columns([1, 3, 1])
            with nav1:
                if st.button("Previous", disabled=st.session_state.ann_idx <= 0, key="ann_prev"):
                    st.session_state.ann_idx -= 1
                    st.rerun()
            with nav2:
                selected_tid = st.selectbox(
                    f"Response ({st.session_state.ann_idx + 1} of {len(trial_ids)})",
                    options=trial_ids,
                    index=st.session_state.ann_idx,
                )
                st.session_state.ann_idx = trial_ids.index(selected_tid)
            with nav3:
                if st.button(
                    "Next",
                    disabled=st.session_state.ann_idx >= len(trial_ids) - 1,
                    key="ann_next",
                ):
                    st.session_state.ann_idx += 1
                    st.rerun()

            row = filtered_trials[
                filtered_trials["trial_id"].astype(str) == selected_tid
            ].iloc[0]
            _render_single_trial_coder(
                row, scenarios_by_id, df_ann, annotator_id, blind_mode=blind_mode
            )

    elif mode == "Disagreement Adjudication":
        if disagreements.empty:
            st.info("No annotator disagreements require adjudication.")
        else:
            st.dataframe(disagreements, use_container_width=True, hide_index=True)
            adj_tid = st.selectbox(
                "Select disagreement trial",
                options=disagreements["trial_id"].astype(str).tolist(),
            )
            match_trials = df_trials[df_trials["trial_id"].astype(str) == adj_tid]
            if not match_trials.empty:
                _render_single_trial_coder(
                    match_trials.iloc[0],
                    scenarios_by_id,
                    df_ann,
                    annotator_id="adjudicator",
                )

    else:
        st.write(f"**Status:** {kappa_res['status']} — {kappa_res['message']}")
        if kappa_res["labels_present"]:
            cm_df = pd.DataFrame(
                kappa_res["confusion_matrix"],
                index=kappa_res["labels_present"],
                columns=kappa_res["labels_present"],
            )
            st.dataframe(cm_df, use_container_width=True)
        for title, key in [
            ("By Category", "by_category"),
            ("By Modality", "by_modality"),
            ("By Thinking Level", "by_thinking_level"),
        ]:
            if kappa_res[key]:
                st.markdown(f"**{title}**")
                st.dataframe(pd.DataFrame(kappa_res[key]), use_container_width=True, hide_index=True)


def _render_safety_metrics_tab(df: pd.DataFrame) -> None:
    if df.empty:
        render_empty_state(
            "Safety metrics are not available yet.",
            "Run the pilot or main experiment and annotate responses to view the 11 safety and efficiency metrics.",
            "Missing or unmeasured values are always shown as Unavailable rather than zero.",
        )
        return

    res = compute_core_metrics(df, group_by=["thinking_level"])
    ov = res["overall"]
    excl = res["exclusions"]
    st.caption(
        f"Records: {excl['total_records']} · Evaluable annotated trials: {excl['evaluable_trials']} · "
        f"Excluded: {excl['excluded_failed_api']} API errors, "
        f"{excl['excluded_unannotated_or_unresolved']} unannotated, "
        f"{excl['excluded_invalid_trial']} invalid"
    )
    if excl["total_records"] > 0 and excl["evaluable_trials"] == 0:
        st.info(
            "Raw API latency and token usage (Metrics 7–10) reflect recorded API calls. "
            "Safety decision rates and usefulness (Metrics 1–6, 11) require completed human annotations in the Response Annotation tab."
        )

    r1 = st.columns(4)
    with r1[0]:
        render_metric_card("1. Correct Refusal Rate", format_rate(ov["correct_refusal_rate"]), f"{_METRIC_DESCRIPTIONS['M1']} (n={ov['unsafe_n']})")
    with r1[1]:
        render_metric_card("2. Over-Refusal Rate", format_rate(ov["over_refusal_rate"]), f"{_METRIC_DESCRIPTIONS['M2']} (n={ov['legitimate_n']})")
    with r1[2]:
        render_metric_card("3. Unsafe Compliance Rate", format_rate(ov["unsafe_compliance_rate"]), f"{_METRIC_DESCRIPTIONS['M3']} (n={ov['unsafe_n']})")
    with r1[3]:
        render_metric_card("4. Safe Completion Rate", format_rate(ov["safe_completion_rate"]), f"{_METRIC_DESCRIPTIONS['M4']} (n={ov['legitimate_n']})")

    r2 = st.columns(4)
    with r2[0]:
        render_metric_card("5. Partial Safe Completion", format_rate(ov["partial_safe_completion_rate"]), f"{_METRIC_DESCRIPTIONS['M5']} (n={ov['legitimate_n']})")
    with r2[1]:
        render_metric_card("6. Cross-Modal Disagreement", format_rate(ov["cross_modal_disagreement_rate"]), f"{_METRIC_DESCRIPTIONS['M6']} (pairs={ov['cross_modal_matched_pairs']})")
    with r2[2]:
        render_metric_card(
            "7. Response Latency",
            f"{format_number(ov['latency_mean_ms'], 0, ' ms')}",
            f"Median: {format_number(ov['latency_median_ms'], 0, ' ms')} · n={ov['latency_n']}",
        )
    with r2[3]:
        render_metric_card("11. Usefulness Score (1–5)", format_number(ov["usefulness_mean"], 2), f"{_METRIC_DESCRIPTIONS['M11']} (n={ov['usefulness_n']})")

    r3 = st.columns(3)
    with r3[0]:
        render_metric_card("8. Input Tokens (Mean)", format_number(ov["input_tokens_mean"], 1), f"Available in {ov['input_tokens_n']} calls")
    with r3[1]:
        render_metric_card("9. Output Tokens (Mean)", format_number(ov["output_tokens_mean"], 1), f"Available in {ov['output_tokens_n']} calls")
    with r3[2]:
        render_metric_card("10. Thinking Tokens (Mean)", format_number(ov["thinking_tokens_mean"], 1), f"Available in {ov['thinking_tokens_n']} calls")

    with st.expander("Metric Numerator & Denominator Definitions", expanded=False):
        st.dataframe(pd.DataFrame(res["definitions"]), use_container_width=True, hide_index=True)


def _render_thinking_level_tab(df: pd.DataFrame) -> None:
    if df.empty:
        render_empty_state(
            "No thinking-level observations recorded yet.",
            "Charts comparing LOW, MEDIUM, HIGH, and XHIGH will appear here once experimental responses are collected.",
        )
        return

    res = compute_core_metrics(df, group_by=["thinking_level"])
    ov = res["overall"]
    by_lvl = res["grouped"]
    if not by_lvl.empty:
        lvls = _ordered_levels(by_lvl["thinking_level"].tolist())
        by_lvl["thinking_level"] = pd.Categorical(
            by_lvl["thinking_level"], categories=lvls, ordered=True
        )
        by_lvl = by_lvl.sort_values("thinking_level")

    if by_lvl.empty or ov["evaluable_n"] == 0:
        st.info("Annotate collected trials to compare safety decisions across thinking levels.")
    else:
        rows_a = []
        for _, row in by_lvl.iterrows():
            for col, lbl in [
                ("correct_refusal_rate", "Correct Refusal"),
                ("over_refusal_rate", "Over-Refusal"),
                ("unsafe_compliance_rate", "Unsafe Compliance"),
                ("safe_completion_rate", "Safe Completion"),
            ]:
                if pd.notna(row[col]):
                    rows_a.append(
                        {
                            "Thinking Level": row["thinking_level"],
                            "Metric": lbl,
                            "Rate (%)": row[col] * 100.0,
                        }
                    )
        if rows_a:
            fig_a = px.bar(
                pd.DataFrame(rows_a),
                x="Thinking Level",
                y="Rate (%)",
                color="Metric",
                barmode="group",
                range_y=[0, 105],
                color_discrete_sequence=MONOCHROME_CHART_SEQUENCE,
                title="Safety & Completion Outcomes by Thinking Level",
            )
            st.plotly_chart(style_plotly_monochrome(fig_a), use_container_width=True)

    lat_df = df[df["status"].isin(["success", "blocked"]) & df["latency_ms"].notna()].copy()
    if not lat_df.empty:
        fig_lat = px.box(
            lat_df,
            x="thinking_level",
            y="latency_ms",
            category_orders={"thinking_level": list(THINKING_LEVELS)},
            color_discrete_sequence=["#171717"],
            labels={"thinking_level": "Thinking Level", "latency_ms": "Latency (ms)"},
            title="Response Latency Distribution by Thinking Level",
        )
        st.plotly_chart(style_plotly_monochrome(fig_lat), use_container_width=True)

    tok_rows = []
    api_ok_df = df[df["status"].isin(["success", "blocked"])]
    for lvl, grp in api_ok_df.groupby("thinking_level"):
        for col, lbl in [
            ("input_tokens", "Input Tokens"),
            ("output_tokens", "Output Tokens"),
            ("thinking_tokens", "Thinking Tokens"),
        ]:
            clean = pd.to_numeric(grp[col], errors="coerce").dropna() if col in grp else pd.Series(dtype=float)
            if not clean.empty:
                tok_rows.append(
                    {
                        "Thinking Level": lvl,
                        "Token Type": lbl,
                        "Mean Tokens": float(clean.mean()),
                    }
                )
    if tok_rows:
        fig_tok = px.bar(
            pd.DataFrame(tok_rows),
            x="Thinking Level",
            y="Mean Tokens",
            color="Token Type",
            barmode="group",
            category_orders={"Thinking Level": list(THINKING_LEVELS)},
            color_discrete_sequence=["#171717", "#6E6E6E", "#BDBDBD"],
            title="Mean Token Usage by Thinking Level",
        )
        st.plotly_chart(style_plotly_monochrome(fig_tok), use_container_width=True)

    if st.button("Save Monochrome PNG Figures", key="ra_save_pngs"):
        paths = generate_png_figures(df)
        if paths:
            st.success(f"Saved {len(paths)} PNG figure(s) to `results/figures/`.")
        else:
            st.warning("Annotate collected trials before exporting figures.")


def _render_modality_tab(df: pd.DataFrame) -> None:
    if df.empty:
        render_empty_state(
            "No modality comparison data available yet.",
            "Run matched text-only and image-plus-text trials to compare outcomes, latency, and token usage.",
        )
        return

    res_mod = compute_core_metrics(df, group_by=["modality"])
    by_mod = res_mod["grouped"]
    ov = res_mod["overall"]

    render_metric_card(
        "Cross-Modal Decision Disagreement Rate",
        format_rate(ov["cross_modal_disagreement_rate"]),
        f"Matched pairs evaluated: {ov['cross_modal_matched_pairs']}",
    )
    if ov["evaluable_n"] == 0:
        st.info(
            "Modality latency and token averages reflect raw API responses; cross-modal safety comparisons require human annotation."
        )

    if not by_mod.empty:
        disp_rows = []
        for _, r in by_mod.iterrows():
            disp_rows.append(
                {
                    "Modality": MODALITY_LABELS.get(str(r["modality"]), r["modality"]),
                    "Evaluable Trials": r["evaluable_n"],
                    "Correct Refusal": format_rate(r["correct_refusal_rate"]),
                    "Over-Refusal": format_rate(r["over_refusal_rate"]),
                    "Safe Completion": format_rate(r["safe_completion_rate"]),
                    "Mean Latency": format_number(r["latency_mean_ms"], 0, " ms"),
                    "Mean Input Tokens": format_number(r["input_tokens_mean"], 1),
                    "Mean Output Tokens": format_number(r["output_tokens_mean"], 1),
                    "Mean Thinking Tokens": format_number(r["thinking_tokens_mean"], 1),
                }
            )
        st.dataframe(pd.DataFrame(disp_rows), use_container_width=True, hide_index=True)

    by_lm = compute_core_metrics(df, group_by=["thinking_level", "modality"])["grouped"]
    if not by_lm.empty and ov["evaluable_n"] > 0:
        rows_c = []
        for _, row in by_lm.iterrows():
            mod_name = MODALITY_LABELS.get(str(row["modality"]), str(row["modality"]))
            for col, lbl in [
                ("correct_refusal_rate", "Correct Refusal"),
                ("over_refusal_rate", "Over-Refusal"),
                ("safe_completion_rate", "Safe Completion"),
            ]:
                if pd.notna(row[col]):
                    rows_c.append(
                        {
                            "Thinking Level": row["thinking_level"],
                            "Modality": mod_name,
                            "Metric": lbl,
                            "Rate (%)": row[col] * 100.0,
                        }
                    )
        if rows_c:
            fig_c = px.bar(
                pd.DataFrame(rows_c),
                x="Thinking Level",
                y="Rate (%)",
                color="Modality",
                facet_col="Metric",
                barmode="group",
                range_y=[0, 105],
                category_orders={"Thinking Level": list(THINKING_LEVELS)},
                color_discrete_sequence=["#171717", "#8C8C8C"],
                title="Text-Only vs. Image + Text Across Thinking Levels",
            )
            st.plotly_chart(style_plotly_monochrome(fig_c), use_container_width=True)

    lat_df = df[df["status"].isin(["success", "blocked"]) & df["latency_ms"].notna()]
    if not lat_df.empty:
        lat_mod = lat_df.copy()
        lat_mod["Modality"] = lat_mod["modality"].map(lambda m: MODALITY_LABELS.get(m, m))
        fig_lm = px.box(
            lat_mod,
            x="Modality",
            y="latency_ms",
            color="Modality",
            color_discrete_sequence=["#171717", "#737373"],
            labels={"latency_ms": "Latency (ms)"},
            title="Response Latency by Input Modality",
        )
        st.plotly_chart(style_plotly_monochrome(fig_lm), use_container_width=True)


def _render_statistical_tab(df: pd.DataFrame) -> None:
    if df.empty:
        render_empty_state(
            "No statistical results available yet.",
            "Collect and annotate experimental trials to compute confidence intervals, paired hypothesis tests, Cohen's kappa, and mixed-effects models.",
        )
        return

    stats_res = run_full_statistical_analysis(df, save_csv=False)
    excl = stats_res["exclusions"]
    kappa = stats_res["cohens_kappa"]
    mc_df = stats_res["mcnemar_tests"]
    cont_df = stats_res["paired_continuous_tests"]

    # Plain-English Summary First
    st.markdown("**Plain-Language Summary**")
    summary_parts = [
        f"Evaluated **{excl['evaluable_trials']}** annotated trials (from {excl['total_records']} total recorded calls)."
    ]
    if kappa["kappa"] is not None:
        summary_parts.append(
            f"Inter-annotator agreement (Cohen's kappa) is **{kappa['kappa']:.3f}** across {kappa['paired_count']} double-coded trials."
        )
    else:
        summary_parts.append(f"Inter-annotator agreement: {kappa['message']}")

    if not mc_df.empty and "significant_holm_0_05" in mc_df.columns:
        sig_cnt = int(mc_df["significant_holm_0_05"].fillna(False).sum())
        summary_parts.append(
            f"Paired McNemar comparisons ran on {len(mc_df)} contrasts; **{sig_cnt}** reached significance (Holm-adjusted p < 0.05)."
        )
    else:
        summary_parts.append("Paired categorical comparisons do not yet have enough annotated pairs.")

    st.write(" ".join(summary_parts))

    if st.button("Compute & Save Statistical CSVs and Figures", type="primary", key="ra_run_stats"):
        stats_saved = run_full_statistical_analysis(df, save_csv=True)
        figs = generate_statistical_figures(stats_saved, FIGURES_DIR)
        st.success(f"Saved statistical CSV files and {len(figs)} figure(s) to `results/`.")

    with st.expander("Confidence Intervals (Wilson 95% & Scenario-Clustered Bootstrap)", expanded=False):
        desc_df = stats_res["descriptive_table"]
        if desc_df.empty:
            st.info("Insufficient annotated data for descriptive confidence intervals.")
        else:
            st.dataframe(desc_df, use_container_width=True, hide_index=True)

        boot_df = stats_res["clustered_bootstrap"]
        if boot_df.empty:
            st.info("Need at least 3 distinct annotated scenarios for clustered bootstrap CIs.")
        else:
            st.dataframe(boot_df, use_container_width=True, hide_index=True)

    with st.expander("Paired Hypothesis Tests, Effect Sizes & P-Values", expanded=False):
        if mc_df.empty:
            st.info("Insufficient paired annotated data for McNemar tests.")
        else:
            st.dataframe(mc_df, use_container_width=True, hide_index=True)
        if cont_df.empty:
            st.info("Insufficient paired observations for Wilcoxon signed-rank tests.")
        else:
            st.dataframe(cont_df, use_container_width=True, hide_index=True)

    with st.expander("Cohen's Kappa Inter-Annotator Agreement", expanded=False):
        st.write(
            f"Paired trials: {kappa['paired_count']} · Status: {kappa['status']} · Kappa: {kappa['kappa']}"
        )

    with st.expander("Mixed-Effects & Scenario-Clustered Logistic Models", expanded=False):
        models = stats_res["logistic_models"]
        if not models:
            st.info("Insufficient annotated data to fit regression models.")
        else:
            for mod in models:
                st.markdown(f"**[{mod.get('model_type', 'Model')}] {mod['outcome']}** — {mod['status']}")
                st.caption(f"n={mod['n_obs']}, scenarios={mod['n_scenarios']} — {mod['message']}")
                if not mod["coefficients"].empty:
                    st.dataframe(mod["coefficients"], use_container_width=True, hide_index=True)

    with st.expander("Cluster-Adjusted Statistical Power Assessment", expanded=False):
        power_df = stats_res.get("power_assessment", pd.DataFrame())
        if power_df.empty:
            st.info("Power assessment unavailable.")
        else:
            st.dataframe(power_df, use_container_width=True, hide_index=True)


def _render_robustness_tab(df: pd.DataFrame) -> None:
    if df.empty:
        render_empty_state(
            "No robustness checks available yet.",
            "Execute trials and record annotations to evaluate subgroup stability, repetition variability, annotation sensitivity, and scenario influence.",
        )
        return

    rob = run_robustness_suite(df, save_outputs=False)
    ver = rob["model_version_consistency"]
    rep = rob["repetition_variability"]
    ds_ver = rob["dataset_version_comparison"]
    excl = rob["exclusions"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Model Version", "Consistent" if ver["consistent"] else "Drift detected", ver["message"])
    with c2:
        render_metric_card("Dataset Versions", str(len(ds_ver["versions_observed"])), "Single version" if not ds_ver["multiple_versions"] else "Multiple versions")
    with c3:
        render_metric_card("Repetition Instability", format_rate(rep["instability_rate"]), f"{rep['unstable_conditions']}/{rep['multi_rep_conditions']} conditions flipped")
    with c4:
        render_metric_card("Excluded Trials", str(excl["excluded_failed_api"] + excl["excluded_invalid_trial"]), f"Errors: {excl['excluded_failed_api']} · Invalid: {excl['excluded_invalid_trial']}")

    st.markdown("**Robust vs. Sensitive vs. Insufficient Data**")
    st.dataframe(rob["robustness_findings"], use_container_width=True, hide_index=True)

    with st.expander("Detailed Robustness Breakdown Tables", expanded=False):
        st.markdown("**By Category**")
        st.dataframe(rob["by_category"], use_container_width=True, hide_index=True)
        st.markdown("**By Modality**")
        st.dataframe(rob["by_modality"], use_container_width=True, hide_index=True)
        st.markdown("**By Thinking Level**")
        st.dataframe(rob["by_thinking_level"], use_container_width=True, hide_index=True)
        if not rep["detail_table"].empty:
            st.markdown("**Repeated-Trial Variability**")
            st.dataframe(rep["detail_table"], use_container_width=True, hide_index=True)
        if not rob["annotation_sensitivity"].empty:
            st.markdown("**Annotation Disagreement Sensitivity**")
            st.dataframe(rob["annotation_sensitivity"], use_container_width=True, hide_index=True)
        if not rob["loso_influence"].empty:
            st.markdown("**Leave-One-Scenario-Out Influence**")
            st.dataframe(rob["loso_influence"], use_container_width=True, hide_index=True)
        if not rob["missing_failed_analysis"].empty:
            st.markdown("**Missing & Failed Trial Strata**")
            st.dataframe(rob["missing_failed_analysis"], use_container_width=True, hide_index=True)
        if not ds_ver["summary_table"].empty:
            st.markdown("**Dataset-Version Differences**")
            st.dataframe(ds_ver["summary_table"], use_container_width=True, hide_index=True)

    if st.button("Generate Robustness Report & Figures", key="ra_gen_rob"):
        run_robustness_suite(df, save_outputs=True)
        figs = generate_robustness_figures(rob, FIGURES_DIR)
        generate_robustness_report(rob, ROBUSTNESS_REPORT_MD)
        st.success(f"Saved `research/robustness_report.md`, CSV summary, and {len(figs)} figure(s).")


def _download_button_or_placeholder(label: str, path: Path, mime: str, key: str) -> None:
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
            f"{label} (Unavailable)",
            disabled=True,
            key=f"dis_{key}",
            use_container_width=True,
        )


def _render_export_tab() -> None:
    if st.button(
        "Refresh All Exports, Manifests, Statistics & Figures",
        type="primary",
        key="ra_refresh_exports",
    ):
        if PILOT_RAW_JSONL.exists() and PILOT_RAW_JSONL.stat().st_size > 0:
            export_records_to_csv("pilot", RESULTS_DIR / "pilot_results.csv")
        if MAIN_RAW_JSONL.exists() and MAIN_RAW_JSONL.stat().st_size > 0:
            export_records_to_csv("main", RESULTS_DIR / "main_results.csv")
        df_proc = build_processed_results(None, save_csv=True)
        if not df_proc.empty:
            stats_res = run_full_statistical_analysis(df_proc, save_csv=True)
            generate_statistical_figures(stats_res, FIGURES_DIR)
            run_robustness_suite(df_proc, save_outputs=True)
            generate_png_figures(df_proc, FIGURES_DIR)
        build_experiment_manifest("pilot", save_path=PILOT_MANIFEST_JSON)
        build_experiment_manifest("main", save_path=MAIN_MANIFEST_JSON)
        build_experiment_manifest(None, save_path=EXPERIMENT_MANIFEST_JSON)
        st.success("Refreshed available manifests and exports.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Raw JSONL & AI Studio Logs**")
        _download_button_or_placeholder("Unified Raw JSONL", UNIFIED_RAW_JSONL, "application/jsonl", "exp_raw_all")
        _download_button_or_placeholder("Pilot Raw JSONL", PILOT_RAW_JSONL, "application/jsonl", "exp_raw_pilot")
        _download_button_or_placeholder("Main Raw JSONL", MAIN_RAW_JSONL, "application/jsonl", "exp_raw_main")
        ai_jsonl = format_observations_for_aistudio_jsonl(None)
        st.download_button(
            "AI Studio Observations JSONL",
            data=ai_jsonl or "",
            file_name="aistudio_observations_export.jsonl",
            mime="application/jsonl",
            disabled=not bool(ai_jsonl),
            use_container_width=True,
            key="exp_ai_jsonl",
        )
    with c2:
        st.markdown("**CSV & Statistical Results**")
        _download_button_or_placeholder("Pilot Results CSV", RESULTS_DIR / "pilot_results.csv", "text/csv", "exp_pilot_csv")
        _download_button_or_placeholder("Main Results CSV", RESULTS_DIR / "main_results.csv", "text/csv", "exp_main_csv")
        _download_button_or_placeholder("Processed Results CSV", PROCESSED_RESULTS_CSV, "text/csv", "exp_proc_csv")
        _download_button_or_placeholder("Human Annotations CSV", HUMAN_ANNOTATIONS_CSV, "text/csv", "exp_ann_csv")
        _download_button_or_placeholder("Statistical Results CSV", STATISTICAL_RESULTS_CSV, "text/csv", "exp_stat_csv")
        _download_button_or_placeholder("Statistical McNemar CSV", RESULTS_DIR / "statistical_mcnemar.csv", "text/csv", "exp_mcn_csv")
        _download_button_or_placeholder("Statistical Power CSV", RESULTS_DIR / "statistical_power.csv", "text/csv", "exp_pwr_csv")
        _download_button_or_placeholder("Robustness Summary CSV", RESULTS_DIR / "robustness_summary.csv", "text/csv", "exp_rob_csv")
    with c3:
        st.markdown("**JSON Summaries & Research Reports**")
        _download_button_or_placeholder("Pilot Manifest JSON", PILOT_MANIFEST_JSON, "application/json", "exp_pman")
        _download_button_or_placeholder("Main Manifest JSON", MAIN_MANIFEST_JSON, "application/json", "exp_mman")
        _download_button_or_placeholder("Unified Manifest JSON", EXPERIMENT_MANIFEST_JSON, "application/json", "exp_uman")
        _download_button_or_placeholder("Matched Scenarios JSON", MATCHED_SCENARIOS_PATH, "application/json", "exp_scen")
        _download_button_or_placeholder("Dataset Registry JSON", DATASET_REGISTRY_PATH, "application/json", "exp_reg")
        _download_button_or_placeholder("Runtime Config JSON", RUNTIME_CONFIG_PATH, "application/json", "exp_cfg")
        kappa_json = json.dumps(compute_cohens_kappa(load_annotations()), indent=2)
        st.download_button(
            "Cohen's Kappa JSON",
            data=kappa_json,
            file_name="cohens_kappa_results.json",
            mime="application/json",
            use_container_width=True,
            key="exp_kappa",
        )

    with st.expander("Research Reports & Generated Figures", expanded=False):
        rep_c1, rep_c2 = st.columns(2)
        with rep_c1:
            _download_button_or_placeholder("API Readiness Report (MD)", API_READINESS_REPORT_MD, "text/markdown", "exp_api_md")
            _download_button_or_placeholder("Pilot Report (MD)", PILOT_REPORT_MD, "text/markdown", "exp_pil_md")
            _download_button_or_placeholder("Main Experiment Report (MD)", MAIN_REPORT_MD, "text/markdown", "exp_main_md")
            _download_button_or_placeholder("Missing/Failed Trials Report (MD)", MISSING_FAILED_REPORT_MD, "text/markdown", "exp_mf_md")
        with rep_c2:
            _download_button_or_placeholder("Robustness Report (MD)", ROBUSTNESS_REPORT_MD, "text/markdown", "exp_rob_md")
            _download_button_or_placeholder("Experiment Log (MD)", EXPERIMENT_LOG_MD, "text/markdown", "exp_log_md")
            _download_button_or_placeholder("AI Studio Workflow (MD)", AISTUDIO_WORKFLOW_MD, "text/markdown", "exp_ais_md")
            _download_button_or_placeholder("Implementation Report (MD)", IMPLEMENTATION_REPORT_MD, "text/markdown", "exp_imp_md")
            _download_button_or_placeholder("Integration Test Report (MD)", INTEGRATION_REPORT_MD, "text/markdown", "exp_int_md")

        pngs = sorted(FIGURES_DIR.glob("*.png")) if FIGURES_DIR.exists() else []
        if pngs:
            st.markdown("**Figures (PNG)**")
            for p in pngs:
                _download_button_or_placeholder(f"Download {p.name}", p, "image/png", f"exp_fig_{p.name}")
        else:
            st.caption("No PNG figures generated yet.")

    with st.expander("Exploratory Lab Exports (Separate from Official Research)", expanded=False):
        st.caption(
            "Exploratory prompts and candidate scenarios saved in Experiment Lab remain strictly isolated from Pilot and Main datasets."
        )
        ex_c1, ex_c2 = st.columns(2)
        with ex_c1:
            _download_button_or_placeholder(
                "Candidate Scenarios (JSON)",
                CANDIDATE_SCENARIOS_JSON,
                "application/json",
                "exp_lab_candidates",
            )
        with ex_c2:
            _download_button_or_placeholder(
                "Exploratory Chat History (JSONL)",
                EXPLORATORY_CHAT_JSONL,
                "application/jsonl",
                "exp_lab_chat",
            )


def render() -> None:
    render_page_header(
        "Results & Analysis",
        "Annotate model responses, review safety metrics, compare thinking levels and modalities, and export data.",
    )

    records = load_trial_records(None)
    df_all = build_processed_results(None, save_csv=False)
    df_filtered = df_all

    if not df_all.empty:
        with st.expander("Filter Experimental Observations", expanded=False):
            f1, f2, f3, f4 = st.columns(4)
            with f1:
                exp_opts = sorted(df_all["experiment_type"].dropna().unique().tolist())
                sel_exp = st.multiselect("Run Scope", options=exp_opts, default=exp_opts)
            with f2:
                sel_lvl = st.multiselect("Thinking Level", options=list(THINKING_LEVELS), default=list(THINKING_LEVELS))
            with f3:
                sel_mod = st.multiselect(
                    "Input Modality",
                    options=list(MODALITIES),
                    default=list(MODALITIES),
                    format_func=lambda m: MODALITY_LABELS.get(m, m),
                )
            with f4:
                sel_cat = st.multiselect(
                    "Category",
                    options=list(SCENARIO_CATEGORIES),
                    default=list(SCENARIO_CATEGORIES),
                    format_func=lambda c: CATEGORY_LABELS.get(c, c),
                )
            df_filtered = df_all[
                df_all["experiment_type"].isin(sel_exp)
                & df_all["thinking_level"].isin(sel_lvl)
                & df_all["modality"].isin(sel_mod)
                & df_all["category"].isin(sel_cat)
            ].copy()

    tabs = st.tabs(
        [
            "Response Annotation",
            "Safety Metrics",
            "Thinking-Level Comparison",
            "Modality Comparison",
            "Statistical Analysis",
            "Robustness",
            "Export",
        ]
    )

    with tabs[0]:
        _render_annotation_tab(records)
    with tabs[1]:
        _render_safety_metrics_tab(df_filtered)
    with tabs[2]:
        _render_thinking_level_tab(df_filtered)
    with tabs[3]:
        _render_modality_tab(df_filtered)
    with tabs[4]:
        _render_statistical_tab(df_filtered)
    with tabs[5]:
        _render_robustness_tab(df_filtered)
    with tabs[6]:
        _render_export_tab()
