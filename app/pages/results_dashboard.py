"""PAGE 7 — Interactive Results Dashboard & Google-Colored Charts (Charts A–G)."""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st

from app.components import (
    GOOGLE_CHART_SEQUENCE,
    format_number,
    format_rate,
    render_metric_card,
    render_page_header,
    style_plotly_light,
)
from config import (
    CATEGORY_LABELS,
    FIGURES_DIR,
    MODALITIES,
    MODALITY_LABELS,
    SCENARIO_CATEGORIES,
    THINKING_LEVELS,
)
from evaluation.annotation import build_processed_results
from evaluation.metrics import compute_core_metrics


def _ordered_levels(levels: List[str]) -> List[str]:
    order = {lvl: i for i, lvl in enumerate(THINKING_LEVELS)}
    return sorted(levels, key=lambda x: order.get(x, 99))


def generate_png_figures(df_proc: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> List[Path]:
    """Generate clean light-background Google-colored PNG figures from actual observations."""
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
                "axes.edgecolor": "#DADCE0",
                "axes.labelcolor": "#202124",
                "xtick.color": "#5F6368",
                "ytick.color": "#5F6368",
                "text.color": "#202124",
            }
        )

        # Chart A: Safety performance
        fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=200)
        x = range(len(by_lvl))
        w = 0.25
        for offset, col, lbl, color in [
            (-w, "correct_refusal_rate", "Correct Refusal", "#34A853"),
            (0.0, "over_refusal_rate", "Over-Refusal", "#FBBC04"),
            (w, "unsafe_compliance_rate", "Unsafe Compliance", "#EA4335"),
        ]:
            vals = [v * 100 if pd.notna(v) else 0.0 for v in by_lvl[col]]
            ax.bar([i + offset for i in x], vals, width=w, label=lbl, color=color)
        ax.set_xticks(list(x))
        ax.set_xticklabels(by_lvl["thinking_level"])
        ax.set_ylabel("Rate (%)")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", color="#F1F3F4", linestyle="-", linewidth=1)
        ax.set_axisbelow(True)
        ax.set_title("Chart A: Safety Performance Across Thinking Levels")
        ax.legend(frameon=True, edgecolor="#DADCE0")
        fig.tight_layout()
        p_a = out_dir / "chart_a_safety_performance.png"
        fig.savefig(p_a)
        plt.close(fig)
        generated.append(p_a)

        # Chart B: Safe completion
        fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=200)
        for offset, col, lbl, color in [
            (-0.18, "safe_completion_rate", "Safe Completion", "#4285F4"),
            (0.18, "partial_safe_completion_rate", "Partial Safe Completion", "#FBBC04"),
        ]:
            vals = [v * 100 if pd.notna(v) else 0.0 for v in by_lvl[col]]
            ax.bar([i + offset for i in x], vals, width=0.35, label=lbl, color=color)
        ax.set_xticks(list(x))
        ax.set_xticklabels(by_lvl["thinking_level"])
        ax.set_ylabel("Rate (%)")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", color="#F1F3F4", linestyle="-", linewidth=1)
        ax.set_axisbelow(True)
        ax.set_title("Chart B: Safe Completion on Legitimate Requests")
        ax.legend(frameon=True, edgecolor="#DADCE0")
        fig.tight_layout()
        p_b = out_dir / "chart_b_safe_completion.png"
        fig.savefig(p_b)
        plt.close(fig)
        generated.append(p_b)

    return generated


def render() -> None:
    render_page_header(
        "Results Dashboard",
        "Empirical Safety, Cross-Modal Consistency, Latency, Token Usage, and Usefulness Metrics",
    )

    df_all = build_processed_results(None, save_csv=False)
    if df_all.empty:
        st.info(
            "No experimental trials collected yet. Execute the Pilot or Main experiment "
            "and record human annotations to populate empirical charts."
        )
        return

    st.subheader("Data Filters")
    f1, f2, f3, f4, f5 = st.columns(5)
    with f1:
        exp_opts = sorted(df_all["experiment_type"].dropna().unique().tolist())
        sel_exp = st.multiselect("Experiment Type", options=exp_opts, default=exp_opts)
    with f2:
        sel_lvl = st.multiselect(
            "Thinking Level", options=list(THINKING_LEVELS), default=list(THINKING_LEVELS)
        )
    with f3:
        sel_mod = st.multiselect(
            "Input Modality",
            options=list(MODALITIES),
            default=list(MODALITIES),
            format_func=lambda m: MODALITY_LABELS.get(m, m),
        )
    with f4:
        sel_cat = st.multiselect(
            "Scenario Category",
            options=list(SCENARIO_CATEGORIES),
            default=list(SCENARIO_CATEGORIES),
            format_func=lambda c: CATEGORY_LABELS.get(c, c),
        )
    with f5:
        rep_opts = sorted(df_all["repetition"].dropna().astype(int).unique().tolist())
        sel_rep = st.multiselect("Repetition", options=rep_opts, default=rep_opts)

    df = df_all[
        df_all["experiment_type"].isin(sel_exp)
        & df_all["thinking_level"].isin(sel_lvl)
        & df_all["modality"].isin(sel_mod)
        & df_all["category"].isin(sel_cat)
        & df_all["repetition"].astype(int).isin(sel_rep)
    ].copy()

    if df.empty:
        st.warning("No observations match the selected filters.")
        return

    res = compute_core_metrics(df, group_by=["thinking_level"])
    ov = res["overall"]
    excl = res["exclusions"]

    st.caption(
        f"Filtered Records: {excl['total_records']} | Evaluable Annotated Trials: {excl['evaluable_trials']} | "
        f"Excluded: {excl['excluded_failed_api']} API errors, "
        f"{excl['excluded_unannotated_or_unresolved']} unannotated/unresolved, "
        f"{excl['excluded_invalid_trial']} invalid trials."
    )

    r1 = st.columns(4)
    with r1[0]:
        render_metric_card("1. Correct Refusal Rate", format_rate(ov["correct_refusal_rate"]), f"n={ov['unsafe_n']} unsafe", accent_color="#34A853")
    with r1[1]:
        render_metric_card("2. Over-Refusal Rate", format_rate(ov["over_refusal_rate"]), f"n={ov['legitimate_n']} legit", accent_color="#FBBC04")
    with r1[2]:
        render_metric_card("3. Unsafe Compliance Rate", format_rate(ov["unsafe_compliance_rate"]), f"n={ov['unsafe_n']} unsafe", accent_color="#EA4335")
    with r1[3]:
        render_metric_card("4. Safe Completion Rate", format_rate(ov["safe_completion_rate"]), f"n={ov['legitimate_n']} legit", accent_color="#4285F4")

    r2 = st.columns(4)
    with r2[0]:
        render_metric_card("5. Partial Safe Completion", format_rate(ov["partial_safe_completion_rate"]), f"n={ov['legitimate_n']} legit")
    with r2[1]:
        render_metric_card(
            "6. Cross-Modal Disagreement",
            format_rate(ov["cross_modal_disagreement_rate"]),
            f"Matched pairs: {ov['cross_modal_matched_pairs']}",
        )
    with r2[2]:
        render_metric_card(
            "7. Latency (Mean / Med)",
            f"{format_number(ov['latency_mean_ms'], 0, ' ms')} / {format_number(ov['latency_median_ms'], 0, ' ms')}",
            f"n={ov['latency_n']}",
        )
    with r2[3]:
        render_metric_card(
            "11. Usefulness Score (1–5)",
            format_number(ov["usefulness_mean"], 2),
            f"n={ov['usefulness_n']} legit",
        )

    r3 = st.columns(3)
    with r3[0]:
        render_metric_card("8. Input Tokens (Mean)", format_number(ov["input_tokens_mean"], 1), f"Available: {ov['input_tokens_n']} calls")
    with r3[1]:
        render_metric_card("9. Output Tokens (Mean)", format_number(ov["output_tokens_mean"], 1), f"Available: {ov['output_tokens_n']} calls")
    with r3[2]:
        render_metric_card("10. Thinking Tokens (Mean)", format_number(ov["thinking_tokens_mean"], 1), f"Available: {ov['thinking_tokens_n']} calls")

    by_lvl = res["grouped"]
    if not by_lvl.empty:
        lvls = _ordered_levels(by_lvl["thinking_level"].tolist())
        by_lvl["thinking_level"] = pd.Categorical(
            by_lvl["thinking_level"], categories=lvls, ordered=True
        )
        by_lvl = by_lvl.sort_values("thinking_level")

    # Chart A
    st.subheader("Chart A — Safety Performance Across Thinking Levels")
    if by_lvl.empty or ov["evaluable_n"] == 0:
        st.info("Annotate collected trials to render Chart A.")
    else:
        rows_a = []
        for _, row in by_lvl.iterrows():
            for col, lo, hi, lbl in [
                ("correct_refusal_rate", "correct_refusal_ci_low", "correct_refusal_ci_high", "Correct Refusal"),
                ("over_refusal_rate", "over_refusal_ci_low", "over_refusal_ci_high", "Over-Refusal"),
                ("unsafe_compliance_rate", "unsafe_compliance_ci_low", "unsafe_compliance_ci_high", "Unsafe Compliance"),
            ]:
                if pd.notna(row[col]):
                    rows_a.append(
                        {
                            "Thinking Level": row["thinking_level"],
                            "Metric": lbl,
                            "Rate (%)": row[col] * 100.0,
                            "Err High": (row[hi] - row[col]) * 100.0 if pd.notna(row[hi]) else 0.0,
                            "Err Low": (row[col] - row[lo]) * 100.0 if pd.notna(row[lo]) else 0.0,
                        }
                    )
        if rows_a:
            fig_a = px.bar(
                pd.DataFrame(rows_a),
                x="Thinking Level",
                y="Rate (%)",
                color="Metric",
                barmode="group",
                error_y="Err High",
                error_y_minus="Err Low",
                range_y=[0, 105],
                color_discrete_map={
                    "Correct Refusal": "#34A853",
                    "Over-Refusal": "#FBBC04",
                    "Unsafe Compliance": "#EA4335",
                },
            )
            st.plotly_chart(style_plotly_light(fig_a), use_container_width=True)

    # Chart B
    st.subheader("Chart B — Safe Completion & Partial Safe Completion")
    if by_lvl.empty or ov["legitimate_n"] == 0:
        st.info("Annotate legitimate request trials to render Chart B.")
    else:
        rows_b = []
        for _, row in by_lvl.iterrows():
            for col, lo, hi, lbl in [
                ("safe_completion_rate", "safe_completion_ci_low", "safe_completion_ci_high", "Safe Completion"),
                ("partial_safe_completion_rate", "partial_safe_completion_ci_low", "partial_safe_completion_ci_high", "Partial Safe Completion"),
            ]:
                if pd.notna(row[col]):
                    rows_b.append(
                        {
                            "Thinking Level": row["thinking_level"],
                            "Outcome": lbl,
                            "Rate (%)": row[col] * 100.0,
                            "Err High": (row[hi] - row[col]) * 100.0 if pd.notna(row[hi]) else 0.0,
                            "Err Low": (row[col] - row[lo]) * 100.0 if pd.notna(row[lo]) else 0.0,
                        }
                    )
        if rows_b:
            fig_b = px.bar(
                pd.DataFrame(rows_b),
                x="Thinking Level",
                y="Rate (%)",
                color="Outcome",
                barmode="group",
                error_y="Err High",
                error_y_minus="Err Low",
                range_y=[0, 105],
                color_discrete_map={
                    "Safe Completion": "#4285F4",
                    "Partial Safe Completion": "#FBBC04",
                },
            )
            st.plotly_chart(style_plotly_light(fig_b), use_container_width=True)

    # Chart C
    st.subheader("Chart C — Cross-Modal Comparison (Text-Only vs Image + Text)")
    by_mod = compute_core_metrics(df, group_by=["thinking_level", "modality"])["grouped"]
    if by_mod.empty or ov["evaluable_n"] == 0:
        st.info("Annotate trials across both modalities to render Chart C.")
    else:
        rows_c = []
        for _, row in by_mod.iterrows():
            mod_name = MODALITY_LABELS.get(row["modality"], row["modality"])
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
                color_discrete_sequence=["#4285F4", "#34A853"],
            )
            st.plotly_chart(style_plotly_light(fig_c), use_container_width=True)

    # Chart D
    st.subheader("Chart D — Inference Latency Distribution by Thinking Level")
    lat_df = df[df["status"].isin(["success", "blocked"]) & df["latency_ms"].notna()].copy()
    if lat_df.empty:
        st.info("No latency observations available.")
    else:
        fig_d = px.box(
            lat_df,
            x="thinking_level",
            y="latency_ms",
            color="modality",
            points="outliers",
            category_orders={"thinking_level": list(THINKING_LEVELS)},
            color_discrete_sequence=["#4285F4", "#34A853"],
            labels={"thinking_level": "Thinking Level", "latency_ms": "Latency (ms)"},
        )
        st.plotly_chart(style_plotly_light(fig_d), use_container_width=True)

    # Chart E
    st.subheader("Chart E — Available Token Usage by Thinking Level")
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
        fig_e = px.bar(
            pd.DataFrame(tok_rows),
            x="Thinking Level",
            y="Mean Tokens",
            color="Token Type",
            barmode="group",
            category_orders={"Thinking Level": list(THINKING_LEVELS)},
            color_discrete_sequence=GOOGLE_CHART_SEQUENCE,
        )
        st.plotly_chart(style_plotly_light(fig_e), use_container_width=True)
    else:
        st.info("Token usage metadata is currently unavailable in recorded responses.")

    # Chart F
    st.subheader("Chart F — Usefulness Scores on Legitimate Requests")
    use_df = df[
        df["resolved_usefulness_score"].notna()
        & df["category"].isin(["clearly_benign", "sensitive_looking_legitimate"])
    ].copy()
    if use_df.empty:
        st.info("No usefulness scores annotated yet.")
    else:
        fig_f = px.box(
            use_df,
            x="thinking_level",
            y="resolved_usefulness_score",
            color="modality",
            category_orders={"thinking_level": list(THINKING_LEVELS)},
            color_discrete_sequence=["#4285F4", "#34A853"],
            labels={"thinking_level": "Thinking Level", "resolved_usefulness_score": "Usefulness (1–5)"},
        )
        fig_f.update_yaxes(range=[0.5, 5.5])
        st.plotly_chart(style_plotly_light(fig_f), use_container_width=True)

    # Chart G
    st.subheader("Chart G — Safety Performance vs Inference Efficiency")
    if not by_lvl.empty and by_lvl["latency_mean_ms"].notna().any():
        eff_rows = by_lvl.dropna(subset=["latency_mean_ms"]).copy()
        eff_rows["Composite Safety Accuracy (%)"] = (
            eff_rows[["correct_refusal_rate", "safe_completion_rate"]]
            .mean(axis=1, skipna=True)
            * 100.0
        )
        eff_rows = eff_rows.dropna(subset=["Composite Safety Accuracy (%)"])
        if not eff_rows.empty:
            fig_g = px.scatter(
                eff_rows,
                x="latency_mean_ms",
                y="Composite Safety Accuracy (%)",
                text="thinking_level",
                color="thinking_level",
                size="evaluable_n",
                color_discrete_sequence=GOOGLE_CHART_SEQUENCE,
                labels={"latency_mean_ms": "Mean Latency (ms)"},
            )
            fig_g.update_traces(textposition="top center")
            st.plotly_chart(style_plotly_light(fig_g), use_container_width=True)
        else:
            st.info("Annotate trials to plot safety vs latency efficiency.")

    if st.button("Save Publication PNG Figures to results/figures/"):
        paths = generate_png_figures(df)
        if paths:
            st.success(f"Saved {len(paths)} light-theme PNG figure(s) in `results/figures/`.")
        else:
            st.warning("Not enough annotated observations to render PNG figures yet.")

    with st.expander("Explicit Definitions for All 11 Research Metrics"):
        st.dataframe(pd.DataFrame(res["definitions"]), use_container_width=True, hide_index=True)

