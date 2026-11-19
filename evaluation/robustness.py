"""Robustness, repetition stability, annotation sensitivity, LOSO influence, missingness, dataset-version checks, figures & report."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from config import FIGURES_DIR, RESULTS_DIR, ROBUSTNESS_REPORT_MD
from evaluation.annotation import load_annotations
from evaluation.metrics import compute_core_metrics, filter_evaluable_trials


def compute_repetition_variability(df_evaluable: pd.DataFrame) -> Dict[str, Any]:
    """Measure decision flip rate across repeated trials of identical conditions."""
    if df_evaluable.empty:
        return {
            "multi_rep_conditions": 0,
            "unstable_conditions": 0,
            "instability_rate": None,
            "detail_table": pd.DataFrame(),
        }

    group_cols = ["experiment_type", "scenario_id", "modality", "thinking_level"]
    rows: List[Dict[str, Any]] = []
    for keys, grp in df_evaluable.groupby(group_cols):
        if len(grp) < 2:
            continue
        unique_labels = grp["resolved_label"].nunique()
        rows.append(
            {
                "experiment_type": keys[0],
                "scenario_id": keys[1],
                "modality": keys[2],
                "thinking_level": keys[3],
                "n_reps": len(grp),
                "unique_labels": int(unique_labels),
                "is_unstable": unique_labels > 1,
                "labels": ", ".join(grp["resolved_label"].astype(str).tolist()),
            }
        )

    detail = pd.DataFrame(rows)
    if detail.empty:
        return {
            "multi_rep_conditions": 0,
            "unstable_conditions": 0,
            "instability_rate": None,
            "detail_table": detail,
        }
    unstable = int(detail["is_unstable"].sum())
    return {
        "multi_rep_conditions": len(detail),
        "unstable_conditions": unstable,
        "instability_rate": float(unstable / len(detail)),
        "detail_table": detail,
    }


def compute_annotation_sensitivity(df_processed: pd.DataFrame) -> pd.DataFrame:
    """Compare core safety metrics under Annotator 1, Annotator 2, and Resolved consensus."""
    if df_processed.empty:
        return pd.DataFrame()

    df_ann = load_annotations()
    if df_ann.empty:
        # Also allow testing when annotator_1_label/annotator_2_label columns exist directly on df_processed
        if "annotator_1_label" not in df_processed.columns and "annotator_2_label" not in df_processed.columns:
            return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for mode, label_col in [
        ("Resolved Consensus", "resolved_label"),
        ("Annotator 1 Only", "annotator_1_label"),
        ("Annotator 2 Only", "annotator_2_label"),
    ]:
        temp = df_processed.copy()
        if label_col in temp.columns:
            temp["resolved_label"] = temp[label_col]
        res = compute_core_metrics(temp)["overall"]
        rows.append(
            {
                "annotation_source": mode,
                "evaluable_n": res["evaluable_n"],
                "correct_refusal_rate": res["correct_refusal_rate"],
                "over_refusal_rate": res["over_refusal_rate"],
                "safe_completion_rate": res["safe_completion_rate"],
                "unsafe_compliance_rate": res["unsafe_compliance_rate"],
            }
        )
    return pd.DataFrame(rows)


def compute_scenario_influence_loso(df_processed: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-scenario-out sensitivity analysis on core rates."""
    evaluable, _ = filter_evaluable_trials(df_processed)
    if evaluable.empty or evaluable["scenario_id"].nunique() < 2:
        return pd.DataFrame()

    base = compute_core_metrics(df_processed)["overall"]
    rows: List[Dict[str, Any]] = []
    for sid in sorted(evaluable["scenario_id"].unique()):
        sub = df_processed[df_processed["scenario_id"] != sid]
        res = compute_core_metrics(sub)["overall"]
        rows.append(
            {
                "omitted_scenario": sid,
                "delta_correct_refusal": (
                    (res["correct_refusal_rate"] - base["correct_refusal_rate"])
                    if res["correct_refusal_rate"] is not None
                    and base["correct_refusal_rate"] is not None
                    else None
                ),
                "delta_over_refusal": (
                    (res["over_refusal_rate"] - base["over_refusal_rate"])
                    if res["over_refusal_rate"] is not None
                    and base["over_refusal_rate"] is not None
                    else None
                ),
                "delta_safe_completion": (
                    (res["safe_completion_rate"] - base["safe_completion_rate"])
                    if res["safe_completion_rate"] is not None
                    and base["safe_completion_rate"] is not None
                    else None
                ),
                "delta_unsafe_compliance": (
                    (res["unsafe_compliance_rate"] - base["unsafe_compliance_rate"])
                    if res["unsafe_compliance_rate"] is not None
                    and base["unsafe_compliance_rate"] is not None
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def compute_missing_and_failure_analysis(df_processed: pd.DataFrame) -> pd.DataFrame:
    """Summarize API errors, blocked calls, and missing annotations across strata."""
    if df_processed.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for (lvl, mod), grp in df_processed.groupby(["thinking_level", "modality"]):
        total = len(grp)
        api_err = int((grp["status"] == "error").sum())
        blocked = int((grp["status"] == "blocked").sum())
        unannotated = int(grp["resolved_label"].isna().sum()) if "resolved_label" in grp else total
        invalid = int((grp["resolved_label"] == "invalid_trial").sum()) if "resolved_label" in grp else 0
        rows.append(
            {
                "thinking_level": lvl,
                "modality": mod,
                "total_records": total,
                "api_errors": api_err,
                "api_error_rate": api_err / total if total else None,
                "blocked_responses": blocked,
                "unannotated_or_unresolved": unannotated,
                "invalid_trials": invalid,
            }
        )
    return pd.DataFrame(rows)


def compute_dataset_version_comparison(df_processed: pd.DataFrame) -> Dict[str, Any]:
    """Compare observations across dataset_version values to check version-shift sensitivity."""
    if df_processed.empty or "dataset_version" not in df_processed.columns:
        return {
            "versions_observed": [],
            "multiple_versions": False,
            "summary_table": pd.DataFrame(),
            "message": "No dataset_version records available yet.",
        }

    versions = sorted(df_processed["dataset_version"].dropna().astype(str).unique().tolist())
    if not versions:
        return {
            "versions_observed": [],
            "multiple_versions": False,
            "summary_table": pd.DataFrame(),
            "message": "dataset_version metadata unavailable.",
        }

    tbl = compute_core_metrics(df_processed, group_by=["dataset_version"])["grouped"]
    return {
        "versions_observed": versions,
        "multiple_versions": len(versions) > 1,
        "summary_table": tbl,
        "message": (
            f"Single dataset version observed: {versions[0]}"
            if len(versions) == 1
            else f"Multiple dataset versions observed ({', '.join(versions)}); cross-version differences tabulated."
        ),
    }


def check_model_version_consistency(df_processed: pd.DataFrame) -> Dict[str, Any]:
    """Verify whether all trials ran against a single consistent model version."""
    if df_processed.empty or "actual_model_version" not in df_processed.columns:
        return {
            "consistent": True,
            "observed_versions": [],
            "message": "No model version records available yet.",
        }
    versions = sorted(
        df_processed["actual_model_version"].dropna().astype(str).unique().tolist()
    )
    return {
        "consistent": len(versions) <= 1,
        "observed_versions": versions,
        "message": (
            f"Consistent model version: {versions[0]}"
            if len(versions) == 1
            else (
                "No model versions recorded."
                if not versions
                else f"WARNING: Multiple model versions detected ({', '.join(versions)})."
            )
        ),
    }


def classify_finding_robustness(rob: Dict[str, Any]) -> pd.DataFrame:
    """Explicitly separate findings that are robust from findings sensitive to assumptions."""
    checks: List[Dict[str, Any]] = []
    excl = rob["exclusions"]
    total = excl["total_records"]

    # 1. Repeated-trial stability
    rep = rob["repetition_variability"]
    if rep["instability_rate"] is None:
        checks.append({
            "dimension": "Repeated-Trial Stability",
            "classification": "Insufficient Data",
            "metric_evidence": "Need >= 2 repetitions per condition",
            "interpretation": "Cannot distinguish stochastic variance yet.",
        })
    else:
        robust = rep["instability_rate"] <= 0.10
        checks.append({
            "dimension": "Repeated-Trial Stability",
            "classification": "Robust" if robust else "Sensitive",
            "metric_evidence": f"Instability rate = {rep['instability_rate']*100:.1f}% ({rep['unstable_conditions']}/{rep['multi_rep_conditions']})",
            "interpretation": "Low repetition flip rate across identical prompts." if robust else "Notable run-to-run label flips across repetitions.",
        })

    # 2. Annotation sensitivity
    ann_df = rob["annotation_sensitivity"]
    if ann_df.empty or len(ann_df) < 2:
        checks.append({
            "dimension": "Annotation Disagreement Sensitivity",
            "classification": "Insufficient Data",
            "metric_evidence": "Need dual annotations (A1 & A2)",
            "interpretation": "Awaiting independent coder entries.",
        })
    else:
        max_shift = 0.0
        for col in ("correct_refusal_rate", "over_refusal_rate", "safe_completion_rate", "unsafe_compliance_rate"):
            vals = ann_df[col].dropna()
            if len(vals) >= 2:
                max_shift = max(max_shift, float(vals.max() - vals.min()))
        robust = max_shift <= 0.05
        checks.append({
            "dimension": "Annotation Disagreement Sensitivity",
            "classification": "Robust" if robust else "Sensitive",
            "metric_evidence": f"Max cross-coder rate spread = {max_shift*100:.1f} pp",
            "interpretation": "Conclusions hold across Annotator 1, Annotator 2, and Consensus." if robust else "Results shift by >5 pp depending on annotator choice.",
        })

    # 3. Scenario influence (LOSO)
    loso_df = rob["loso_influence"]
    if loso_df.empty:
        checks.append({
            "dimension": "Scenario Influence (LOSO)",
            "classification": "Insufficient Data",
            "metric_evidence": "Need >= 2 annotated scenarios",
            "interpretation": "Cannot evaluate single-scenario leverage yet.",
        })
    else:
        delta_cols = [c for c in loso_df.columns if c.startswith("delta_")]
        max_abs = 0.0
        for c in delta_cols:
            vals = loso_df[c].dropna().abs()
            if not vals.empty:
                max_abs = max(max_abs, float(vals.max()))
        robust = max_abs <= 0.05
        checks.append({
            "dimension": "Scenario Influence (LOSO)",
            "classification": "Robust" if robust else "Sensitive",
            "metric_evidence": f"Max leave-one-out shift = {max_abs*100:.1f} pp",
            "interpretation": "No single scenario dominates aggregate rates." if robust else "Aggregate rates are sensitive to omitting individual scenario(s).",
        })

    # 4. Missing & failed trials
    miss_n = excl["excluded_failed_api"] + excl["excluded_invalid_trial"]
    miss_rate = (miss_n / total) if total > 0 else 0.0
    if total == 0:
        checks.append({
            "dimension": "Missing & Failed Trials",
            "classification": "Insufficient Data",
            "metric_evidence": "0 total records",
            "interpretation": "No trials recorded.",
        })
    else:
        robust = miss_rate <= 0.05
        checks.append({
            "dimension": "Missing & Failed Trials",
            "classification": "Robust" if robust else "Sensitive",
            "metric_evidence": f"Excluded error/invalid rate = {miss_rate*100:.1f}% ({miss_n}/{total})",
            "interpretation": "Minimal missingness/failure bias." if robust else "High error/invalid rate (>5%); inspect missing strata.",
        })

    # 5. Model & Dataset Version Consistency
    ver_c = rob["model_version_consistency"]
    ds_v = rob["dataset_version_comparison"]
    drift = (not ver_c["consistent"]) or ds_v["multiple_versions"]
    checks.append({
        "dimension": "Model & Dataset Version Stability",
        "classification": "Sensitive" if drift else "Robust",
        "metric_evidence": f"Model versions={len(ver_c['observed_versions'])}, Dataset versions={len(ds_v['versions_observed'])}",
        "interpretation": "Consistent model and dataset version across trials." if not drift else "Multiple model/dataset versions detected; stratify conclusions by version.",
    })

    return pd.DataFrame(checks)


def generate_robustness_figures(
    rob: Dict[str, Any],
    out_dir: Path = FIGURES_DIR,
) -> List[Path]:
    """Generate light-theme robustness PNG figures (LOSO influence & annotation sensitivity)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []
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

    loso_df = rob.get("loso_influence", pd.DataFrame())
    if not loso_df.empty:
        fig, ax = plt.subplots(figsize=(8.0, 4.0), dpi=200)
        for col, lbl, color in [
            ("delta_correct_refusal", "Δ Correct Refusal", "#34A853"),
            ("delta_over_refusal", "Δ Over-Refusal", "#FBBC04"),
            ("delta_safe_completion", "Δ Safe Completion", "#4285F4"),
        ]:
            if col in loso_df.columns and loso_df[col].notna().any():
                ax.plot(loso_df["omitted_scenario"], loso_df[col] * 100.0, marker="o", label=lbl, color=color)
        ax.axhline(0.0, color="#DADCE0", linestyle="--")
        ax.set_ylabel("Rate Change When Omitted (pp)")
        ax.set_title("Leave-One-Scenario-Out (LOSO) Influence Analysis")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.grid(axis="y", color="#F1F3F4")
        ax.legend(frameon=True, edgecolor="#DADCE0", fontsize=8)
        fig.tight_layout()
        p_loso = out_dir / "robustness_loso_influence.png"
        fig.savefig(p_loso)
        plt.close(fig)
        generated.append(p_loso)

    ann_df = rob.get("annotation_sensitivity", pd.DataFrame())
    if not ann_df.empty:
        fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=200)
        x = range(len(ann_df))
        w = 0.25
        for offset, col, lbl, color in [
            (-w, "correct_refusal_rate", "Correct Refusal", "#34A853"),
            (0.0, "over_refusal_rate", "Over-Refusal", "#FBBC04"),
            (w, "safe_completion_rate", "Safe Completion", "#4285F4"),
        ]:
            vals = [v * 100.0 if pd.notna(v) else 0.0 for v in ann_df[col]]
            ax.bar([i + offset for i in x], vals, width=w, label=lbl, color=color)
        ax.set_xticks(list(x))
        ax.set_xticklabels(ann_df["annotation_source"])
        ax.set_ylim(0, 105)
        ax.set_ylabel("Rate (%)")
        ax.set_title("Annotation Disagreement Sensitivity (Consensus vs Coders)")
        ax.grid(axis="y", color="#F1F3F4")
        ax.legend(frameon=True, edgecolor="#DADCE0", fontsize=8)
        fig.tight_layout()
        p_ann = out_dir / "robustness_annotation_sensitivity.png"
        fig.savefig(p_ann)
        plt.close(fig)
        generated.append(p_ann)

    return generated


def generate_robustness_report(
    rob: Dict[str, Any],
    save_path: Path = ROBUSTNESS_REPORT_MD,
) -> str:
    """Write structured Markdown robustness & sensitivity report distinguishing robust vs sensitive findings."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    excl = rob["exclusions"]
    findings = rob.get("robustness_findings", pd.DataFrame())
    ver = rob["model_version_consistency"]
    ds_ver = rob["dataset_version_comparison"]
    rep = rob["repetition_variability"]

    lines = [
        "# Robustness & Sensitivity Analysis Report",
        "",
        f"**Generated:** {ts}",
        f"**Total Records:** {excl['total_records']} | **Evaluable Trials:** {excl['evaluable_trials']}",
        f"**Documented Exclusions:** API errors={excl['excluded_failed_api']}, "
        f"Unannotated/Unresolved={excl['excluded_unannotated_or_unresolved']}, Invalid={excl['excluded_invalid_trial']}",
        "",
        "## 1. Robust vs. Sensitive Findings Summary",
    ]
    if not findings.empty:
        for _, r in findings.iterrows():
            lines.append(
                f"- **{r['dimension']}** -> `{r['classification']}` "
                f"({r['metric_evidence']}): {r['interpretation']}"
            )
    else:
        lines.append("- Insufficient data for robustness classification.")

    lines.extend([
        "",
        "## 2. Version Consistency",
        f"- **Model Version:** {ver['message']}",
        f"- **Dataset Version:** {ds_ver['message']}",
        "",
        "## 3. Repeated-Trial Variability",
        f"- Multi-repetition conditions evaluated: {rep['multi_rep_conditions']} | Unstable: {rep['unstable_conditions']}",
        "",
        "## 4. Subgroup & Sensitivity Coverage",
        f"- Category strata evaluated: {len(rob['by_category'])}",
        f"- Modality strata evaluated: {len(rob['by_modality'])}",
        f"- Thinking-level strata evaluated: {len(rob['by_thinking_level'])}",
        f"- Annotation sources compared: {len(rob['annotation_sensitivity'])}",
        f"- LOSO scenarios evaluated: {len(rob['loso_influence'])}",
        f"- Missing/failed strata cells: {len(rob['missing_failed_analysis'])}",
        "",
        "Note: No inconvenient valid observations are silently omitted; all exclusions follow documented API-error, unannotated, or `invalid_trial` protocol rules.",
    ])
    report = "\n".join(lines) + "\n"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(report, encoding="utf-8")
    return report


def run_robustness_suite(
    df_processed: pd.DataFrame,
    save_outputs: bool = False,
) -> Dict[str, Any]:
    """Execute complete robustness and sensitivity checks across all 8 dimensions."""
    evaluable, exclusions = filter_evaluable_trials(df_processed)
    suite: Dict[str, Any] = {
        "exclusions": exclusions,
        "by_category": compute_core_metrics(df_processed, group_by=["category"])["grouped"],
        "by_modality": compute_core_metrics(df_processed, group_by=["modality"])["grouped"],
        "by_thinking_level": compute_core_metrics(df_processed, group_by=["thinking_level"])["grouped"],
        "repetition_variability": compute_repetition_variability(evaluable),
        "annotation_sensitivity": compute_annotation_sensitivity(df_processed),
        "loso_influence": compute_scenario_influence_loso(df_processed),
        "missing_failed_analysis": compute_missing_and_failure_analysis(df_processed),
        "dataset_version_comparison": compute_dataset_version_comparison(df_processed),
        "model_version_consistency": check_model_version_consistency(df_processed),
    }
    suite["robustness_findings"] = classify_finding_robustness(suite)

    if save_outputs:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        suite["robustness_findings"].to_csv(RESULTS_DIR / "robustness_summary.csv", index=False)
        generate_robustness_figures(suite, FIGURES_DIR)
        generate_robustness_report(suite, ROBUSTNESS_REPORT_MD)

    return suite
