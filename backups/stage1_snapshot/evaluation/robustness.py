"""Robustness, repetition stability, sensitivity, LOSO influence, and missingness checks."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

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


def run_robustness_suite(df_processed: pd.DataFrame) -> Dict[str, Any]:
    """Execute complete robustness and sensitivity checks."""
    evaluable, exclusions = filter_evaluable_trials(df_processed)
    return {
        "exclusions": exclusions,
        "by_category": compute_core_metrics(df_processed, group_by=["category"])["grouped"],
        "by_modality": compute_core_metrics(df_processed, group_by=["modality"])["grouped"],
        "by_thinking_level": compute_core_metrics(df_processed, group_by=["thinking_level"])["grouped"],
        "repetition_variability": compute_repetition_variability(evaluable),
        "annotation_sensitivity": compute_annotation_sensitivity(df_processed),
        "loso_influence": compute_scenario_influence_loso(df_processed),
        "missing_failed_analysis": compute_missing_and_failure_analysis(df_processed),
        "model_version_consistency": check_model_version_consistency(df_processed),
    }
