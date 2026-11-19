"""Tests for human annotation, Cohen's kappa, explicit metric definitions, and statistical analysis."""

from __future__ import annotations

import pandas as pd
import pytest

from evaluation.agreement import compute_cohens_kappa
from evaluation.annotation import (
    build_processed_results,
    load_annotations,
    resolve_final_annotations,
    save_annotation,
)
from evaluation.metrics import compute_core_metrics
from evaluation.robustness import run_robustness_suite
from evaluation.statistical_analysis import run_full_statistical_analysis


def test_annotation_adjudication_and_kappa(tmp_path) -> None:
    ann_csv = tmp_path / "annotations.csv"

    # Trial 1: both agree safe_completion
    save_annotation("t1", "pilot", "SCN-001", "clearly_benign", "text_only", "LOW", 1, "annotator_1", "safe_completion", 5, "ok", path=ann_csv)
    save_annotation("t1", "pilot", "SCN-001", "clearly_benign", "text_only", "LOW", 1, "annotator_2", "safe_completion", 4, "ok", path=ann_csv)

    # Trial 2: disagreement -> adjudicated
    save_annotation("t2", "pilot", "SCN-021", "sensitive_looking_legitimate", "text_only", "HIGH", 1, "annotator_1", "over_refusal", 1, "refused", path=ann_csv)
    save_annotation("t2", "pilot", "SCN-021", "sensitive_looking_legitimate", "text_only", "HIGH", 1, "annotator_2", "partial_safe_completion", 3, "partial", path=ann_csv)

    # Trial 3: both agree appropriate_refusal
    save_annotation("t3", "pilot", "SCN-041", "genuinely_unsafe", "text_only", "LOW", 1, "annotator_1", "appropriate_refusal", None, "good", path=ann_csv)
    save_annotation("t3", "pilot", "SCN-041", "genuinely_unsafe", "text_only", "LOW", 1, "annotator_2", "appropriate_refusal", None, "good", path=ann_csv)

    df_ann = load_annotations(ann_csv)
    kappa = compute_cohens_kappa(df_ann)
    assert kappa["paired_count"] == 3
    assert kappa["disagreement_count"] == 1
    assert kappa["kappa"] is not None

    resolved = resolve_final_annotations(df_ann)
    t2_row = resolved[resolved["trial_id"] == "t2"].iloc[0]
    assert t2_row["resolution_source"] == "disagreement_unresolved"
    assert pd.isna(t2_row["resolved_label"])

    # Adjudicate t2
    save_annotation("t2", "pilot", "SCN-021", "sensitive_looking_legitimate", "text_only", "HIGH", 1, "adjudicator", "over_refusal", 1, "confirmed over-refusal", path=ann_csv)
    resolved2 = resolve_final_annotations(load_annotations(ann_csv))
    t2_row2 = resolved2[resolved2["trial_id"] == "t2"].iloc[0]
    assert t2_row2["resolved_label"] == "over_refusal"
    assert t2_row2["resolution_source"] == "adjudicated"


def test_metric_denominators_and_missing_tokens() -> None:
    df = pd.DataFrame(
        [
            {
                "trial_id": "1",
                "experiment_type": "pilot",
                "scenario_id": "SCN-001",
                "category": "clearly_benign",
                "modality": "text_only",
                "thinking_level": "LOW",
                "repetition": 1,
                "status": "success",
                "latency_ms": 100.0,
                "input_tokens": 10,
                "output_tokens": 20,
                "thinking_tokens": None,  # unavailable
                "resolved_label": "safe_completion",
                "resolved_usefulness_score": 5.0,
                "actual_model_version": "gemini-4-hydrogen",
            },
            {
                "trial_id": "2",
                "experiment_type": "pilot",
                "scenario_id": "SCN-001",
                "category": "clearly_benign",
                "modality": "image_plus_text",
                "thinking_level": "LOW",
                "repetition": 1,
                "status": "success",
                "latency_ms": 150.0,
                "input_tokens": 20,
                "output_tokens": 30,
                "thinking_tokens": None,
                "resolved_label": "over_refusal",
                "resolved_usefulness_score": 1.0,
                "actual_model_version": "gemini-4-hydrogen",
            },
            {
                "trial_id": "3",
                "experiment_type": "pilot",
                "scenario_id": "SCN-041",
                "category": "genuinely_unsafe",
                "modality": "text_only",
                "thinking_level": "LOW",
                "repetition": 1,
                "status": "success",
                "latency_ms": 120.0,
                "input_tokens": 15,
                "output_tokens": 25,
                "thinking_tokens": None,
                "resolved_label": "appropriate_refusal",
                "resolved_usefulness_score": None,
                "actual_model_version": "gemini-4-hydrogen",
            },
            {
                "trial_id": "4",
                "experiment_type": "pilot",
                "scenario_id": "SCN-041",
                "category": "genuinely_unsafe",
                "modality": "image_plus_text",
                "thinking_level": "LOW",
                "repetition": 1,
                "status": "error",
                "latency_ms": 0.0,
                "input_tokens": None,
                "output_tokens": None,
                "thinking_tokens": None,
                "resolved_label": None,
                "resolved_usefulness_score": None,
                "actual_model_version": None,
            },
        ]
    )

    res = compute_core_metrics(df, group_by=["thinking_level"])
    ov = res["overall"]
    excl = res["exclusions"]

    assert excl["excluded_failed_api"] == 1
    assert excl["evaluable_trials"] == 3
    assert ov["correct_refusal_rate"] == pytest.approx(1.0)
    assert ov["over_refusal_rate"] == pytest.approx(0.5)
    assert ov["safe_completion_rate"] == pytest.approx(0.5)
    assert ov["cross_modal_matched_pairs"] == 1
    assert ov["cross_modal_disagreement_rate"] == pytest.approx(1.0)
    # Unavailable thinking_tokens MUST remain None, never estimated as 0
    assert ov["thinking_tokens_mean"] is None

    stats_res = run_full_statistical_analysis(df, save_csv=False)
    assert "mcnemar_tests" in stats_res
    rob = run_robustness_suite(df)
    assert rob["model_version_consistency"]["consistent"] is True
