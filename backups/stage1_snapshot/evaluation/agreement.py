"""Inter-annotator agreement calculations (Cohen's kappa and disagreement breakdown)."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from config import ANNOTATION_LABELS


def compute_cohens_kappa(df_annotations: pd.DataFrame) -> Dict[str, Any]:
    """Compute Cohen's kappa between annotator_1 and annotator_2 on doubly annotated trials."""
    if df_annotations.empty:
        return {
            "paired_count": 0,
            "kappa": None,
            "percent_agreement": None,
            "disagreement_count": 0,
            "status": "insufficient_data",
            "message": "No annotations recorded yet.",
            "labels_present": [],
            "confusion_matrix": [],
            "by_category": [],
            "by_modality": [],
            "by_thinking_level": [],
        }

    a1 = df_annotations[df_annotations["annotator_id"] == "annotator_1"][
        ["trial_id", "annotation_label", "category", "modality", "thinking_level"]
    ].rename(columns={"annotation_label": "label_1"})

    a2 = df_annotations[df_annotations["annotator_id"] == "annotator_2"][
        ["trial_id", "annotation_label"]
    ].rename(columns={"annotation_label": "label_2"})

    paired = a1.merge(a2, on="trial_id", how="inner")
    n = len(paired)
    if n < 2:
        return {
            "paired_count": n,
            "kappa": None,
            "percent_agreement": float((paired["label_1"] == paired["label_2"]).mean())
            if n == 1
            else None,
            "disagreement_count": int((paired["label_1"] != paired["label_2"]).sum())
            if n == 1
            else 0,
            "status": "insufficient_data",
            "message": "At least 2 trials annotated by both Annotator 1 and Annotator 2 are required.",
            "labels_present": [],
            "confusion_matrix": [],
            "by_category": [],
            "by_modality": [],
            "by_thinking_level": [],
        }

    agree = (paired["label_1"] == paired["label_2"]).sum()
    pct_agree = float(agree / n)
    disagreements = int(n - agree)

    labels_present = [
        lbl
        for lbl in ANNOTATION_LABELS
        if (paired["label_1"] == lbl).any() or (paired["label_2"] == lbl).any()
    ]

    if len(labels_present) <= 1:
        kappa_val = None
        status = "undefined_single_class"
        message = (
            "Cohen's kappa is undefined when all paired annotations fall into a single class."
        )
    else:
        raw_kappa = cohen_kappa_score(
            paired["label_1"], paired["label_2"], labels=labels_present
        )
        kappa_val = None if pd.isna(raw_kappa) else float(raw_kappa)
        status = "computed"
        message = "Computed on doubly annotated trials."

    cm = confusion_matrix(
        paired["label_1"], paired["label_2"], labels=labels_present
    ).tolist()

    def _subset_kappa(group_col: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for val, sub in paired.groupby(group_col):
            sub_n = len(sub)
            sub_agree = float((sub["label_1"] == sub["label_2"]).mean())
            sub_labels = sorted(set(sub["label_1"]).union(set(sub["label_2"])))
            if sub_n >= 2 and len(sub_labels) > 1:
                k = cohen_kappa_score(sub["label_1"], sub["label_2"], labels=sub_labels)
                sub_k = None if pd.isna(k) else float(k)
            else:
                sub_k = None
            out.append(
                {
                    group_col: val,
                    "paired_count": sub_n,
                    "percent_agreement": sub_agree,
                    "kappa": sub_k,
                }
            )
        return out

    return {
        "paired_count": n,
        "kappa": kappa_val,
        "percent_agreement": pct_agree,
        "disagreement_count": disagreements,
        "status": status,
        "message": message,
        "labels_present": labels_present,
        "confusion_matrix": cm,
        "by_category": _subset_kappa("category"),
        "by_modality": _subset_kappa("modality"),
        "by_thinking_level": _subset_kappa("thinking_level"),
    }
