"""Human response annotation storage, dual-annotator management, and adjudication."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from config import (
    ANNOTATION_LABELS,
    HUMAN_ANNOTATIONS_CSV,
    LEGITIMATE_CATEGORIES,
    PROCESSED_RESULTS_CSV,
    ensure_directories,
)
from experiment.storage import load_trial_records

ANNOTATION_COLUMNS = [
    "trial_id",
    "experiment_type",
    "scenario_id",
    "category",
    "modality",
    "thinking_level",
    "repetition",
    "annotator_id",
    "annotation_label",
    "usefulness_score",
    "notes",
    "timestamp_utc",
]


def load_annotations(path: Path = HUMAN_ANNOTATIONS_CSV) -> pd.DataFrame:
    """Load human annotations CSV or return empty typed DataFrame."""
    ensure_directories()
    if not path.exists():
        return pd.DataFrame(columns=ANNOTATION_COLUMNS)
    try:
        df = pd.read_csv(path)
        for col in ANNOTATION_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA
        return df[ANNOTATION_COLUMNS]
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=ANNOTATION_COLUMNS)


def save_annotation(
    trial_id: str,
    experiment_type: str,
    scenario_id: str,
    category: str,
    modality: str,
    thinking_level: str,
    repetition: int,
    annotator_id: str,
    annotation_label: str,
    usefulness_score: Optional[int] = None,
    notes: str = "",
    path: Path = HUMAN_ANNOTATIONS_CSV,
) -> pd.DataFrame:
    """Upsert one human annotation for (trial_id, annotator_id) without altering raw JSONL."""
    if annotation_label not in ANNOTATION_LABELS:
        raise ValueError(
            f"Invalid annotation_label '{annotation_label}'. Expected {ANNOTATION_LABELS}."
        )
    if annotator_id not in ("annotator_1", "annotator_2", "adjudicator"):
        raise ValueError(
            "annotator_id must be 'annotator_1', 'annotator_2', or 'adjudicator'."
        )

    if category in LEGITIMATE_CATEGORIES and annotation_label != "invalid_trial":
        if usefulness_score is not None and not (1 <= int(usefulness_score) <= 5):
            raise ValueError("usefulness_score must be between 1 and 5.")
    else:
        usefulness_score = None

    df = load_annotations(path)
    mask = (df["trial_id"].astype(str) == str(trial_id)) & (
        df["annotator_id"].astype(str) == str(annotator_id)
    )
    df = df.loc[~mask].copy()

    new_row = {
        "trial_id": str(trial_id),
        "experiment_type": str(experiment_type),
        "scenario_id": str(scenario_id),
        "category": str(category),
        "modality": str(modality),
        "thinking_level": str(thinking_level).upper(),
        "repetition": int(repetition),
        "annotator_id": str(annotator_id),
        "annotation_label": str(annotation_label),
        "usefulness_score": int(usefulness_score) if usefulness_score is not None else pd.NA,
        "notes": str(notes).strip(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    new_df = pd.DataFrame([new_row])
    df = new_df if df.empty else pd.concat([df, new_df], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def resolve_final_annotations(df_annotations: pd.DataFrame) -> pd.DataFrame:
    """Resolve dual human annotations and adjudications into one record per trial_id."""
    cols = [
        "trial_id",
        "resolved_label",
        "resolved_usefulness_score",
        "resolution_source",
        "annotator_1_label",
        "annotator_2_label",
        "adjudicator_label",
        "has_disagreement",
    ]
    if df_annotations.empty:
        return pd.DataFrame(columns=cols)

    rows: List[Dict[str, Any]] = []
    for trial_id, grp in df_annotations.groupby("trial_id", sort=False):
        by_ann = {
            str(r["annotator_id"]): r for _, r in grp.iterrows()
        }
        a1 = by_ann.get("annotator_1")
        a2 = by_ann.get("annotator_2")
        adj = by_ann.get("adjudicator")

        l1 = str(a1["annotation_label"]) if a1 is not None else None
        l2 = str(a2["annotation_label"]) if a2 is not None else None
        ladj = str(adj["annotation_label"]) if adj is not None else None

        disagreement = l1 is not None and l2 is not None and l1 != l2

        def _num(row: Any) -> Optional[float]:
            if row is None:
                return None
            val = row["usefulness_score"]
            if pd.isna(val):
                return None
            try:
                return float(val)
            except (TypeError, ValueError):
                return None

        u1, u2, uadj = _num(a1), _num(a2), _num(adj)

        if ladj is not None:
            resolved_label = ladj
            resolved_use = uadj if uadj is not None else (
                (u1 + u2) / 2.0 if (u1 is not None and u2 is not None) else (u1 if u1 is not None else u2)
            )
            source = "adjudicated"
        elif l1 is not None and l2 is not None:
            if l1 == l2:
                resolved_label = l1
                if u1 is not None and u2 is not None:
                    resolved_use = (u1 + u2) / 2.0
                else:
                    resolved_use = u1 if u1 is not None else u2
                source = "consensus"
            else:
                resolved_label = None
                resolved_use = None
                source = "disagreement_unresolved"
        elif l1 is not None:
            resolved_label = l1
            resolved_use = u1
            source = "annotator_1_only"
        elif l2 is not None:
            resolved_label = l2
            resolved_use = u2
            source = "annotator_2_only"
        else:
            resolved_label = None
            resolved_use = None
            source = "unannotated"

        rows.append(
            {
                "trial_id": str(trial_id),
                "resolved_label": resolved_label,
                "resolved_usefulness_score": resolved_use,
                "resolution_source": source,
                "annotator_1_label": l1,
                "annotator_2_label": l2,
                "adjudicator_label": ladj,
                "has_disagreement": disagreement,
            }
        )
    return pd.DataFrame(rows, columns=cols)


def list_disagreements(df_annotations: pd.DataFrame) -> pd.DataFrame:
    """Return trials where annotator_1 and annotator_2 disagree and need adjudication."""
    resolved = resolve_final_annotations(df_annotations)
    if resolved.empty:
        return resolved
    return resolved[resolved["has_disagreement"]].copy()


def build_processed_results(
    experiment_type: Optional[str] = None,
    save_csv: bool = True,
    annotations_path: Path = HUMAN_ANNOTATIONS_CSV,
    output_csv: Path = PROCESSED_RESULTS_CSV,
) -> pd.DataFrame:
    """Combine raw trial observations with resolved human annotations cleanly."""
    records = load_trial_records(experiment_type)
    if not records:
        df_empty = pd.DataFrame()
        if save_csv:
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            df_empty.to_csv(output_csv, index=False)
        return df_empty

    df_raw = pd.DataFrame(records)
    # Keep latest record per unique trial_id if ever duplicated, though UUID prevents it
    df_ann = load_annotations(annotations_path)
    df_res = resolve_final_annotations(df_ann)

    if df_res.empty:
        df_merged = df_raw.copy()
        df_merged["resolved_label"] = pd.NA
        df_merged["resolved_usefulness_score"] = pd.NA
        df_merged["resolution_source"] = "unannotated"
        df_merged["annotator_1_label"] = pd.NA
        df_merged["annotator_2_label"] = pd.NA
        df_merged["adjudicator_label"] = pd.NA
        df_merged["has_disagreement"] = False
    else:
        df_merged = df_raw.merge(df_res, on="trial_id", how="left")
        df_merged["resolution_source"] = df_merged["resolution_source"].fillna(
            "unannotated"
        )
        df_merged["has_disagreement"] = (
            df_merged["has_disagreement"].fillna(False).astype(bool)
        )

    if save_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df_merged.to_csv(output_csv, index=False)
    return df_merged
