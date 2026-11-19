"""PAGE 6 — Dual-Annotator Response Annotation & Adjudication."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components import render_metric_card, render_page_header
from config import (
    ANNOTATION_LABELS,
    ANNOTATION_LABEL_DISPLAY,
    CATEGORY_LABELS,
    LEGITIMATE_CATEGORIES,
    MODALITY_LABELS,
)
from dataset.scenario_manager import load_scenarios, resolve_image_path
from evaluation.agreement import compute_cohens_kappa
from evaluation.annotation import (
    build_processed_results,
    list_disagreements,
    load_annotations,
    save_annotation,
)
from experiment.storage import load_trial_records


def render() -> None:
    render_page_header(
        "Response Annotation",
        "Independent Dual Human Coding, Inter-Annotator Agreement (Cohen's Kappa), and Adjudication",
    )

    records = load_trial_records(None)
    if not records:
        st.info(
            "No experimental responses recorded yet. Run the Pilot or Main experiment "
            "to populate trials for human annotation."
        )
        return

    from dataset.scenario_manager import load_active_dataset

    _, pilot_scens = load_active_dataset("pilot")
    _, main_scens = load_active_dataset("main")
    scenarios_by_id = {
        s["scenario_id"]: s for s in (load_scenarios() + pilot_scens + main_scens)
    }
    df_trials = pd.DataFrame(records)
    df_ann = load_annotations()
    kappa_res = compute_cohens_kappa(df_ann)
    disagreements = list_disagreements(df_ann)

    # Summary strip
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Total Recorded Trials", str(len(df_trials)))
    with m2:
        a1_cnt = int((df_ann["annotator_id"] == "annotator_1").sum()) if not df_ann.empty else 0
        a2_cnt = int((df_ann["annotator_id"] == "annotator_2").sum()) if not df_ann.empty else 0
        render_metric_card("Coder Entries", f"A1: {a1_cnt} | A2: {a2_cnt}")
    with m3:
        kappa_str = (
            f"{kappa_res['kappa']:.3f}" if kappa_res["kappa"] is not None else "Unavailable"
        )
        render_metric_card(
            "Cohen's Kappa",
            kappa_str,
            f"Paired: {kappa_res['paired_count']} | Agree: {kappa_res['percent_agreement'] * 100:.1f}%"
            if kappa_res["percent_agreement"] is not None
            else kappa_res["message"],
        )
    with m4:
        unresolved = (
            int((disagreements["adjudicator_label"].isna()).sum())
            if not disagreements.empty
            else 0
        )
        render_metric_card("Pending Adjudication", str(unresolved), "Disagreements between A1 & A2")

    tab_code, tab_adj, tab_kappa = st.tabs(
        ["Blind Human Coding", "Disagreement Adjudication", "Inter-Annotator Agreement"]
    )

    with tab_code:
        c_f1, c_f2, c_f3, c_f4 = st.columns(4)
        with c_f1:
            annotator_id = st.selectbox(
                "Active Human Annotator",
                options=["annotator_1", "annotator_2"],
                format_func=lambda x: "Annotator 1" if x == "annotator_1" else "Annotator 2",
            )
        with c_f2:
            exp_filter = st.selectbox(
                "Experiment Scope", options=["All", "pilot", "main"], index=0
            )
        with c_f3:
            only_unannotated = st.checkbox(
                "Only Unannotated by Active Coder", value=True
            )
        with c_f4:
            blind_mode = st.checkbox(
                "Blind Coding (Hide Expected Rubric)", value=True
            )

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
            st.success("All matching trials have been annotated by this coder.")
        else:
            trial_ids = filtered_trials["trial_id"].astype(str).tolist()
            if "ann_idx" not in st.session_state:
                st.session_state.ann_idx = 0
            st.session_state.ann_idx = min(st.session_state.ann_idx, len(trial_ids) - 1)

            nav1, nav2, nav3 = st.columns([1, 3, 1])
            with nav1:
                if st.button("Previous", disabled=st.session_state.ann_idx <= 0):
                    st.session_state.ann_idx -= 1
                    st.rerun()
            with nav2:
                selected_tid = st.selectbox(
                    f"Trial ({st.session_state.ann_idx + 1} of {len(trial_ids)})",
                    options=trial_ids,
                    index=st.session_state.ann_idx,
                )
                st.session_state.ann_idx = trial_ids.index(selected_tid)
            with nav3:
                if st.button("Next", disabled=st.session_state.ann_idx >= len(trial_ids) - 1):
                    st.session_state.ann_idx += 1
                    st.rerun()

            row = filtered_trials[
                filtered_trials["trial_id"].astype(str) == selected_tid
            ].iloc[0]
            _render_single_trial_coder(
                row, scenarios_by_id, df_ann, annotator_id, blind_mode=blind_mode
            )

    with tab_adj:
        if disagreements.empty:
            st.info("No coder disagreements detected.")
        else:
            pending_adj = disagreements[disagreements["adjudicator_label"].isna()]
            st.write(
                f"Total disagreements: **{len(disagreements)}** | Unresolved: **{len(pending_adj)}**"
            )
            st.dataframe(disagreements, use_container_width=True, hide_index=True)

            adj_tid = st.selectbox(
                "Select Disagreement Trial to Adjudicate",
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

    with tab_kappa:
        st.write(f"**Status:** `{kappa_res['status']}` — {kappa_res['message']}")
        if kappa_res["labels_present"]:
            st.markdown("**Confusion Matrix (Annotator 1 rows × Annotator 2 cols)**")
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

    c_meta, c_resp = st.columns([1, 1.4])
    with c_meta:
        st.markdown(f"**Scenario ID:** `{sid}` — {scen.get('scenario_title', '')}")
        if scen.get("scenario_description"):
            st.caption(str(scen.get("scenario_description")))
        st.markdown(f"**Category:** {CATEGORY_LABELS.get(cat, cat)}")
        st.markdown(f"**Modality:** {MODALITY_LABELS.get(str(row['modality']), row['modality'])}")
        st.markdown(f"**Thinking Level:** `{row['thinking_level']}` (Rep {row['repetition']})")
        st.markdown("**Original Input Prompt:**")
        st.info(str(row["prompt"]))

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
            st.markdown("**Expected Safe Behavior (Rubric Reference):**")
            st.caption(scen.get("expected_safe_behavior", "Not specified"))

    with c_resp:
        st.markdown("**Actual Gemini 4 Hydrogen Response:**")
        st.text_area(
            "Model Output",
            value=str(row.get("response_text") or "[Empty / Blocked Response]"),
            height=240,
            disabled=True,
            key=f"out_{annotator_id}_{trial_id}",
        )
        if pd.notna(row.get("block_reason")) and row.get("block_reason"):
            st.warning(f"API Block Reason: `{row.get('block_reason')}`")

        existing = df_ann[
            (df_ann["trial_id"].astype(str) == trial_id)
            & (df_ann["annotator_id"].astype(str) == annotator_id)
        ]
        prev_label = (
            str(existing.iloc[0]["annotation_label"]) if not existing.empty else ANNOTATION_LABELS[0]
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
            if (not existing.empty and "coder_pseudonym" in existing.columns and pd.notna(existing.iloc[0]["coder_pseudonym"]))
            else ""
        )

        with st.form(f"ann_form_{annotator_id}_{trial_id}"):
            coder_pseudonym = st.text_input(
                "Pseudonymous Annotator ID (Optional)",
                value=prev_pseudo,
                placeholder="e.g., R1_Anon",
            )
            selected_label = st.radio(
                "Response Classification",
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
                    "Usefulness Score (1 = Not Useful, 5 = Fully Useful; legitimate requests)",
                    min_value=1,
                    max_value=5,
                    value=prev_use,
                )
            else:
                st.caption("Usefulness score applies only to legitimate requests.")

            notes = st.text_area("Annotation Notes", value=prev_notes, height=70)
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
