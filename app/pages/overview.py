"""Overview — Minimal academic dashboard explaining the study, status, and next step."""

from __future__ import annotations

import html
from typing import Tuple

import streamlit as st

from app.components import (
    navigate_to,
    render_metric_card,
    render_page_header,
    render_study_pillar,
)
from config import (
    DEFAULT_MODEL_ID,
    MAIN_TARGET_CALLS,
    PILOT_REPORT_MD,
    PILOT_TARGET_CALLS,
    RESEARCH_OBJECTIVE,
    RESEARCH_TITLE,
    STATISTICAL_RESULTS_CSV,
    THINKING_LEVELS,
)
from dataset.scenario_manager import load_active_dataset
from evaluation.annotation import load_annotations, resolve_final_annotations
from experiment.hydrogen_client import (
    get_api_key,
    is_api_fully_validated,
    load_validation_state,
)
from experiment.manifest import verify_main_experiment_readiness
from experiment.storage import load_experiment_state, load_trial_records


def _compute_next_step(
    api_key_present: bool,
    api_validated: bool,
    pilot_approved: bool,
    main_approved: bool,
    pilot_completed: int,
    pilot_annotated: int,
    main_completed: int,
    total_completed: int,
    annotated_count: int,
) -> Tuple[str, str, str]:
    """Return (next_step_message, target_page, button_label)."""
    if not api_key_present or not api_validated:
        return (
            "Your next step is to configure your Gemini API key and verify the four thinking levels in Settings.",
            "Settings",
            "Open Settings",
        )
    if not pilot_approved:
        return (
            "Your next step is to review and approve the 12-scenario Pilot Dataset in Research Runs.",
            "Research Runs",
            "Open Research Runs",
        )
    if pilot_completed < PILOT_TARGET_CALLS:
        return (
            "Your next step is to run the 96-call Pilot Experiment in Research Runs.",
            "Research Runs",
            "Open Research Runs",
        )
    if pilot_annotated < pilot_completed:
        return (
            "Your next step is to complete human annotation and review of the pilot responses in Results & Analysis.",
            "Results & Analysis",
            "Open Results & Analysis",
        )
    if not main_approved:
        return (
            "Your next step is to review and approve the 60-scenario Main Dataset in Research Runs.",
            "Research Runs",
            "Open Research Runs",
        )
    if main_completed < MAIN_TARGET_CALLS:
        return (
            "Your next step is to authorize and run the 1,440-call Main Experiment in Research Runs.",
            "Research Runs",
            "Open Research Runs",
        )
    if annotated_count < total_completed:
        return (
            "Your next step is to annotate the remaining experimental responses in Results & Analysis.",
            "Results & Analysis",
            "Open Results & Analysis",
        )
    return (
        "All experimental runs and human annotations are complete. Review your findings and export results in Results & Analysis.",
        "Results & Analysis",
        "View Final Results",
    )


def render() -> None:
    render_page_header(
        "Overview",
        f"{RESEARCH_TITLE}. {RESEARCH_OBJECTIVE}",
    )

    api_key_present = bool(get_api_key())
    api_validated = is_api_fully_validated()
    val_state = load_validation_state()
    val_levels = val_state.get("levels", {})
    validated_lvl_count = sum(1 for v in val_levels.values() if v.get("validated"))

    pilot_entry, _ = load_active_dataset("pilot")
    main_entry, _ = load_active_dataset("main")
    pilot_approval = pilot_entry.get("researcher_approval_status") if pilot_entry else "none"
    main_approval = main_entry.get("researcher_approval_status") if main_entry else "none"
    pilot_approved = pilot_approval == "approved"
    main_approved = main_approval == "approved"

    pilot_records = load_trial_records("pilot")
    main_records = load_trial_records("main")
    pilot_completed = sum(1 for r in pilot_records if r.get("status") in ("success", "blocked"))
    main_completed = sum(1 for r in main_records if r.get("status") in ("success", "blocked"))
    total_completed = pilot_completed + main_completed

    df_ann = load_annotations()
    df_resolved = resolve_final_annotations(df_ann)
    annotated_count = (
        int(df_resolved["resolved_label"].notna().sum()) if not df_resolved.empty else 0
    )

    pilot_tids = {str(r.get("trial_id")) for r in pilot_records if r.get("status") in ("success", "blocked")}
    pilot_annotated = (
        int(
            df_resolved[
                df_resolved["trial_id"].astype(str).isin(pilot_tids)
                & df_resolved["resolved_label"].notna()
            ].shape[0]
        )
        if (not df_resolved.empty and pilot_tids)
        else 0
    )

    pilot_state = load_experiment_state("pilot")
    main_state = load_experiment_state("main")

    # 1. Four Core Statistics
    if api_validated:
        api_stat = f"Connected ({validated_lvl_count}/4)"
        api_sub = f"{DEFAULT_MODEL_ID} verified"
    elif api_key_present:
        api_stat = "Not verified"
        api_sub = f"Key present · {validated_lvl_count}/4 levels verified"
    else:
        api_stat = "Not connected"
        api_sub = "API key not configured"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("API Status", api_stat, api_sub)
    with c2:
        render_metric_card(
            "Pilot Progress",
            f"{pilot_completed}/{PILOT_TARGET_CALLS}",
            f"Dataset: {'Approved' if pilot_approved else 'Pending approval'}",
        )
    with c3:
        render_metric_card(
            "Main Experiment Progress",
            f"{main_completed:,}/{MAIN_TARGET_CALLS:,}",
            f"Dataset: {'Approved' if main_approved else 'Pending approval'}",
        )
    with c4:
        render_metric_card(
            "Annotated Responses",
            f"{annotated_count:,}",
            f"Of {total_completed:,} completed responses" if total_completed > 0 else "No responses recorded yet",
        )

    # 2. Next Step Callout with Action Shortcut
    next_step, next_page, next_btn_label = _compute_next_step(
        api_key_present=api_key_present,
        api_validated=api_validated,
        pilot_approved=pilot_approved,
        main_approved=main_approved,
        pilot_completed=pilot_completed,
        pilot_annotated=pilot_annotated,
        main_completed=main_completed,
        total_completed=total_completed,
        annotated_count=annotated_count,
    )
    ns_left, ns_right = st.columns([4, 1.2], vertical_alignment="center")
    with ns_left:
        st.markdown(
            f"""
            <div class="next-step-panel">
                <div class="next-step-kicker">Next Step</div>
                <div class="next-step-body">{html.escape(next_step)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with ns_right:
        if st.button(next_btn_label, type="primary", use_container_width=True, key="ov_next_btn"):
            navigate_to(next_page)

    st.divider()

    # 3. Study Design Summary (Explains the Project Wisely)
    st.subheader("Study Architecture")
    p1, p2, p3 = st.columns(3)
    with p1:
        render_study_pillar(
            "60 Matched Scenario Families",
            "Balanced across Clearly Benign (20), Sensitive-Looking Legitimate (20), and Controlled Unsafe (20) cases. Each family pairs a Text-Only prompt with a semantically matched Image + Text prompt.",
        )
    with p2:
        render_study_pillar(
            "4 Thinking Levels & Two-Stage Runs",
            "Evaluates Gemini 4 Hydrogen at LOW, MEDIUM, HIGH, and XHIGH reasoning levels. Begins with a 96-call Pilot (12 scenarios × 1 repetition) before unlocking the 1,440-call Main study (60 scenarios × 3 repetitions).",
        )
    with p3:
        render_study_pillar(
            "Human Coding & 11 Metrics",
            "Blind dual human annotation with Cohen's kappa and adjudication measures refusal appropriateness, over-refusal, unsafe compliance, safe completion, cross-modal disagreement, latency, tokens, and usefulness.",
        )

    st.divider()

    # 4. Research Progress
    st.subheader("Research Progress")

    if api_validated:
        api_step_status = "Completed"
        api_step_detail = "All four thinking levels (LOW, MEDIUM, HIGH, XHIGH) passed endpoint checks."
    elif validated_lvl_count > 0 or api_key_present:
        api_step_status = "In progress" if validated_lvl_count > 0 else "Not started"
        api_step_detail = f"{validated_lvl_count} of {len(THINKING_LEVELS)} thinking levels verified."
    else:
        api_step_status = "Not started"
        api_step_detail = "Configure GEMINI_API_KEY and verify endpoint support in Settings."

    if pilot_approved and main_approved:
        ds_status = "Completed"
        ds_detail = "Both pilot (12 scenarios) and main (60 scenarios) versions are approved."
    elif pilot_approval == "rejected" or main_approval == "rejected":
        ds_status = "Blocked"
        ds_detail = "A dataset version was rejected and requires replacement."
    elif pilot_entry or main_entry:
        ds_status = "Pending approval"
        ds_detail = (
            f"Pilot ({pilot_entry.get('dataset_version') if pilot_entry else 'None'}): {pilot_approval.replace('_', ' ')} · "
            f"Main ({main_entry.get('dataset_version') if main_entry else 'None'}): {main_approval.replace('_', ' ')}"
        )
    else:
        ds_status = "Not started"
        ds_detail = "Import pilot and main scenario datasets in Research Runs."

    if pilot_completed >= PILOT_TARGET_CALLS:
        pilot_status = "Completed"
        pilot_detail = f"{pilot_completed} of {PILOT_TARGET_CALLS} calls completed."
    elif pilot_completed > 0 or pilot_state.get("status") in ("RUNNING", "PAUSED"):
        pilot_status = "In progress"
        pilot_detail = f"{pilot_completed} of {PILOT_TARGET_CALLS} calls completed."
    elif not api_validated or not pilot_approved:
        pilot_status = "Blocked"
        pilot_detail = "Requires API verification and pilot dataset approval."
    else:
        pilot_status = "Not started"
        pilot_detail = "Ready to run 96 pilot calls in Research Runs."

    if pilot_completed >= PILOT_TARGET_CALLS and pilot_annotated >= pilot_completed and PILOT_REPORT_MD.exists():
        pilot_rev_status = "Completed"
        pilot_rev_detail = f"{pilot_annotated} of {pilot_completed} pilot responses annotated."
    elif pilot_annotated > 0:
        pilot_rev_status = "In progress"
        pilot_rev_detail = f"{pilot_annotated} of {pilot_completed} pilot responses annotated."
    elif pilot_completed == 0:
        pilot_rev_status = "Blocked"
        pilot_rev_detail = "Waiting for pilot experiment responses."
    else:
        pilot_rev_status = "Not started"
        pilot_rev_detail = f"{pilot_completed} pilot responses ready for annotation."

    main_readiness = verify_main_experiment_readiness(explicit_approved=False)
    if main_completed >= MAIN_TARGET_CALLS:
        main_status = "Completed"
        main_detail = f"{main_completed:,} of {MAIN_TARGET_CALLS:,} calls completed."
    elif main_completed > 0 or main_state.get("status") in ("RUNNING", "PAUSED"):
        main_status = "In progress"
        main_detail = f"{main_completed:,} of {MAIN_TARGET_CALLS:,} calls completed."
    elif not all(
        c["passed"]
        for c in main_readiness["checks"]
        if c["check"] != "Explicit Researcher Execution Approval"
    ):
        main_status = "Blocked"
        main_detail = "Requires completed pilot, API verification, and main dataset approval."
    else:
        main_status = "Pending approval"
        main_detail = "Prerequisites met. Requires explicit researcher confirmation in Research Runs."

    stats_ready = STATISTICAL_RESULTS_CSV.exists() and STATISTICAL_RESULTS_CSV.stat().st_size > 10
    if main_completed >= MAIN_TARGET_CALLS and annotated_count >= main_completed and stats_ready:
        analysis_status = "Completed"
        analysis_detail = "Annotated dataset and statistical outputs are generated."
    elif annotated_count > 0:
        analysis_status = "In progress"
        analysis_detail = f"{annotated_count:,} annotated responses available for analysis."
    else:
        analysis_status = "Blocked"
        analysis_detail = "Requires annotated experimental responses."

    steps = [
        ("API verification", api_step_status, api_step_detail),
        ("Dataset approval", ds_status, ds_detail),
        ("Pilot experiment", pilot_status, pilot_detail),
        ("Pilot annotation and review", pilot_rev_status, pilot_rev_detail),
        ("Main experiment", main_status, main_detail),
        ("Final analysis", analysis_status, analysis_detail),
    ]

    rows_html = "".join(
        f"""
        <div class="progress-row">
            <div>
                <div class="progress-step-name">{html.escape(name)}</div>
                <div class="progress-step-detail">{html.escape(detail)}</div>
            </div>
            <div class="progress-step-status">{html.escape(status)}</div>
        </div>
        """
        for name, status, detail in steps
    )
    st.markdown(rows_html, unsafe_allow_html=True)
