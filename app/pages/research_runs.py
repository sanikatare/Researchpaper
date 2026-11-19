"""Research Runs — Unified Dataset, Pilot Experiment, and Main Experiment workflow."""

from __future__ import annotations

import json
import time
from typing import Optional

import pandas as pd
import streamlit as st

from app.components import render_metric_card, render_page_header
from config import (
    AISTUDIO_TEMPLATE_PATH,
    CATEGORY_LABELS,
    MAIN_TARGET_CALLS,
    MAIN_TARGET_REPETITIONS,
    MODALITIES,
    PILOT_MANIFEST_JSON,
    PILOT_TARGET_CALLS,
    PILOT_TARGET_REPETITIONS,
    SCENARIO_SCHEMA_PATH,
    THINKING_LEVELS,
)
from dataset.scenario_manager import (
    check_image_valid,
    get_version_entry,
    import_dataset_version,
    load_active_dataset,
    load_registry,
    load_review_log,
    load_scenarios_for_version,
    record_scenario_review,
    resolve_image_path,
    set_active_version,
    set_version_approval,
    validate_dataset,
    verify_version_integrity,
)
from evaluation.annotation import build_processed_results
from evaluation.metrics import generate_pilot_report
from experiment.experiment_controller import get_controller
from experiment.experiment_runner import build_trial_plan, filter_pending_trials
from experiment.hydrogen_client import is_api_fully_validated
from experiment.manifest import (
    build_experiment_manifest,
    generate_main_post_run_reports,
    verify_main_experiment_readiness,
)
from experiment.storage import (
    get_config_snapshot_path,
    load_experiment_state,
    load_runtime_config,
    load_trial_records,
)


def _format_approval(status: Optional[str]) -> str:
    return {
        "approved": "Approved",
        "pending_review": "Pending approval",
        "rejected": "Rejected",
    }.get(status or "", "Not set")


def _render_dataset_section() -> None:
    st.subheader("A. Dataset")
    st.caption(
        "Scenario datasets are authored in Google AI Studio and stored as immutable, SHA-256 verified versions."
    )

    registry = load_registry()
    versions = registry.get("versions", [])
    pilot_entry, pilot_scenarios = load_active_dataset("pilot")
    main_entry, main_scenarios = load_active_dataset("main")

    pilot_integrity = (
        verify_version_integrity(pilot_entry["dataset_version"])
        if pilot_entry
        else {"intact": False}
    )
    main_integrity = (
        verify_version_integrity(main_entry["dataset_version"])
        if main_entry
        else {"intact": False}
    )

    def _cat_summary(scens: list) -> str:
        b = sum(1 for s in scens if s.get("category") == "clearly_benign")
        sl = sum(1 for s in scens if s.get("category") == "sensitive_looking_legitimate")
        u = sum(1 for s in scens if s.get("category") == "genuinely_unsafe")
        return f"Benign {b} / Sensitive {sl} / Unsafe {u}"

    col_p, col_m = st.columns(2)
    with col_p:
        p_ver = pilot_entry["dataset_version"] if pilot_entry else "None"
        p_app = _format_approval(
            pilot_entry.get("researcher_approval_status") if pilot_entry else None
        )
        p_sha = "Verified (SHA-256 intact)" if pilot_integrity.get("intact") else "Mismatch / Missing"
        render_metric_card(
            f"Pilot Dataset ({p_ver})",
            f"{len(pilot_scenarios)} scenarios",
            f"{_cat_summary(pilot_scenarios)} · Approval: {p_app} · {p_sha}",
        )
    with col_m:
        m_ver = main_entry["dataset_version"] if main_entry else "None"
        m_app = _format_approval(
            main_entry.get("researcher_approval_status") if main_entry else None
        )
        m_sha = "Verified (SHA-256 intact)" if main_integrity.get("intact") else "Mismatch / Missing"
        render_metric_card(
            f"Main Dataset ({m_ver})",
            f"{len(main_scenarios)} scenarios",
            f"{_cat_summary(main_scenarios)} · Approval: {m_app} · {m_sha}",
        )

    if "rr_review_scope" not in st.session_state:
        st.session_state.rr_review_scope = "pilot"

    if versions:
        st.markdown("**Dataset Approvals Queue**")
        queue_rows = []
        active_p = registry.get("active_pilot_version")
        active_m = registry.get("active_main_version")
        for v in versions:
            vid = v["dataset_version"]
            is_act = vid == active_p or vid == active_m
            queue_rows.append(
                {
                    "Version": vid,
                    "Dataset Name": v.get("dataset_name", ""),
                    "Scope": str(v.get("dataset_scope", "")).title(),
                    "Scenarios": v.get("scenario_count", 0),
                    "Validation": str(v.get("technical_validation_status", "")).upper(),
                    "Approval": _format_approval(v.get("researcher_approval_status")),
                    "Active": "Active" if is_act else "—",
                    "Source": v.get("dataset_source", ""),
                }
            )
        st.dataframe(pd.DataFrame(queue_rows), use_container_width=True, hide_index=True)

    b_p, b_m, b_sel = st.columns([1.0, 1.0, 2.2], vertical_alignment="bottom")
    with b_p:
        if st.button(
            "Review Pilot Dataset",
            type="primary" if st.session_state.rr_review_scope == "pilot" else "secondary",
            use_container_width=True,
        ):
            st.session_state.rr_review_scope = "pilot"
            st.rerun()
    with b_m:
        if st.button(
            "Review Main Dataset",
            type="primary" if st.session_state.rr_review_scope == "main" else "secondary",
            use_container_width=True,
        ):
            st.session_state.rr_review_scope = "main"
            st.rerun()

    scope = st.session_state.rr_review_scope
    scope_versions = [v for v in versions if v.get("dataset_scope") == scope]
    active_entry = pilot_entry if scope == "pilot" else main_entry
    active_ver_id = active_entry["dataset_version"] if active_entry else None

    if not scope_versions:
        st.info(f"No {scope} dataset versions are registered.")
        return

    scope_ver_ids = [v["dataset_version"] for v in scope_versions]
    default_idx = scope_ver_ids.index(active_ver_id) if active_ver_id in scope_ver_ids else 0
    with b_sel:
        selected_ver_id = st.selectbox(
            f"Select {scope.title()} version for inspection & approval",
            options=scope_ver_ids,
            index=default_idx,
            format_func=lambda vid: next(
                (
                    f"{v['dataset_version']} — {v.get('dataset_name', '')} ({_format_approval(v.get('researcher_approval_status'))})"
                    for v in scope_versions
                    if v["dataset_version"] == vid
                ),
                vid,
            ),
            key=f"rr_inspect_ver_{scope}",
        )

    selected_entry = get_version_entry(selected_ver_id)
    if selected_entry is None:
        st.info(f"Selected {scope} dataset version was not found.")
        return

    ver_id = selected_entry["dataset_version"]
    scenarios = load_scenarios_for_version(ver_id)
    integrity = verify_version_integrity(ver_id)
    review_log = load_review_log(ver_id)
    scenario_reviews = review_log.get("scenario_reviews", {})
    rejected_count = sum(
        1 for r in scenario_reviews.values() if r.get("decision") == "rejected"
    )
    approved_scen_count = sum(
        1 for r in scenario_reviews.values() if r.get("decision") == "approved"
    )
    is_approved = selected_entry.get("researcher_approval_status") == "approved"
    val_rep = validate_dataset(scenarios, expected_scope=scope, dataset_approved=is_approved)

    cat_counts = val_rep.get("category_counts", {})
    insp_l, insp_r = st.columns([3.2, 1.2], vertical_alignment="center")
    with insp_l:
        st.markdown(
            f"**Inspecting {scope.title()} Version:** `{ver_id}` ({selected_entry.get('dataset_name', '')}) · "
            f"Source: {selected_entry.get('dataset_source', 'Local')} · "
            f"Technical validation: `{val_rep['validation_status']}` · "
            f"Approval: `{_format_approval(selected_entry.get('researcher_approval_status'))}` · "
            f"Categories: {cat_counts.get('clearly_benign', 0)} benign / "
            f"{cat_counts.get('sensitive_looking_legitimate', 0)} sensitive / "
            f"{cat_counts.get('genuinely_unsafe', 0)} unsafe · "
            f"Per-scenario reviews: {approved_scen_count} approved, {rejected_count} rejected"
        )
    with insp_r:
        is_currently_active = ver_id == active_ver_id
        if is_currently_active:
            st.caption(f"Currently active {scope} version")
        else:
            if st.button(
                f"Set {ver_id} Active",
                key=f"rr_activate_{scope}_{ver_id}",
                use_container_width=True,
            ):
                set_active_version(scope, ver_id)
                st.rerun()

    with st.expander("Validation Checks Summary (Schema, Matching, Exclusions & Signoff)", expanded=False):
        checks_rows = val_rep.get("checks_matrix", [])
        if checks_rows:
            st.dataframe(
                pd.DataFrame(checks_rows).rename(
                    columns={"check": "Validation Check", "status": "Result", "details": "Details"}
                ),
                use_container_width=True,
                hide_index=True,
            )

    if scenarios:
        f_col1, f_col2 = st.columns([1, 2])
        with f_col1:
            cat_filter = st.selectbox(
                "Filter by category",
                options=["all"] + list(CATEGORY_LABELS.keys()),
                format_func=lambda c: "All categories" if c == "all" else CATEGORY_LABELS[c],
                key=f"rr_cat_{scope}_{ver_id}",
            )
        filtered = (
            scenarios
            if cat_filter == "all"
            else [s for s in scenarios if s.get("category") == cat_filter]
        )
        if filtered:
            with f_col2:
                sel_sid = st.selectbox(
                    f"Select scenario ({len(filtered)} shown)",
                    options=[s["scenario_id"] for s in filtered],
                    format_func=lambda sid: next(
                        (
                            f"{s['scenario_id']} — {s['scenario_title']}"
                            for s in filtered
                            if s["scenario_id"] == sid
                        ),
                        sid,
                    ),
                    key=f"rr_scen_{scope}_{ver_id}",
                )
            scen = next(s for s in filtered if s["scenario_id"] == sel_sid)
            left_col, right_col = st.columns([1.45, 1])
            with left_col:
                st.markdown(f"**{scen['scenario_id']} — {scen['scenario_title']}**")
                src_note = (
                    f" · Benchmark: {scen['source_benchmark']} ({scen.get('source_license', 'Open Source')})"
                    if scen.get("source_benchmark")
                    else ""
                )
                st.caption(
                    f"Category: {CATEGORY_LABELS.get(scen['category'], scen['category'])} · "
                    f"Expected classification: {scen['expected_response_classification']}{src_note}"
                )
                st.markdown("**Text-only prompt**")
                st.write(scen["text_only_prompt"])
                st.markdown("**Image + Text prompt**")
                st.write(scen["image_text_prompt"])
                st.markdown("**Expected safe behavior**")
                st.write(scen["expected_safe_behavior"])
                with st.expander("Description, Intent & Provenance", expanded=False):
                    st.write(f"**Description:** {scen['scenario_description']}")
                    st.write(f"**Underlying intent:** {scen['underlying_user_intent']}")
                    st.write(f"**Rationale:** {scen['classification_rationale']}")
                    st.write(f"**Image provenance:** {scen['image_provenance']}")
            with right_col:
                img_ok, img_msg = check_image_valid(scen.get("image_asset_reference"))
                img_path = resolve_image_path(scen.get("image_asset_reference"))
                if img_ok and img_path and img_path.exists():
                    st.image(
                        str(img_path),
                        caption=f"{scen['image_asset_reference']} ({img_msg})",
                        use_container_width=True,
                    )
                else:
                    st.warning(f"Image unavailable: {img_msg}")

            cur_rev = scenario_reviews.get(sel_sid, {})
            st.caption(
                f"Scenario review status: `{cur_rev.get('decision', 'pending_review')}` "
                f"(Reviewer: `{cur_rev.get('reviewer_id') or 'None'}`)"
            )
        else:
            st.info("No scenarios match the selected category.")

    # Explicit Researcher Approval & Per-Scenario Signoff
    with st.container(border=True):
        st.markdown(f"**Researcher Approval — `{ver_id}`**")
        app_c1, app_c2 = st.columns([1.5, 2])
        with app_c1:
            reviewer_id = st.text_input(
                "Reviewer ID (required)",
                placeholder="e.g., researcher_01",
                key=f"rr_rev_{ver_id}",
            )
        with app_c2:
            confirm_approve = st.checkbox(
                f"I have reviewed `{ver_id}` and confirm it is ready for experimental use.",
                value=False,
                key=f"rr_confirm_{ver_id}",
            )

        has_rev = bool(reviewer_id.strip())
        if scenarios and filtered:
            s_notes = st.text_input(
                f"Optional review note for {sel_sid}",
                value=scenario_reviews.get(sel_sid, {}).get("notes", ""),
                key=f"rr_snotes_{ver_id}_{sel_sid}",
            )
            sr1, sr2, sr3 = st.columns(3)
            with sr1:
                if st.button(
                    f"Mark {sel_sid} Approved",
                    disabled=not has_rev,
                    key=f"rr_sapp_{ver_id}_{sel_sid}",
                    use_container_width=True,
                ):
                    record_scenario_review(
                        ver_id, sel_sid, "approved", reviewer_id.strip(), s_notes
                    )
                    st.rerun()
            with sr2:
                if st.button(
                    f"Reject {sel_sid}",
                    disabled=not has_rev,
                    key=f"rr_srej_{ver_id}_{sel_sid}",
                    use_container_width=True,
                ):
                    record_scenario_review(
                        ver_id, sel_sid, "rejected", reviewer_id.strip(), s_notes
                    )
                    st.rerun()
            with sr3:
                if st.button(
                    f"Reset {sel_sid} Pending",
                    disabled=not has_rev,
                    key=f"rr_spen_{ver_id}_{sel_sid}",
                    use_container_width=True,
                ):
                    record_scenario_review(
                        ver_id, sel_sid, "pending_review", reviewer_id.strip(), s_notes
                    )
                    st.rerun()

        if rejected_count > 0:
            st.error(
                f"{rejected_count} scenario(s) in `{ver_id}` are marked rejected. "
                "Export a revised dataset from Google AI Studio and import it as a new version."
            )

        can_approve = (
            selected_entry["technical_validation_status"] == "valid"
            and integrity["intact"]
            and rejected_count == 0
            and has_rev
            and confirm_approve
            and not is_approved
        )
        ab1, ab2, ab3 = st.columns(3)
        with ab1:
            if st.button(
                "Approve Dataset",
                type="primary",
                disabled=not can_approve,
                key=f"rr_approve_ds_{ver_id}",
                use_container_width=True,
            ):
                updated = set_version_approval(
                    ver_id, "approved", reviewer_id=reviewer_id.strip()
                )
                st.success(
                    f"Approved `{ver_id}` (SHA-256: `{updated['approved_sha256'][:16]}...`)."
                )
                st.rerun()
        with ab2:
            if st.button(
                "Revoke Approval",
                disabled=selected_entry.get("researcher_approval_status") == "pending_review",
                key=f"rr_revoke_ds_{ver_id}",
                use_container_width=True,
            ):
                set_version_approval(
                    ver_id, "pending_review", reviewer_id=reviewer_id.strip() or None
                )
                st.rerun()
        with ab3:
            st.download_button(
                "Validation Report (JSON)",
                data=json.dumps(val_rep, indent=2, ensure_ascii=False),
                file_name=f"{ver_id}_validation_report.json",
                mime="application/json",
                use_container_width=True,
            )

    with st.expander("Import New AI Studio Version & Version History", expanded=False):
        if versions:
            hist_rows = []
            for v in versions:
                vid = v["dataset_version"]
                integ = verify_version_integrity(vid)
                hist_rows.append(
                    {
                        "Version": vid,
                        "Scope": v["dataset_scope"],
                        "Scenarios": v["scenario_count"],
                        "Approval": _format_approval(v["researcher_approval_status"]),
                        "Reviewer": v.get("reviewer_id") or "Pending",
                        "SHA-256": (v.get("sha256") or "")[:16] + "...",
                        "Integrity": "Intact" if integ["intact"] else "Mismatch",
                    }
                )
            st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)

            ver_sel = st.selectbox(
                "Switch active version",
                options=[v["dataset_version"] for v in versions],
                key="rr_switch_ver",
            )
            sel_entry = get_version_entry(ver_sel)
            if sel_entry and st.button(f"Set `{ver_sel}` Active for {sel_entry['dataset_scope'].title()}"):
                set_active_version(sel_entry["dataset_scope"], ver_sel)
                st.rerun()

        dl1, dl2 = st.columns(2)
        if AISTUDIO_TEMPLATE_PATH.exists():
            dl1.download_button(
                "AI Studio Prompt Template (MD)",
                data=AISTUDIO_TEMPLATE_PATH.read_text(encoding="utf-8"),
                file_name="aistudio_prompt_template.md",
                mime="text/markdown",
                use_container_width=True,
            )
        if SCENARIO_SCHEMA_PATH.exists():
            dl2.download_button(
                "Scenario Schema (JSON)",
                data=SCENARIO_SCHEMA_PATH.read_text(encoding="utf-8"),
                file_name="scenario_schema.json",
                mime="application/json",
                use_container_width=True,
            )

        with st.form("rr_import_form"):
            i1, i2 = st.columns(2)
            with i1:
                imp_name = st.text_input("Dataset Name", value="Gemini4H-Curated")
                imp_ver = st.text_input("New Version ID", placeholder="e.g., v1.1.0-main")
            with i2:
                imp_scope = st.selectbox("Scope", options=["pilot", "main"], index=1)
                imp_src = st.text_input("Source", value="Google AI Studio Structured Export")
            uploaded = st.file_uploader("Upload JSON, JSONL, or CSV", type=["json", "jsonl", "csv"])
            if st.form_submit_button("Import Immutable Version"):
                if uploaded is None:
                    st.error("Select a JSON, JSONL, or CSV file first.")
                else:
                    try:
                        entry = import_dataset_version(
                            raw_bytes=uploaded.getvalue(),
                            filename=uploaded.name,
                            dataset_name=imp_name,
                            dataset_version=imp_ver,
                            dataset_scope=imp_scope,
                            dataset_source=imp_src,
                            set_active_if_valid=False,
                        )
                        st.success(f"Imported `{entry['dataset_version']}` (pending human approval).")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Import failed: {exc}")


def _render_pilot_section() -> None:
    st.subheader("B. Pilot Experiment")
    st.write(
        "12 scenarios × 2 input types × 4 thinking levels × 1 repetition = 96 calls."
    )

    pilot_entry, pilot_scenarios = load_active_dataset("pilot")
    if pilot_entry is None:
        st.warning("No active pilot dataset selected.")
        return

    is_approved = pilot_entry.get("researcher_approval_status") == "approved"
    api_ok = is_api_fully_validated()
    ctrl = get_controller("pilot")
    snap = ctrl.snapshot()
    pilot_records = load_trial_records("pilot")

    pilot_cfg = load_runtime_config()
    pilot_cfg.modalities = list(MODALITIES)
    pilot_cfg.thinking_levels = list(THINKING_LEVELS)
    pilot_cfg.repetitions = PILOT_TARGET_REPETITIONS

    plan = build_trial_plan(
        scenarios=pilot_scenarios,
        modalities=pilot_cfg.modalities,
        thinking_levels=pilot_cfg.thinking_levels,
        repetitions=pilot_cfg.repetitions,
        randomize_order=pilot_cfg.randomize_order,
        random_seed=pilot_cfg.random_seed,
    )
    pending, completed_cnt = filter_pending_trials("pilot", plan)
    failed_cnt = sum(1 for r in pilot_records if r.get("status") == "error")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Target", f"{PILOT_TARGET_CALLS} calls", f"Status: {snap['status']}")
    with m2:
        render_metric_card("Completed Calls", str(completed_cnt), "Saved in pilot JSONL")
    with m3:
        render_metric_card("Failed Calls", str(failed_cnt), "Logged without discard")
    with m4:
        render_metric_card("Remaining Calls", str(len(pending)), f"Dataset: {pilot_entry['dataset_version']}")

    st.caption(
        f"Runtime configuration: Max output tokens {pilot_cfg.max_output_tokens} · "
        f"Timeout {pilot_cfg.timeout_seconds:.0f}s · Random seed {pilot_cfg.random_seed}"
    )

    blockers = []
    if not api_ok:
        blockers.append("verify all 4 thinking levels in Settings")
    if not is_approved:
        blockers.append(f"approve pilot dataset `{pilot_entry['dataset_version']}` above")
    if blockers:
        st.info(f"Start Pilot is blocked until you {' and '.join(blockers)}.")

    is_running = snap["is_alive"]
    prereqs_ok = api_ok and is_approved and len(pending) > 0

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        if st.button(
            "Start Pilot",
            type="primary",
            disabled=not (not is_running and prereqs_ok and completed_cnt == 0),
            use_container_width=True,
            key="rr_pilot_start",
        ):
            ctrl.start(
                scenarios=pilot_scenarios,
                config=pilot_cfg,
                explicit_approved=True,
                dataset_version=pilot_entry.get("dataset_version"),
                dataset_hash=pilot_entry.get("approved_sha256") or pilot_entry.get("sha256"),
                require_approved_dataset=True,
                require_validated_levels=True,
            )
            st.rerun()
    with b2:
        if st.button(
            "Pause",
            disabled=not is_running,
            use_container_width=True,
            key="rr_pilot_pause",
        ):
            ctrl.pause()
            st.rerun()
    with b3:
        if st.button(
            "Resume",
            disabled=not (not is_running and prereqs_ok and completed_cnt > 0),
            use_container_width=True,
            key="rr_pilot_resume",
        ):
            ctrl.start(
                scenarios=pilot_scenarios,
                config=pilot_cfg,
                explicit_approved=True,
                dataset_version=pilot_entry.get("dataset_version"),
                dataset_hash=pilot_entry.get("approved_sha256") or pilot_entry.get("sha256"),
                require_approved_dataset=True,
                require_validated_levels=True,
            )
            st.rerun()
    with b4:
        if st.button(
            "Stop",
            disabled=not is_running,
            use_container_width=True,
            key="rr_pilot_stop",
        ):
            ctrl.stop()
            st.rerun()

    total_done = completed_cnt + snap["succeeded_in_run"] if is_running else completed_cnt
    if total_done > 0 or is_running:
        st.progress(min(1.0, total_done / max(1, len(plan))))
        st.caption(
            f"Execution status: {snap['status']} · Progress: {total_done}/{len(plan)}"
        )

    if snap.get("logs"):
        with st.expander("Recent Pilot Runner Activity", expanded=is_running):
            st.code("\n".join(snap["logs"][-12:]), language="text")

    if snap["last_error"]:
        st.error(f"Pilot runner error: {snap['last_error']}")

    if is_running:
        time.sleep(2.0)
        st.rerun()

    with st.expander("Pilot Report & Recorded Trials", expanded=False):
        pr1, pr2 = st.columns(2)
        with pr1:
            if st.button("Generate Pilot Evaluation Report", use_container_width=True):
                df_proc = build_processed_results("pilot", save_csv=False)
                rep = generate_pilot_report(df_proc, pilot_scenarios)
                build_experiment_manifest("pilot", save_path=PILOT_MANIFEST_JSON, config=pilot_cfg)
                st.success("Updated `research/pilot_report.md` and `pilot_experiment_manifest.json`.")
                st.markdown(rep)
        with pr2:
            if st.button("Refresh Pilot Manifest (JSON)", use_container_width=True):
                build_experiment_manifest("pilot", save_path=PILOT_MANIFEST_JSON, config=pilot_cfg)
                st.success("Updated `results/pilot/pilot_experiment_manifest.json`.")
        if pilot_records:
            st.dataframe(pd.DataFrame(pilot_records), use_container_width=True, hide_index=True)
        else:
            st.caption("No pilot trials recorded yet.")


def _render_main_section() -> None:
    st.subheader("C. Main Experiment")
    st.write(
        "60 scenarios × 2 input types × 4 thinking levels × 3 repetitions = 1,440 calls."
    )

    main_entry, main_scenarios = load_active_dataset("main")
    if main_entry is None:
        st.warning("No active main dataset selected.")
        return

    is_approved = main_entry.get("researcher_approval_status") == "approved"
    api_ok = is_api_fully_validated()
    ctrl = get_controller("main")
    snap = ctrl.snapshot()
    main_records = load_trial_records("main")
    main_state = load_experiment_state("main")

    pilot_records = load_trial_records("pilot")
    pilot_done = sum(1 for r in pilot_records if r.get("status") in ("success", "blocked"))

    main_cfg = load_runtime_config()
    main_cfg.modalities = list(MODALITIES)
    main_cfg.thinking_levels = list(THINKING_LEVELS)
    main_cfg.repetitions = MAIN_TARGET_REPETITIONS

    plan = build_trial_plan(
        scenarios=main_scenarios,
        modalities=main_cfg.modalities,
        thinking_levels=main_cfg.thinking_levels,
        repetitions=main_cfg.repetitions,
        randomize_order=main_cfg.randomize_order,
        random_seed=main_cfg.random_seed,
    )
    pending, completed_cnt = filter_pending_trials("main", plan)
    failed_cnt = sum(1 for r in main_records if r.get("status") == "error")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Target", f"{MAIN_TARGET_CALLS:,} calls", f"Status: {snap['status']}")
    with m2:
        render_metric_card("Completed Calls", f"{completed_cnt:,}", "Preserved on resume")
    with m3:
        render_metric_card("Failed Calls", str(failed_cnt), "Logged in main JSONL")
    with m4:
        render_metric_card("Remaining Calls", f"{len(pending):,}", f"Dataset: {main_entry['dataset_version']}")

    st.caption(
        f"Pilot completion: {pilot_done}/{PILOT_TARGET_CALLS} calls · "
        f"Main dataset approval: {_format_approval(main_entry.get('researcher_approval_status'))} · "
        f"Runtime configuration: Max output tokens {main_cfg.max_output_tokens} · "
        f"Timeout {main_cfg.timeout_seconds:.0f}s · Random seed {main_cfg.random_seed}"
    )

    explicit_approval = st.checkbox(
        "I confirm the pilot review is complete and explicitly authorize running the 1,440-call Main Experiment.",
        value=False,
        key="rr_main_explicit_approval",
    )

    readiness = verify_main_experiment_readiness(main_cfg, explicit_approved=explicit_approval)
    failing_checks = [c for c in readiness["checks"] if not c["passed"]]
    if failing_checks:
        reasons = "; ".join(f"{c['check']} ({c['detail']})" for c in failing_checks)
        st.info(f"Main Experiment is blocked until remaining checks pass: {reasons}")

    snap_file = get_config_snapshot_path("main")
    if snap_file.exists():
        st.caption(
            f"Configuration lock active (`{snap_file.name}`): execution halts if model or core config drifts."
        )

    is_running = snap["is_alive"]
    prereqs_ok = (
        readiness["ready"]
        and explicit_approval
        and is_approved
        and api_ok
        and len(pending) > 0
    )

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        if st.button(
            "Start Main Experiment",
            type="primary",
            disabled=not (not is_running and prereqs_ok and completed_cnt == 0),
            use_container_width=True,
            key="rr_main_start",
        ):
            ctrl.start(
                scenarios=main_scenarios,
                config=main_cfg,
                explicit_approved=True,
                dataset_version=main_entry.get("dataset_version"),
                dataset_hash=main_entry.get("approved_sha256") or main_entry.get("sha256"),
                require_approved_dataset=True,
                require_validated_levels=True,
            )
            st.rerun()
    with b2:
        if st.button(
            "Pause",
            disabled=not is_running,
            use_container_width=True,
            key="rr_main_pause",
        ):
            ctrl.pause()
            st.rerun()
    with b3:
        if st.button(
            "Resume",
            disabled=not (not is_running and prereqs_ok and completed_cnt > 0),
            use_container_width=True,
            key="rr_main_resume",
        ):
            ctrl.start(
                scenarios=main_scenarios,
                config=main_cfg,
                explicit_approved=True,
                dataset_version=main_entry.get("dataset_version"),
                dataset_hash=main_entry.get("approved_sha256") or main_entry.get("sha256"),
                require_approved_dataset=True,
                require_validated_levels=True,
            )
            st.rerun()
    with b4:
        if st.button(
            "Stop",
            disabled=not is_running,
            use_container_width=True,
            key="rr_main_stop",
        ):
            ctrl.stop()
            st.rerun()

    total_done = completed_cnt + snap["succeeded_in_run"] if is_running else completed_cnt
    if total_done > 0 or is_running:
        st.progress(min(1.0, total_done / max(1, len(plan))))
        st.caption(
            f"Execution status: {snap['status']} · Progress: {total_done:,}/{len(plan):,}"
        )

    if snap.get("logs"):
        with st.expander("Recent Main Runner Activity", expanded=is_running):
            st.code("\n".join(snap["logs"][-12:]), language="text")

    if snap["last_error"]:
        st.error(f"Main runner error: {snap['last_error']}")

    if is_running:
        time.sleep(2.0)
        st.rerun()

    with st.expander("Readiness Checks, Execution History & Reports", expanded=False):
        st.dataframe(pd.DataFrame(readiness["checks"]), use_container_width=True, hide_index=True)
        history = main_state.get("history", [])
        if history:
            st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)
        if st.button("Generate Main Manifest & Status Reports", key="rr_main_reports"):
            paths = generate_main_post_run_reports(explicit_approved=explicit_approval)
            st.success(
                f"Generated {len(paths)} files: {', '.join(p.name for p in paths.values())}"
            )


def render() -> None:
    render_page_header(
        "Research Runs",
        "Review and approve scenario datasets, run the 96-call Pilot, and execute the 1,440-call Main study.",
    )
    _render_dataset_section()
    st.divider()
    _render_pilot_section()
    st.divider()
    _render_main_section()
