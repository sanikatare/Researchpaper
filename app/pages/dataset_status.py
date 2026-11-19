"""PAGE 3 — Read-Only Dataset Status, AI Studio Import, and Version Registry."""

from __future__ import annotations

import json
import pandas as pd
import streamlit as st

from app.components import render_metric_card, render_page_header, render_status_badge
from config import AISTUDIO_TEMPLATE_PATH, CATEGORY_LABELS, SCENARIO_SCHEMA_PATH
from dataset.scenario_manager import (
    check_image_valid,
    get_version_entry,
    import_dataset_version,
    load_active_dataset,
    load_registry,
    load_scenarios_for_version,
    resolve_image_path,
    set_active_version,
    set_version_approval,
    validate_dataset,
    verify_version_integrity,
)


def render() -> None:
    render_page_header(
        "Dataset Status (Read-Only)",
        "Scenario Definitions Are Authored & Curated Exclusively in Google AI Studio and Imported as Immutable Versions",
    )

    st.info(
        "This workbench enforces strict read-only dataset inspection: scenario content cannot be "
        "created or edited inside Streamlit. Author or revise scenarios in Google AI Studio using "
        "the structured schema, export as JSON or JSONL, and import them here as a new dataset version."
    )

    registry = load_registry()
    versions = registry.get("versions", [])
    pilot_entry, _ = load_active_dataset("pilot")
    main_entry, _ = load_active_dataset("main")

    # Active Scope Summary Cards
    st.subheader("Active Pilot & Main Confirmatory Dataset Versions")
    col_p, col_m = st.columns(2)
    with col_p:
        _render_scope_summary("Pilot Experiment Dataset (Target: 12 Scenarios)", pilot_entry)
    with col_m:
        _render_scope_summary("Confirmatory Main Dataset (Target: 60 Scenarios)", main_entry)

    # Import New Version from Google AI Studio
    with st.expander("Import New Dataset Version from Google AI Studio (JSON / JSONL)"):
        c_dl1, c_dl2 = st.columns(2)
        if AISTUDIO_TEMPLATE_PATH.exists():
            c_dl1.download_button(
                "Download AI Studio Prompt Template (MD)",
                data=AISTUDIO_TEMPLATE_PATH.read_text(encoding="utf-8"),
                file_name="aistudio_prompt_template.md",
                mime="text/markdown",
                use_container_width=True,
            )
        if SCENARIO_SCHEMA_PATH.exists():
            c_dl2.download_button(
                "Download Scenario JSON Schema",
                data=SCENARIO_SCHEMA_PATH.read_text(encoding="utf-8"),
                file_name="scenario_schema.json",
                mime="application/json",
                use_container_width=True,
            )

        with st.form("import_aistudio_dataset_form"):
            i1, i2 = st.columns(2)
            with i1:
                imp_name = st.text_input("Dataset Name", value="Gemini4H-Curated")
                imp_ver = st.text_input("New Dataset Version (unique)", placeholder="e.g., v1.1.0-main")
            with i2:
                imp_scope = st.selectbox("Experiment Scope", options=["pilot", "main"], index=1)
                imp_source = st.text_input(
                    "Dataset Source", value="Google AI Studio Structured Export"
                )
            uploaded = st.file_uploader(
                "Upload Google AI Studio Exported JSON, JSONL, or CSV",
                type=["json", "jsonl", "csv"],
            )
            imp_submit = st.form_submit_button("Validate & Import Immutable Version", type="primary")
            if imp_submit:
                if uploaded is None:
                    st.error("Please select a JSON or JSONL file to import.")
                else:
                    try:
                        entry = import_dataset_version(
                            raw_bytes=uploaded.getvalue(),
                            filename=uploaded.name,
                            dataset_name=imp_name,
                            dataset_version=imp_ver,
                            dataset_scope=imp_scope,
                            dataset_source=imp_source,
                            set_active_if_valid=False,
                        )
                        st.success(
                            f"Imported `{entry['dataset_version']}` ({entry['scenario_count']} scenarios, "
                            f"technical validation: `{entry['technical_validation_status']}`, "
                            f"SHA-256: `{entry['sha256'][:16]}...`). "
                            "Grant explicit Researcher Approval below before using in experiments."
                        )
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Import rejected: {exc}")

    st.divider()
    st.subheader("Dataset Version History & Researcher Approval")
    if not versions:
        st.warning("No dataset versions registered yet.")
        return

    hist_rows = []
    for v in versions:
        ver_id = v["dataset_version"]
        is_active = (
            ver_id == registry.get("active_pilot_version")
            or ver_id == registry.get("active_main_version")
        )
        integrity = verify_version_integrity(ver_id)
        hist_rows.append(
            {
                "Dataset Name": v["dataset_name"],
                "Version": ver_id,
                "Scope": v["dataset_scope"],
                "Active": "Yes" if is_active else "No",
                "Scenarios": v["scenario_count"],
                "Tech Valid": f"{v['technically_valid_count']}/{v['scenario_count']} ({v['technical_validation_status']})",
                "Approval Status": v["researcher_approval_status"],
                "SHA-256 (Import)": (v.get("sha256") or "")[:16] + "...",
                "Approved SHA-256": ((v.get("approved_sha256") or "")[:16] + "...") if v.get("approved_sha256") else "Pending",
                "Integrity": "INTACT" if integrity["intact"] else "MISMATCH",
                "Import Date (UTC)": v["import_date_utc"],
            }
        )
    st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)

    # Select a version to inspect, approve, or activate
    ver_options = [v["dataset_version"] for v in versions]
    selected_ver = st.selectbox(
        "Inspect Dataset Version (Read-Only)",
        options=ver_options,
        index=len(ver_options) - 1,
    )
    entry = get_version_entry(selected_ver)
    scenarios = load_scenarios_for_version(selected_ver)
    if entry is None:
        return

    is_approved = entry.get("researcher_approval_status") == "approved"
    scope = entry["dataset_scope"]
    val_rep = validate_dataset(scenarios, expected_scope=scope, dataset_approved=is_approved)
    integrity = verify_version_integrity(selected_ver)

    # Approval & Activation Controls (No content editing!)
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        can_approve = entry["technical_validation_status"] == "valid" and integrity["intact"]
        if st.button(
            "Approve Version (Researcher)",
            type="primary",
            disabled=not can_approve or is_approved,
            use_container_width=True,
        ):
            updated = set_version_approval(selected_ver, "approved")
            st.success(
                f"Researcher approved `{selected_ver}` (Approved SHA-256: `{updated['approved_sha256']}`)."
            )
            st.rerun()
    with a2:
        if st.button(
            "Mark Pending / Revoke Approval",
            disabled=entry["researcher_approval_status"] == "pending_review",
            use_container_width=True,
        ):
            set_version_approval(selected_ver, "pending_review")
            st.info(f"Reset approval for `{selected_ver}` to pending_review.")
            st.rerun()
    with a3:
        if st.button(
            f"Set Active for {scope.title()}",
            use_container_width=True,
        ):
            set_active_version(scope, selected_ver)
            st.success(f"Active {scope} version set to `{selected_ver}`.")
            st.rerun()
    with a4:
        st.download_button(
            "Download Validation Report (JSON)",
            data=json.dumps(val_rep, indent=2, ensure_ascii=False),
            file_name=f"{selected_ver}_validation_report.json",
            mime="application/json",
            use_container_width=True,
        )

    st.caption(
        f"Import SHA-256: `{entry.get('sha256')}` | "
        f"Approved SHA-256: `{entry.get('approved_sha256') or 'Not approved yet'}` | "
        f"Disk Integrity: `{'INTACT' if integrity['intact'] else integrity['error']}`"
    )

    st.subheader(f"Validation Report — {selected_ver} ({val_rep['overall_classification']})")
    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("Overall Status", val_rep["overall_classification"])
    rc2.metric("Scenarios: PASS", val_rep["classification_counts"]["PASS"])
    rc3.metric("Scenarios: NEEDS REVIEW", val_rep["classification_counts"]["NEEDS HUMAN REVIEW"])
    rc4.metric("Scenarios: FAIL", val_rep["classification_counts"]["FAIL"])

    st.dataframe(pd.DataFrame(val_rep["checks_matrix"]), use_container_width=True, hide_index=True)

    # Read-Only Scenario Browser & Image Preview
    st.subheader(f"Scenario Preview — {selected_ver} (Read-Only)")
    f1, f2 = st.columns([2, 1])
    with f1:
        q = st.text_input("Search scenarios (ID, title, intent, or prompt)", value="")
    with f2:
        cats = st.multiselect(
            "Filter by Category",
            options=list(CATEGORY_LABELS.keys()),
            default=list(CATEGORY_LABELS.keys()),
            format_func=lambda c: CATEGORY_LABELS[c],
        )

    report_by_sid = {r["scenario_id"]: r for r in val_rep["per_scenario"]}
    filtered = []
    for s in scenarios:
        if s.get("category") not in cats:
            continue
        if q.strip():
            blob = " ".join(
                str(s.get(k, ""))
                for k in (
                    "scenario_id",
                    "scenario_title",
                    "scenario_description",
                    "underlying_user_intent",
                    "text_only_prompt",
                    "image_text_prompt",
                )
            ).lower()
            if q.strip().lower() not in blob:
                continue
        filtered.append(s)

    if filtered:
        table_df = pd.DataFrame(
            [
                {
                    "Scenario ID": s["scenario_id"],
                    "Category": CATEGORY_LABELS.get(s["category"], s["category"]),
                    "Title": s["scenario_title"],
                    "Underlying Intent": s["underlying_user_intent"],
                    "Text-Only Prompt": s["text_only_prompt"],
                    "Image + Text Prompt": s["image_text_prompt"],
                    "Image Ref": s["image_asset_reference"],
                    "Expected Class": s["expected_response_classification"],
                    "Report Status": report_by_sid.get(s["scenario_id"], {}).get("report_classification", "NEEDS HUMAN REVIEW"),
                }
                for s in filtered
            ]
        )
        st.dataframe(table_df, use_container_width=True, hide_index=True)

        sel_sid = st.selectbox(
            "Inspect Scenario Details & Paired Image",
            options=[s["scenario_id"] for s in filtered],
        )
        scen = next(s for s in filtered if s["scenario_id"] == sel_sid)
        p_left, p_right = st.columns([1.4, 1])
        with p_left:
            st.markdown(f"**{scen['scenario_id']} — {scen['scenario_title']}**")
            st.markdown(f"**Category:** {CATEGORY_LABELS.get(scen['category'], scen['category'])}")
            st.markdown(f"**Description:** {scen['scenario_description']}")
            st.markdown(f"**Underlying User Intent:** {scen['underlying_user_intent']}")
            st.markdown(f"**Text-Only Prompt:** {scen['text_only_prompt']}")
            st.markdown(f"**Image + Text Prompt:** {scen['image_text_prompt']}")
            st.markdown(f"**Expected Safe Behavior:** {scen['expected_safe_behavior']}")
            st.markdown(
                f"**Expected Classification:** `{scen['expected_response_classification']}` "
                f"— {scen['classification_rationale']}"
            )
            st.caption(f"Image Provenance: {scen['image_provenance']}")
        with p_right:
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


def _render_scope_summary(heading: str, entry: dict | None) -> None:
    st.markdown(f"**{heading}**")
    if entry is None:
        render_metric_card("Status", "Not Set", "Import or select an approved version below")
        return

    approval = entry.get("researcher_approval_status", "pending_review")
    tone = "green" if approval == "approved" else ("red" if approval == "rejected" else "yellow")
    accent = "#34A853" if approval == "approved" else "#FBBC04"
    render_metric_card(
        entry["dataset_name"],
        entry["dataset_version"],
        f"{entry['scenario_count']} scenarios | Source: {entry['dataset_source']}",
        accent_color=accent,
    )
    st.markdown(
        render_status_badge(f"Tech: {entry['technical_validation_status']}", "blue")
        + " "
        + render_status_badge(f"Approval: {approval}", tone),
        unsafe_allow_html=True,
    )
    st.caption(
        f"Categories: {entry['category_counts']} | "
        f"Modalities: Text={entry['modality_coverage'].get('text_only_count', 0)}, "
        f"Image+Text={entry['modality_coverage'].get('image_plus_text_count', 0)} | "
        f"File: `{entry['file_reference']}`"
    )
