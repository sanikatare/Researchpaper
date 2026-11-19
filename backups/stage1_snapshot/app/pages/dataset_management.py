"""PAGE 3 — Dataset Management & Researcher Scenario Approval."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.components import render_metric_card, render_page_header
from config import CATEGORY_LABELS, IMAGES_DIR, SCENARIO_CATEGORIES
from dataset.scenario_manager import (
    check_image_valid,
    load_scenarios,
    resolve_image_path,
    save_scenarios,
    validate_dataset,
    validate_scenario,
)


def render() -> None:
    render_page_header(
        "Dataset Management",
        "Inspect, Edit, Validate, and Researcher-Approve the 60 Matched Scenario Families",
    )

    scenarios = load_scenarios()
    summary = validate_dataset(scenarios)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Total Scenarios", str(summary["total_scenarios"]), "Target: 60 (20 per category)")
    with c2:
        render_metric_card("Technically Valid", str(summary["technically_valid_count"]), "Format, image & domain checks")
    with c3:
        render_metric_card(
            "Researcher Approved",
            str(summary["researcher_approved_count"]),
            "Required before entering experiments",
        )
    with c4:
        render_metric_card("Pilot Set Selected", str(summary["pilot_selected_count"]), "Target: 12 (4 per category)")

    dups = summary["duplicates"]
    if dups["duplicate_ids"] or dups["duplicate_prompts"]:
        st.error(
            f"Duplicate issues detected — IDs: {dups['duplicate_ids']} | "
            f"Prompts: {dups['duplicate_prompts']}"
        )

    # Filter controls
    f1, f2, f3 = st.columns([2, 1, 1])
    with f1:
        search_q = st.text_input("Search scenarios (ID, description, or prompt)", value="")
    with f2:
        cat_filter = st.multiselect(
            "Filter by Category",
            options=list(SCENARIO_CATEGORIES),
            default=list(SCENARIO_CATEGORIES),
            format_func=lambda c: CATEGORY_LABELS.get(c, c),
        )
    with f3:
        pilot_only = st.checkbox("Pilot Scenarios Only", value=False)

    val_by_id = {v["scenario_id"]: v for v in summary["per_scenario"]}
    filtered = []
    for s in scenarios:
        if s.get("category") not in cat_filter:
            continue
        if pilot_only and not s.get("in_pilot"):
            continue
        if search_q.strip():
            q = search_q.strip().lower()
            blob = " ".join(
                str(s.get(k, ""))
                for k in ("scenario_id", "scenario_description", "text_prompt", "image_text_prompt")
            ).lower()
            if q not in blob:
                continue
        filtered.append(s)

    table_rows = []
    for s in filtered:
        sid = s.get("scenario_id", "")
        v = val_by_id.get(sid, {})
        img_ok, img_msg = check_image_valid(s.get("image_path"))
        review_status = (
            "Approved" if s.get("researcher_approved") is True else "Pending Review"
        )
        tech_status = "Tech Pass" if v.get("technical_valid") else f"Issues ({len(v.get('issues', []))})"
        table_rows.append(
            {
                "Scenario ID": sid,
                "Category": CATEGORY_LABELS.get(s.get("category", ""), s.get("category", "")),
                "Pilot": "Yes" if s.get("in_pilot") else "No",
                "Scenario Description": s.get("scenario_description", ""),
                "Text Prompt": s.get("text_prompt", ""),
                "Image + Text Prompt": s.get("image_text_prompt", ""),
                "Image Availability": img_msg if img_ok else f"Missing ({img_msg})",
                "Expected Safe Behavior": s.get("expected_safe_behavior", ""),
                "Expected Classification": s.get("expected_classification", ""),
                "Validation Status": f"{tech_status} | {review_status}",
            }
        )

    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    # Batch researcher approval controls (never automatic)
    st.subheader("Researcher Review & Validation Actions")
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        if st.button("Run Duplicate & Modality Validation", use_container_width=True):
            if not dups["duplicate_ids"] and not dups["duplicate_prompts"]:
                st.success(
                    f"Duplicate check passed. {summary['technically_valid_count']}/{summary['total_scenarios']} "
                    "scenarios passed technical & matched-modality checks."
                )
            else:
                st.warning("Duplicate IDs or prompts found; see alert above.")
    with b2:
        if st.button("Approve Filtered (Tech-Valid Only)", use_container_width=True):
            count = 0
            filtered_ids = {s["scenario_id"] for s in filtered}
            for s in scenarios:
                if s["scenario_id"] in filtered_ids and val_by_id.get(s["scenario_id"], {}).get("technical_valid"):
                    s["researcher_approved"] = True
                    s["review_status"] = "approved"
                    count += 1
            save_scenarios(scenarios)
            st.success(f"Researcher approved {count} technically valid scenarios.")
            st.rerun()
    with b3:
        if st.button("Reset Approval on Filtered", use_container_width=True):
            filtered_ids = {s["scenario_id"] for s in filtered}
            for s in scenarios:
                if s["scenario_id"] in filtered_ids:
                    s["researcher_approved"] = False
                    s["review_status"] = "pending_review"
            save_scenarios(scenarios)
            st.info("Reset researcher approval on filtered scenarios.")
            st.rerun()
    with b4:
        st.download_button(
            "Export Dataset JSON",
            data=json.dumps({"scenarios": scenarios}, indent=2, ensure_ascii=False),
            file_name="matched_scenarios.json",
            mime="application/json",
            use_container_width=True,
        )

    st.divider()
    st.subheader("Inspect, Edit, or Delete a Scenario")
    if scenarios:
        sid_list = [s["scenario_id"] for s in scenarios]
        selected_sid = st.selectbox("Select Scenario ID", options=sid_list)
        target_idx = next(i for i, s in enumerate(scenarios) if s["scenario_id"] == selected_sid)
        scen = scenarios[target_idx]
        scen_val = validate_scenario(scen)

        col_prev, col_edit = st.columns([1, 2])
        with col_prev:
            st.markdown("**Associated Image Preview**")
            img_path = resolve_image_path(scen.get("image_path"))
            if img_path and img_path.exists():
                st.image(str(img_path), caption=scen.get("image_path"), use_container_width=True)
            else:
                st.warning("No image file found.")

            uploaded_img = st.file_uploader(
                "Upload Replacement Image (PNG/JPG/WEBP)",
                type=["png", "jpg", "jpeg", "webp"],
                key=f"upload_{selected_sid}",
            )
            if uploaded_img is not None:
                IMAGES_DIR.mkdir(parents=True, exist_ok=True)
                ext = Path(uploaded_img.name).suffix.lower() or ".png"
                dest_name = f"{selected_sid.lower()}{ext}"
                dest_path = IMAGES_DIR / dest_name
                dest_path.write_bytes(uploaded_img.getvalue())
                scen["image_path"] = dest_name
                save_scenarios(scenarios)
                st.success(f"Uploaded image to {dest_name}")
                st.rerun()

            if scen_val["issues"]:
                st.error("Validation Issues:\n- " + "\n- ".join(scen_val["issues"]))
            else:
                st.success("Technical validation passed.")

        with col_edit:
            with st.form(f"edit_form_{selected_sid}"):
                new_cat = st.selectbox(
                    "Category",
                    options=list(SCENARIO_CATEGORIES),
                    index=list(SCENARIO_CATEGORIES).index(scen.get("category", "clearly_benign")),
                )
                new_desc = st.text_input("Scenario Description", value=scen.get("scenario_description", ""))
                new_text = st.text_area("Text-Only Prompt", value=scen.get("text_prompt", ""), height=90)
                new_img_text = st.text_area("Image + Text Prompt", value=scen.get("image_text_prompt", ""), height=90)
                new_behavior = st.text_area("Expected Safe Behavior", value=scen.get("expected_safe_behavior", ""), height=70)
                new_exp_cls = st.selectbox(
                    "Expected Classification",
                    options=["safe_completion", "appropriate_refusal"],
                    index=0 if scen.get("expected_classification") == "safe_completion" else 1,
                )
                e1, e2 = st.columns(2)
                new_in_pilot = e1.checkbox("Include in Pilot (12-scenario subset)", value=bool(scen.get("in_pilot")))
                new_approved = e2.checkbox("Researcher Approved", value=bool(scen.get("researcher_approved")))

                if st.form_submit_button("Save Scenario Changes"):
                    scenarios[target_idx].update(
                        {
                            "category": new_cat,
                            "scenario_description": new_desc.strip(),
                            "text_prompt": new_text.strip(),
                            "image_text_prompt": new_img_text.strip(),
                            "expected_safe_behavior": new_behavior.strip(),
                            "expected_classification": new_exp_cls,
                            "in_pilot": new_in_pilot,
                            "researcher_approved": new_approved,
                            "review_status": "approved" if new_approved else "pending_review",
                        }
                    )
                    save_scenarios(scenarios)
                    st.success(f"Updated {selected_sid}.")
                    st.rerun()

            with st.expander("Delete Scenario (Requires Confirmation)"):
                confirm_del = st.checkbox(f"Confirm permanent deletion of {selected_sid}")
                if st.button("Delete Selected Scenario", disabled=not confirm_del):
                    scenarios.pop(target_idx)
                    save_scenarios(scenarios)
                    st.warning(f"Deleted {selected_sid}.")
                    st.rerun()

    st.divider()
    with st.expander("Add New Scenario Family"):
        with st.form("add_scenario_form"):
            a_id = st.text_input("Scenario ID (e.g., SCN-061)")
            a_cat = st.selectbox("Category", options=list(SCENARIO_CATEGORIES), key="add_cat")
            a_desc = st.text_input("Scenario Description")
            a_text = st.text_area("Text Prompt")
            a_img_text = st.text_area("Image + Text Prompt")
            a_behavior = st.text_area("Expected Safe Behavior")
            a_exp_cls = st.selectbox("Expected Classification", options=["safe_completion", "appropriate_refusal"], key="add_cls")
            a_img_file = st.file_uploader("Upload Associated Image", type=["png", "jpg", "jpeg", "webp"], key="add_img")
            if st.form_submit_button("Add Scenario"):
                sid_clean = a_id.strip()
                if not sid_clean or any(s["scenario_id"] == sid_clean for s in scenarios):
                    st.error("Scenario ID must be non-empty and unique.")
                elif a_img_file is None:
                    st.error("Please upload an associated image for the multimodal pair.")
                else:
                    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
                    ext = Path(a_img_file.name).suffix.lower() or ".png"
                    img_name = f"{sid_clean.lower()}{ext}"
                    (IMAGES_DIR / img_name).write_bytes(a_img_file.getvalue())
                    scenarios.append(
                        {
                            "scenario_id": sid_clean,
                            "category": a_cat,
                            "scenario_description": a_desc.strip(),
                            "text_prompt": a_text.strip(),
                            "image_text_prompt": a_img_text.strip(),
                            "image_path": img_name,
                            "expected_safe_behavior": a_behavior.strip(),
                            "expected_classification": a_exp_cls,
                            "in_pilot": False,
                            "researcher_approved": False,
                            "review_status": "pending_review",
                        }
                    )
                    save_scenarios(scenarios)
                    st.success(f"Added {sid_clean} (pending researcher review).")
                    st.rerun()
