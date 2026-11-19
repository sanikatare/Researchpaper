"""Experiment Lab — Exploratory chatbot interface isolated from official trials."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from app.components import navigate_to, render_empty_state, render_page_header
from config import DEFAULT_MODEL_ID, EXPLORATORY_UPLOADS_DIR, THINKING_LEVELS
from experiment.exploratory_lab import (
    check_thinking_level_availability,
    load_candidate_scenarios,
    save_candidate_scenario,
    save_uploaded_exploratory_image,
    send_exploratory_prompt,
)
from experiment.storage import load_runtime_config


def _format_token_summary(turn: Dict[str, Any]) -> str:
    in_t = turn.get("input_tokens")
    out_t = turn.get("output_tokens")
    th_t = turn.get("thinking_tokens")
    in_s = str(in_t) if in_t is not None else "Unavailable"
    out_s = str(out_t) if out_t is not None else "Unavailable"
    th_s = str(th_t) if th_t is not None else "Unavailable"
    lat_ms = turn.get("latency_ms")
    lat_s = f"{lat_ms:.0f} ms" if lat_ms is not None else "Unavailable"
    model_used = turn.get("actual_model_version") or turn.get("model_id") or DEFAULT_MODEL_ID
    mod_lbl = "Image + Text" if turn.get("modality") == "image_plus_text" else "Text only"
    return (
        f"{model_used} · {turn.get('thinking_level')} · {mod_lbl} · "
        f"Latency {lat_s} · Tokens (in/out/think): {in_s} / {out_s} / {th_s}"
    )


def render(client: Optional[Any] = None) -> None:
    render_page_header(
        "Experiment Lab",
        "Interactive prompt testing with Gemini 4 Hydrogen across thinking levels and modalities.",
    )

    st.info(
        "Exploratory testing only. These interactions are not part of the official research dataset."
    )

    if "lab_messages" not in st.session_state:
        st.session_state.lab_messages = []
    if "lab_attached_image" not in st.session_state:
        st.session_state.lab_attached_image = None
    if "lab_uploader_nonce" not in st.session_state:
        st.session_state.lab_uploader_nonce = 0

    cfg = load_runtime_config()

    # Central conversation history
    messages: List[Dict[str, Any]] = st.session_state.lab_messages
    if not messages:
        render_empty_state(
            "Start an exploratory session",
            "Type a prompt below to test Gemini 4 Hydrogen. Select a validated thinking level and optionally attach an image when using Image + Text mode.",
            "Saved chats stay separate from your 96-call Pilot and 1,440-call Main experiments.",
        )
    else:
        for idx, turn in enumerate(messages):
            user_text = html.escape(str(turn.get("prompt", "")))
            img_name = turn.get("image_filename")
            user_meta = (
                f"Attachment: {img_name}"
                if img_name and turn.get("modality") == "image_plus_text"
                else "Text only"
            )
            st.markdown(
                f"""
                <div class="chat-wrap">
                    <div class="chat-row-user">
                        <div class="chat-bubble-user">{user_text}</div>
                    </div>
                    <div class="chat-meta-user">{html.escape(user_meta)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if img_name and turn.get("image_path"):
                p = Path(str(turn["image_path"]))
                if p.exists():
                    _, img_col = st.columns([3, 1])
                    with img_col:
                        st.image(str(p), caption=img_name, use_container_width=True)

            if turn.get("status") == "error":
                err_code = html.escape(str(turn.get("error_code") or "API_ERROR"))
                err_msg = html.escape(str(turn.get("error_message") or "Unknown API error"))
                st.markdown(
                    f"""
                    <div class="chat-row-assistant">
                        <div class="chat-error-box">
                            <strong>API Error ({err_code})</strong><br/>{err_msg}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                resp_raw = str(turn.get("response_text") or "")
                if not resp_raw.strip() and turn.get("block_reason"):
                    resp_raw = f"[Response blocked by API: {turn.get('block_reason')}]"
                elif not resp_raw.strip():
                    resp_raw = "[Empty response returned by API]"
                meta_line = _format_token_summary(turn)
                st.markdown(
                    f"""
                    <div class="chat-wrap">
                        <div class="chat-row-assistant">
                            <div class="chat-bubble-assistant">{html.escape(resp_raw)}</div>
                        </div>
                        <div class="chat-meta-assistant">{html.escape(meta_line)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if turn.get("thought_summary"):
                    with st.expander("Thought Summary", expanded=False):
                        st.write(turn["thought_summary"])

            # Candidate scenario action
            act_l, act_r = st.columns([3, 1.4])
            with act_r:
                if st.button(
                    "Save as Candidate Scenario",
                    key=f"save_cand_{idx}_{turn.get('turn_id')}",
                    use_container_width=True,
                ):
                    cand = save_candidate_scenario(
                        prompt=str(turn.get("prompt", "")),
                        modality=str(turn.get("modality", "text_only")),
                        thinking_level=str(turn.get("thinking_level", "MEDIUM")),
                        response_text=str(turn.get("response_text", "")),
                        actual_model_version=turn.get("actual_model_version"),
                        image_filename=turn.get("image_filename"),
                    )
                    st.success(
                        f"Saved `{cand['candidate_id']}` to candidate scenarios (pending curation; not in official dataset)."
                    )

    st.divider()

    # Minimal controls directly above the bottom message composer
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.4, 1.2, 1])
    with ctrl_col1:
        selected_level = st.selectbox(
            "Thinking level",
            options=list(THINKING_LEVELS),
            index=1 if "MEDIUM" in THINKING_LEVELS else 0,
            key="lab_thinking_level",
        )
    with ctrl_col2:
        input_mode_label = st.radio(
            "Input type",
            options=["Text only", "Image + Text"],
            horizontal=True,
            key="lab_input_mode",
        )
    with ctrl_col3:
        st.write("")
        if st.button("Clear Chat", use_container_width=True, key="lab_clear_chat"):
            st.session_state.lab_messages = []
            st.rerun()

    input_mode = "image_plus_text" if input_mode_label == "Image + Text" else "text_only"

    # Validate selected thinking level against real verification state
    level_status = check_thinking_level_availability(selected_level)
    if not level_status["available"]:
        warn_col, nav_col = st.columns([4, 1.2], vertical_alignment="center")
        with warn_col:
            st.warning(level_status["reason"])
        with nav_col:
            if st.button("Open Settings", key="lab_open_settings", use_container_width=True):
                navigate_to("Settings")

    # Optional image uploader when Image + Text mode is active
    attached_meta = st.session_state.lab_attached_image
    if input_mode == "image_plus_text":
        uploaded = st.file_uploader(
            "Attach image (PNG, JPEG, WEBP, GIF — max 20 MB)",
            type=["png", "jpg", "jpeg", "webp", "gif"],
            key=f"lab_uploader_{st.session_state.lab_uploader_nonce}",
        )
        if uploaded is not None:
            try:
                saved_path, meta = save_uploaded_exploratory_image(
                    uploaded.getvalue(), uploaded.name
                )
                attached_meta = {
                    "path": str(saved_path),
                    "filename": uploaded.name,
                    "mime_type": meta["mime_type"],
                    "width": meta["width"],
                    "height": meta["height"],
                    "size_bytes": meta["size_bytes"],
                }
                st.session_state.lab_attached_image = attached_meta
            except Exception as exc:
                st.error(f"Invalid image attachment: {exc}")
                attached_meta = None
                st.session_state.lab_attached_image = None

        if attached_meta and Path(attached_meta["path"]).exists():
            prev_c, rem_c = st.columns([1.2, 3])
            with prev_c:
                st.image(
                    attached_meta["path"],
                    caption=(
                        f"{attached_meta['filename']} · "
                        f"{attached_meta['width']}×{attached_meta['height']} · "
                        f"{attached_meta['mime_type']}"
                    ),
                    use_container_width=True,
                )
            with rem_c:
                if st.button("Remove Image", key="lab_remove_image"):
                    used_paths = {
                        str(m.get("image_path"))
                        for m in st.session_state.get("lab_messages", [])
                        if m.get("image_path")
                    }
                    try:
                        img_p = Path(attached_meta["path"]).resolve()
                        if (
                            str(attached_meta["path"]) not in used_paths
                            and str(img_p) not in used_paths
                            and img_p.is_file()
                            and img_p.parent == EXPLORATORY_UPLOADS_DIR.resolve()
                        ):
                            img_p.unlink(missing_ok=True)
                    except Exception:
                        pass
                    st.session_state.lab_attached_image = None
                    st.session_state.lab_uploader_nonce += 1
                    st.rerun()

    with st.expander("Advanced Settings", expanded=False):
        adv_c1, adv_c2 = st.columns(2)
        with adv_c1:
            max_output_tokens = st.number_input(
                "Output token limit",
                min_value=64,
                max_value=65536,
                value=int(cfg.max_output_tokens),
                step=256,
                key="lab_max_output_tokens",
            )
        with adv_c2:
            include_thoughts = st.checkbox(
                "Include thought summaries when supported",
                value=bool(cfg.include_thoughts),
                key="lab_include_thoughts",
            )

    # Message composer pinned at the bottom of the interaction area
    with st.form("lab_prompt_form", clear_on_submit=True):
        inp_col, btn_col = st.columns([5, 1])
        with inp_col:
            user_prompt = st.text_area(
                "Prompt",
                placeholder="Type your safety or research prompt for Gemini 4 Hydrogen...",
                height=72,
                label_visibility="collapsed",
                key="lab_prompt_input",
            )
        with btn_col:
            send_clicked = st.form_submit_button(
                "Send",
                type="primary",
                disabled=not level_status["available"],
                use_container_width=True,
            )

    if send_clicked:
        if not user_prompt.strip():
            st.error("Please enter a prompt before sending.")
        elif input_mode == "image_plus_text" and (
            not attached_meta or not Path(attached_meta["path"]).exists()
        ):
            st.error("Please attach a valid image for Image + Text mode, or switch to Text only.")
        else:
            img_arg = (
                Path(attached_meta["path"])
                if (input_mode == "image_plus_text" and attached_meta)
                else None
            )
            with st.spinner(f"Calling {DEFAULT_MODEL_ID} ({selected_level})..."):
                try:
                    turn = send_exploratory_prompt(
                        prompt=user_prompt,
                        thinking_level=selected_level,
                        input_mode=input_mode,
                        image_path=img_arg,
                        max_output_tokens=int(max_output_tokens),
                        include_thoughts=bool(include_thoughts),
                        client=client,
                    )
                    st.session_state.lab_messages.append(turn)
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    candidates = load_candidate_scenarios()
    if candidates:
        with st.expander(f"Saved Candidate Scenarios ({len(candidates)})", expanded=False):
            st.caption(
                "Candidate scenarios are stored in `results/exploratory/candidate_scenarios.json` "
                "and remain separate from official dataset versions."
            )
            for c in reversed(candidates[-10:]):
                st.markdown(
                    f"**{c['candidate_id']}** · {c['modality']} · {c['thinking_level']} · `{c['review_status']}`"
                )
                st.caption(c["prompt"])
