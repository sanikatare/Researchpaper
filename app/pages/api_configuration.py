"""PAGE 2 — API Configuration & Thinking-Level Verification."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components import render_metric_card, render_page_header
from config import (
    API_READINESS_REPORT_MD,
    DEFAULT_MODEL_ID,
    THINKING_LEVELS,
    get_api_key,
    mask_api_key,
)
from dataset.scenario_manager import resolve_image_path
from experiment.hydrogen_client import (
    API_ERROR_LOG_PATH,
    generate_api_readiness_report,
    get_sdk_version,
    load_validation_state,
    test_single_thinking_level,
    verify_model_endpoint,
)
from experiment.storage import load_runtime_config, save_runtime_config


def render() -> None:
    render_page_header(
        "API Configuration",
        "Environment Credentials, Gemini 4 Hydrogen Parameters, and Live Thinking-Level Verification",
    )

    api_key = get_api_key()
    sdk_ver = get_sdk_version()

    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card(
            "GEMINI_API_KEY (Env)",
            mask_api_key(api_key),
            "Loaded from environment / .env; never saved in results",
            accent_color="#34A853" if api_key else "#EA4335",
        )
    with c2:
        render_metric_card(
            "Model Identifier",
            DEFAULT_MODEL_ID,
            "Fixed experimental subject",
            accent_color="#4285F4",
        )
    with c3:
        render_metric_card(
            "google-genai SDK",
            sdk_ver,
            "Official Python SDK",
            accent_color="#4285F4",
        )

    st.info(
        "Google AI Studio Project Logging: When logging is enabled for your project in "
        "Google AI Studio, API calls appear in the AI Studio Logs dashboard for curation and "
        "CSV/JSONL/Sheets export. Regardless of account logging tier, this engine always "
        "preserves immutable local JSONL records."
    )

    if not api_key:
        st.error(
            "GEMINI_API_KEY was not detected in the runtime environment. Set `GEMINI_API_KEY` "
            "in your environment variables, `.env` file, or AI Studio Secrets panel, then restart/reload."
        )

    cfg = load_runtime_config()

    st.subheader("Experiment Execution Parameters")
    with st.form("api_config_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.text_input("Model Identifier", value=DEFAULT_MODEL_ID, disabled=True)
            selected_levels = st.multiselect(
                "Active Thinking Levels",
                options=list(THINKING_LEVELS),
                default=[lvl for lvl in cfg.thinking_levels if lvl in THINKING_LEVELS]
                or list(THINKING_LEVELS),
            )
            max_output_tokens = st.number_input(
                "Output Token Limit (max_output_tokens)",
                min_value=64,
                max_value=65536,
                value=int(cfg.max_output_tokens),
                step=256,
                help="Includes both thinking tokens and visible response tokens.",
            )
        with col_b:
            timeout_seconds = st.number_input(
                "Request Timeout (seconds)",
                min_value=5.0,
                max_value=600.0,
                value=float(cfg.timeout_seconds),
                step=5.0,
            )
            max_retries = st.number_input(
                "Max Retries (transient errors / rate limits)",
                min_value=0,
                max_value=10,
                value=int(cfg.max_retries),
                step=1,
            )
            retry_base_delay = st.number_input(
                "Retry Base Delay (seconds)",
                min_value=0.5,
                max_value=30.0,
                value=float(cfg.retry_base_delay),
                step=0.5,
            )
            min_request_interval = st.number_input(
                "Rate-Limit Pacing Interval (seconds)",
                min_value=0.0,
                max_value=60.0,
                value=float(getattr(cfg, "min_request_interval", 0.0)),
                step=0.25,
                help="Minimum delay enforced between consecutive API requests.",
            )
            include_thoughts = st.checkbox(
                "Request Thought Summaries (include_thoughts)",
                value=bool(cfg.include_thoughts),
            )

        saved = st.form_submit_button("Save Runtime Configuration")
        if saved:
            cfg.thinking_levels = selected_levels or list(THINKING_LEVELS)
            cfg.max_output_tokens = int(max_output_tokens)
            cfg.timeout_seconds = float(timeout_seconds)
            cfg.max_retries = int(max_retries)
            cfg.retry_base_delay = float(retry_base_delay)
            cfg.min_request_interval = float(min_request_interval)
            cfg.include_thoughts = bool(include_thoughts)
            errs = cfg.validate()
            if errs:
                st.error("; ".join(errs))
            else:
                save_runtime_config(cfg)
                st.success("Runtime configuration saved (without credentials).")

    st.divider()
    st.subheader("Live API Connection & Thinking-Level Verification")
    st.caption(
        "Each thinking level is marked as validated only after a real API call succeeds "
        "with that exact configuration. Unsupported levels are never simulated."
    )

    if st.button("Verify Model Endpoint (models.get)"):
        with st.spinner(f"Querying Google AI Studio model metadata for {DEFAULT_MODEL_ID}..."):
            ep = verify_model_endpoint(
                model_id=DEFAULT_MODEL_ID,
                timeout_seconds=min(30.0, float(cfg.timeout_seconds)),
            )
        if ep["accessible"]:
            st.success(
                f"Endpoint accessible: `{ep['endpoint_name']}` "
                f"(Display: {ep.get('display_name') or 'N/A'} | "
                f"Input limit: {ep.get('input_token_limit')} | "
                f"Output limit: {ep.get('output_token_limit')})"
            )
        else:
            st.error(f"Endpoint check failed: {ep['error']}")

    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        single_level = st.selectbox(
            "Select Thinking Level to Test",
            options=list(THINKING_LEVELS),
            index=0,
        )
        include_multimodal_probe = st.checkbox(
            "Test using image-plus-text stimulus (dataset/images/scn-001.png)",
            value=False,
        )
    with col_t2:
        st.write("")
        st.write("")
        run_single = st.button("Test Selected Level", use_container_width=True)
        run_all = st.button("Test All 4 Levels", type="primary", use_container_width=True)

    probe_image = resolve_image_path("scn-001.png") if include_multimodal_probe else None

    if run_single:
        with st.spinner(f"Testing {DEFAULT_MODEL_ID} with thinking_level={single_level}..."):
            res = test_single_thinking_level(
                thinking_level=single_level,
                model_id=DEFAULT_MODEL_ID,
                max_output_tokens=min(1024, int(cfg.max_output_tokens)),
                timeout_seconds=float(cfg.timeout_seconds),
                max_retries=int(cfg.max_retries),
                image_path=probe_image,
            )
        _render_test_result(res)

    if run_all:
        results = []
        for lvl in THINKING_LEVELS:
            with st.spinner(f"Testing {DEFAULT_MODEL_ID} with thinking_level={lvl}..."):
                res = test_single_thinking_level(
                    thinking_level=lvl,
                    model_id=DEFAULT_MODEL_ID,
                    max_output_tokens=min(1024, int(cfg.max_output_tokens)),
                    timeout_seconds=float(cfg.timeout_seconds),
                    max_retries=int(cfg.max_retries),
                    image_path=probe_image,
                )
                results.append(res)
        for res in results:
            _render_test_result(res)

    st.subheader("Persisted Thinking-Level Validation Matrix")
    val_state = load_validation_state()
    rows = []
    for lvl in THINKING_LEVELS:
        info = val_state.get("levels", {}).get(lvl, {})
        supported = info.get("supported_by_endpoint")
        if supported is True:
            supported_label = "SUPPORTED"
        elif supported is False:
            supported_label = "UNSUPPORTED"
        else:
            supported_label = "UNTESTED"
        rows.append(
            {
                "Thinking Level": lvl,
                "Validated": "VALIDATED" if info.get("validated") else "NOT VALIDATED",
                "Endpoint Supported": supported_label,
                "Modality Tested": info.get("modality_tested") or "None",
                "Actual Model Version": info.get("actual_model_version") or "N/A",
                "Latency (ms)": info.get("latency_ms"),
                "Input Tokens": info.get("input_tokens"),
                "Output Tokens": info.get("output_tokens"),
                "Thinking Tokens": info.get("thinking_tokens"),
                "Last Tested (UTC)": info.get("last_tested_utc") or "Never",
                "Error": info.get("error") or "",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    mm_probe = val_state.get("multimodal_probe", {})
    st.caption(
        f"Multimodal (`image_plus_text`) Probe: "
        f"`{'VALIDATED' if mm_probe.get('validated') else 'NOT VALIDATED'}` "
        f"(level={mm_probe.get('thinking_level') or 'N/A'}, last_tested={mm_probe.get('last_tested_utc') or 'Never'})"
    )

    rep_col1, rep_col2 = st.columns(2)
    with rep_col1:
        if st.button("Generate API Readiness Report (MD)", use_container_width=True):
            rep_md = generate_api_readiness_report(API_READINESS_REPORT_MD)
            st.success("Generated `research/api_readiness_report.md`.")
            st.markdown(rep_md)
    with rep_col2:
        if API_READINESS_REPORT_MD.exists():
            st.download_button(
                "Download API Readiness Report (MD)",
                data=API_READINESS_REPORT_MD.read_text(encoding="utf-8"),
                file_name="api_readiness_report.md",
                mime="text/markdown",
                use_container_width=True,
            )

    with st.expander("API Error Log (results/api_errors.log)"):
        if API_ERROR_LOG_PATH.exists():
            lines = API_ERROR_LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = "\n".join(lines[-50:]) if lines else "Log file is empty."
            st.code(tail, language="text")
        else:
            st.caption("No API errors have been logged yet.")


def _render_test_result(res) -> None:
    if res.status == "success":
        st.success(
            f"[{res.thinking_level}] Connection succeeded | Actual model: "
            f"`{res.actual_model_version}` | Latency: {res.latency_ms:.1f} ms"
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Input Tokens", str(res.input_tokens) if res.input_tokens is not None else "Unavailable")
        m2.metric("Output Tokens", str(res.output_tokens) if res.output_tokens is not None else "Unavailable")
        m3.metric("Thinking Tokens", str(res.thinking_tokens) if res.thinking_tokens is not None else "Unavailable")
        m4.metric("Total Tokens", str(res.total_tokens) if res.total_tokens is not None else "Unavailable")
        st.text_area(
            f"Response ({res.thinking_level})",
            value=res.response_text,
            height=90,
            disabled=True,
            key=f"resp_{res.thinking_level}_{res.timestamp_utc}",
        )
    else:
        st.error(
            f"[{res.thinking_level}] Failed ({res.error_code}): {res.error_message}"
        )
