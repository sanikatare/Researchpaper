"""Settings — Consolidated API connection, live verification, and runtime parameters."""

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
from dataset.aistudio_workflow import write_integration_validation_report
from dataset.scenario_manager import resolve_image_path
from experiment.hydrogen_client import (
    API_ERROR_LOG_PATH,
    generate_api_readiness_report,
    get_sdk_version,
    is_api_fully_validated,
    load_validation_state,
    test_all_thinking_levels,
    verify_model_endpoint,
)
from experiment.storage import load_runtime_config, save_runtime_config


def render() -> None:
    render_page_header(
        "Settings",
        "Check API credentials, verify thinking-level support on Gemini 4 Hydrogen, and configure runtime limits.",
    )

    api_key = get_api_key()
    sdk_ver = get_sdk_version()
    val_state = load_validation_state()
    levels = val_state.get("levels", {})
    mm_probe = val_state.get("multimodal_probe", {})
    validated_count = sum(1 for v in levels.values() if v.get("validated"))

    conn_status = (
        "Connected (4/4 verified)"
        if is_api_fully_validated()
        else (
            f"Partially verified ({validated_count}/4)"
            if validated_count > 0
            else ("Key configured (unverified)" if api_key else "Not connected")
        )
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("API Connection", conn_status)
    with c2:
        render_metric_card("API Key Status", mask_api_key(api_key), "Read from environment / .env")
    with c3:
        render_metric_card("Model Identifier", DEFAULT_MODEL_ID, "Exclusive experimental model")
    with c4:
        render_metric_card("SDK Version", f"google-genai {sdk_ver}")

    if not api_key:
        st.info(
            "GEMINI_API_KEY is not set in the runtime environment. Add GEMINI_API_KEY to your "
            "environment variables or .env file before running live verification."
        )

    st.divider()
    st.subheader("Live API & Thinking-Level Verification")
    st.caption(
        "Verification sends real requests to the Gemini 4 Hydrogen endpoint. "
        "Unsupported or unverified thinking levels are never simulated."
    )

    include_multimodal = st.checkbox(
        "Also test image-plus-text attachment during verification (dataset/images/scn-001.png)",
        value=True,
        key="settings_include_mm",
    )

    cfg = load_runtime_config()
    if st.button("Verify Gemini API", type="primary", key="settings_verify_btn"):
        with st.spinner(f"Verifying {DEFAULT_MODEL_ID} endpoint and all 4 thinking levels..."):
            ep = verify_model_endpoint(
                model_id=DEFAULT_MODEL_ID,
                timeout_seconds=min(30.0, float(cfg.timeout_seconds)),
            )
            probe_img = resolve_image_path("scn-001.png") if include_multimodal else None
            results = test_all_thinking_levels(
                model_id=DEFAULT_MODEL_ID,
                max_output_tokens=min(1024, int(cfg.max_output_tokens)),
                timeout_seconds=float(cfg.timeout_seconds),
                max_retries=int(cfg.max_retries),
                image_path=probe_img,
            )
            generate_api_readiness_report(API_READINESS_REPORT_MD, endpoint_check=ep)
        st.session_state["settings_last_verify"] = {
            "endpoint": ep,
            "results": [
                {
                    "level": r.thinking_level,
                    "status": r.status,
                    "latency_ms": r.latency_ms,
                    "model": r.actual_model_version,
                    "error_code": r.error_code,
                    "error_message": r.error_message,
                }
                for r in results
            ],
        }
        st.rerun()

    last_verify = st.session_state.get("settings_last_verify")
    if last_verify:
        ep = last_verify.get("endpoint", {})
        if ep.get("accessible"):
            st.success(f"Endpoint reachable: `{ep.get('endpoint_name')}`.")
        else:
            st.error(f"Endpoint check failed: {ep.get('error')}")
        for r in last_verify.get("results", []):
            if r["status"] == "success":
                st.write(
                    f"**{r['level']}:** Supported · {r['latency_ms']:.0f} ms · "
                    f"Model: `{r['model']}`"
                )
            else:
                st.error(
                    f"**{r['level']}:** Unavailable ({r['error_code']}) — {r['error_message']}"
                )

    rows = []
    for lvl in THINKING_LEVELS:
        info = levels.get(lvl, {})
        sup = info.get("supported_by_endpoint")
        sup_lbl = "Supported" if sup is True else ("Unsupported" if sup is False else "Untested")
        rows.append(
            {
                "Thinking Level": lvl,
                "Validated": "Yes" if info.get("validated") else "No",
                "Endpoint Support": sup_lbl,
                "Modality Tested": info.get("modality_tested") or "None",
                "Latency (ms)": round(info["latency_ms"], 1) if info.get("latency_ms") is not None else None,
                "Last Tested (UTC)": info.get("last_tested_utc") or "Never",
                "Error": info.get("error") or "",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    mm_status = "Validated" if mm_probe.get("validated") else "Not validated"
    st.caption(
        f"Multimodal validation status: {mm_status} "
        f"(Last tested: {mm_probe.get('last_tested_utc') or 'Never'})"
    )

    st.divider()
    st.subheader("Runtime Parameters")
    with st.form("settings_runtime_form"):
        p1, p2 = st.columns(2)
        with p1:
            max_output_tokens = st.number_input(
                "Output token limit",
                min_value=64,
                max_value=65536,
                value=int(cfg.max_output_tokens),
                step=256,
            )
            timeout_seconds = st.number_input(
                "Request timeout (seconds)",
                min_value=5.0,
                max_value=600.0,
                value=float(cfg.timeout_seconds),
                step=5.0,
            )
        with p2:
            max_retries = st.number_input(
                "Maximum retries",
                min_value=0,
                max_value=10,
                value=int(cfg.max_retries),
                step=1,
            )
            min_request_interval = st.number_input(
                "Rate-limit pacing interval (seconds)",
                min_value=0.0,
                max_value=60.0,
                value=float(getattr(cfg, "min_request_interval", 0.0)),
                step=0.25,
            )

        with st.expander("Advanced Settings", expanded=False):
            adv1, adv2 = st.columns(2)
            with adv1:
                retry_base_delay = st.number_input(
                    "Retry base delay (seconds)",
                    min_value=0.5,
                    max_value=30.0,
                    value=float(cfg.retry_base_delay),
                    step=0.5,
                )
            with adv2:
                include_thoughts = st.checkbox(
                    "Request thought summaries when supported",
                    value=bool(cfg.include_thoughts),
                )

        if st.form_submit_button("Save Settings", type="primary"):
            cfg.max_output_tokens = int(max_output_tokens)
            cfg.timeout_seconds = float(timeout_seconds)
            cfg.max_retries = int(max_retries)
            cfg.min_request_interval = float(min_request_interval)
            cfg.retry_base_delay = float(retry_base_delay)
            cfg.include_thoughts = bool(include_thoughts)
            errs = cfg.validate()
            if errs:
                st.error("; ".join(errs))
            else:
                save_runtime_config(cfg)
                st.success("Runtime settings saved.")

    with st.expander("Diagnostics, Reports & Error Log", expanded=False):
        d1, d2 = st.columns(2)
        with d1:
            if API_READINESS_REPORT_MD.exists():
                st.download_button(
                    "Download API Readiness Report (MD)",
                    data=API_READINESS_REPORT_MD.read_text(encoding="utf-8"),
                    file_name="api_readiness_report.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
        with d2:
            if st.button("Generate AI Studio Integration Audit (MD)", use_container_width=True):
                audit_path = write_integration_validation_report()
                st.success(f"Updated `{audit_path.name}`.")

        if API_ERROR_LOG_PATH.exists():
            lines = API_ERROR_LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
            st.code("\n".join(lines[-50:]) if lines else "Error log is empty.", language="text")
        else:
            st.caption("No API errors recorded.")
