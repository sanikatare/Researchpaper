"""Streamlit entry point for the Gemini 4 Hydrogen Safety Research Workbench."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from app.components import inject_light_theme_css, render_status_badge
from app.navigation import PAGES
from config import DEFAULT_MODEL_ID, ensure_directories
from dataset.scenario_manager import load_registry
from experiment.hydrogen_client import is_api_fully_validated


def main() -> None:
    ensure_directories()
    st.set_page_config(
        page_title="Gemini 4 Hydrogen Safety Study",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_light_theme_css()

    with st.sidebar:
        st.markdown("### Safety Research Workbench")
        st.caption("Cross-Modal Thinking-Level Evaluation")
        st.divider()
        selected_page = st.radio(
            "Navigation",
            list(PAGES.keys()),
            label_visibility="collapsed",
        )
        st.divider()
        api_ok = is_api_fully_validated()
        reg = load_registry()
        st.caption(f"Model: `{DEFAULT_MODEL_ID}`")
        st.caption(f"Pilot Dataset: `{reg.get('active_pilot_version') or 'None'}`")
        st.caption(f"Main Dataset: `{reg.get('active_main_version') or 'None'}`")
        if api_ok:
            st.markdown(
                render_status_badge("API: 4/4 Levels Validated", "green"),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                render_status_badge("API Validation Incomplete", "yellow"),
                unsafe_allow_html=True,
            )

    PAGES[selected_page]()


if __name__ == "__main__":
    main()
