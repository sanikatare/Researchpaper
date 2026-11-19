"""Streamlit entry point for the Gemini 4 Hydrogen Cross-Modal Safety Research Workbench."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from app.components import inject_theme_css
from app.navigation import PAGES
from config import DEFAULT_MODEL_ID, ensure_directories
from dataset.scenario_manager import load_registry
from experiment.hydrogen_client import get_api_key, is_api_fully_validated


def main() -> None:
    ensure_directories()
    st.set_page_config(
        page_title="Gemini 4 Hydrogen Research",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_theme_css()

    with st.sidebar:
        st.markdown("### Gemini 4 Hydrogen")
        st.caption("Cross-Modal Safety Study")
        st.divider()
        selected_page = st.radio(
            "Navigation",
            list(PAGES.keys()),
            index=0,
            label_visibility="collapsed",
        )
        st.divider()
        api_ok = is_api_fully_validated()
        key_set = bool(get_api_key())
        reg = load_registry()
        api_label = (
            "Verified (4/4)"
            if api_ok
            else ("Key set (unverified)" if key_set else "Not connected")
        )
        st.caption(f"Model: `{DEFAULT_MODEL_ID}`")
        st.caption(f"API: {api_label}")
        st.caption(f"Pilot dataset: `{reg.get('active_pilot_version') or 'None'}`")
        st.caption(f"Main dataset: `{reg.get('active_main_version') or 'None'}`")

    PAGES[selected_page]()


if __name__ == "__main__":
    main()
