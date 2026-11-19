"""Streamlit entry point for the Gemini 4 Hydrogen Safety Research Workbench."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from app.components import inject_academic_css
from app.navigation import PAGES
from config import DEFAULT_MODEL_ID, ensure_directories
from experiment.hydrogen_client import is_api_fully_validated


def main() -> None:
    ensure_directories()
    st.set_page_config(
        page_title="Gemini 4 Hydrogen Safety Research",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_academic_css()

    with st.sidebar:
        st.markdown("### Gemini 4 Hydrogen")
        st.caption("Cross-Modal Safety & Thinking-Level Study")
        st.divider()
        selected_page = st.radio(
            "Research Workflow",
            list(PAGES.keys()),
            label_visibility="collapsed",
        )
        st.divider()
        api_ok = is_api_fully_validated()
        st.caption(f"Model: `{DEFAULT_MODEL_ID}`")
        if api_ok:
            st.success("API: 4/4 Levels Validated")
        else:
            st.warning("API Validation Incomplete")

    PAGES[selected_page]()


if __name__ == "__main__":
    main()
