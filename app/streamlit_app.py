"""Streamlit entry point with black top navigation bar for the Gemini 4 Hydrogen Research Workbench."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from app.components import inject_theme_css
from app.navigation import PAGES
from config import ensure_directories


def _render_top_navbar(active_page: str) -> str:
    """Render solid black top navigation bar and return selected page."""
    with st.container(key="top_navbar"):
        nav_l, nav_c = st.columns([2.2, 6.8], vertical_alignment="center")
        with nav_l:
            st.markdown(
                '<div class="navbar-brand">Gemini 4 Hydrogen</div>',
                unsafe_allow_html=True,
            )
        with nav_c:
            with st.container(horizontal=True, gap="small", horizontal_alignment="right"):
                for page_name in PAGES:
                    is_active = page_name == active_page
                    if st.button(
                        page_name,
                        key=f"top_nav_{page_name}",
                        type="primary" if is_active else "secondary",
                    ):
                        if not is_active:
                            st.session_state["active_page"] = page_name
                            st.rerun()

    return st.session_state.get("active_page", active_page)


def main() -> None:
    ensure_directories()
    st.set_page_config(
        page_title="Gemini 4 Hydrogen Research",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_theme_css()

    if "active_page" not in st.session_state or st.session_state["active_page"] not in PAGES:
        st.session_state["active_page"] = "Overview"

    active_page = _render_top_navbar(st.session_state["active_page"])
    PAGES[active_page]()


if __name__ == "__main__":
    main()
