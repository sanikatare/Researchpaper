"""Shared academic UI styling and reusable dashboard components."""

from __future__ import annotations

from typing import Optional

import streamlit as st

ACADEMIC_CSS = """
<style>
    /* Restrained Academic Research Theme */
    .stApp {
        background-color: #F7F8FA;
        color: #111827;
        font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    section[data-testid="stSidebar"] {
        background-color: #151C28;
        border-right: 1px solid #263143;
    }
    section[data-testid="stSidebar"] * {
        color: #E5E7EB !important;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        background-color: #1E293B;
        border-radius: 4px;
    }
    .research-header {
        border-bottom: 2px solid #CBD5E1;
        padding-bottom: 0.75rem;
        margin-bottom: 1.25rem;
    }
    .research-title {
        font-size: 1.45rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.3;
        margin: 0;
    }
    .research-subtitle {
        font-size: 0.9rem;
        color: #475569;
        margin-top: 0.35rem;
    }
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748B;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0F172A;
        font-variant-numeric: tabular-nums;
    }
    .metric-sub {
        font-size: 0.76rem;
        color: #64748B;
        margin-top: 0.2rem;
    }
    .status-badge-ok {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        background-color: #DCFCE7;
        color: #166534;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .status-badge-warn {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        background-color: #FEF3C7;
        color: #92400E;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .status-badge-err {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        background-color: #FEE2E2;
        color: #991B1B;
        font-size: 0.78rem;
        font-weight: 600;
    }
</style>
"""


def inject_academic_css() -> None:
    """Apply restrained academic styling once per page run."""
    st.markdown(ACADEMIC_CSS, unsafe_allow_html=True)


def render_page_header(title: str, subtitle: str = "") -> None:
    """Render clean academic section header."""
    sub_html = f'<div class="research-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="research-header"><div class="research-title">{title}</div>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, subtext: str = "") -> None:
    """Render compact metric card."""
    sub_html = f'<div class="metric-sub">{subtext}</div>' if subtext else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_rate(rate: Optional[float], digits: int = 1) -> str:
    """Format proportion safely without turning missing data into 0%."""
    if rate is None:
        return "Unavailable"
    return f"{rate * 100:.{digits}f}%"


def format_number(val: Optional[float], digits: int = 1, suffix: str = "") -> str:
    """Format numeric metric safely without turning missing data into 0."""
    if val is None:
        return "Unavailable"
    return f"{val:.{digits}f}{suffix}"
