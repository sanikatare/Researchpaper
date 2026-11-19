"""Google-inspired light theme styling, typography, and reusable research UI components."""

from __future__ import annotations

from typing import Optional

import streamlit as st

GOOGLE_CHART_SEQUENCE = ["#4285F4", "#34A853", "#FBBC04", "#EA4335"]

GOOGLE_LIGHT_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700&family=Roboto:wght@400;500;700&display=swap');

    /* Enforce Light Mode Workspace */
    .stApp {
        background-color: #F8F9FA;
        color: #202124;
        font-family: 'Roboto', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    h1, h2, h3, .research-title {
        font-family: 'Plus Jakarta Sans', 'Roboto', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #202124;
    }

    /* Light Sidebar with Google Blue Active State */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #DADCE0;
    }
    section[data-testid="stSidebar"] * {
        color: #202124;
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
        padding: 0.42rem 0.65rem;
        border-radius: 0 20px 20px 0;
        margin-bottom: 2px;
        transition: background-color 0.12s ease;
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
        background-color: #F1F3F4;
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:has(input:checked) {
        background-color: #E8F0FE;
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:has(input:checked) p {
        color: #1967D2 !important;
        font-weight: 600 !important;
    }

    /* Buttons */
    .stButton > button[kind="primary"] {
        background-color: #4285F4;
        border-color: #4285F4;
        color: #FFFFFF;
        border-radius: 6px;
        font-weight: 500;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #3367D6;
        border-color: #3367D6;
    }
    .stButton > button, .stDownloadButton > button {
        border-radius: 6px;
        border: 1px solid #DADCE0;
        background-color: #FFFFFF;
        color: #202124;
        font-weight: 500;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background-color: #F1F3F4;
        border-color: #C6C9CE;
    }

    /* Clean Header */
    .research-header {
        background-color: #FFFFFF;
        border: 1px solid #DADCE0;
        border-top: 4px solid #4285F4;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 2px rgba(60, 64, 67, 0.08);
    }
    .research-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #202124;
        line-height: 1.3;
        margin: 0;
    }
    .research-subtitle {
        font-size: 0.9rem;
        color: #5F6368;
        margin-top: 0.3rem;
    }

    /* Compact Rounded Metric Cards */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #DADCE0;
        border-radius: 8px;
        padding: 0.9rem 1.05rem;
        margin-bottom: 0.6rem;
        box-shadow: 0 1px 2px rgba(60, 64, 67, 0.06);
    }
    .metric-label {
        font-size: 0.74rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #5F6368;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #202124;
        font-variant-numeric: tabular-nums;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #5F6368;
        margin-top: 0.22rem;
    }

    /* Google Accent Pills / Badges */
    .badge-blue {
        display: inline-block;
        padding: 2px 9px;
        border-radius: 12px;
        background-color: #E8F0FE;
        color: #1967D2;
        font-size: 0.76rem;
        font-weight: 600;
    }
    .badge-green {
        display: inline-block;
        padding: 2px 9px;
        border-radius: 12px;
        background-color: #E6F4EA;
        color: #137333;
        font-size: 0.76rem;
        font-weight: 600;
    }
    .badge-yellow {
        display: inline-block;
        padding: 2px 9px;
        border-radius: 12px;
        background-color: #FEF7E0;
        color: #B06000;
        font-size: 0.76rem;
        font-weight: 600;
    }
    .badge-red {
        display: inline-block;
        padding: 2px 9px;
        border-radius: 12px;
        background-color: #FCE8E6;
        color: #C5221F;
        font-size: 0.76rem;
        font-weight: 600;
    }

    /* Surface Containers */
    .surface-card {
        background-color: #FFFFFF;
        border: 1px solid #DADCE0;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.85rem;
        box-shadow: 0 1px 2px rgba(60, 64, 67, 0.05);
    }
</style>
"""


def inject_light_theme_css() -> None:
    """Apply Google-inspired light Workspace CSS once per render."""
    st.markdown(GOOGLE_LIGHT_CSS, unsafe_allow_html=True)


def render_page_header(title: str, subtitle: str = "") -> None:
    """Render clean light surface header with Google Blue top accent."""
    sub_html = f'<div class="research-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="research-header"><div class="research-title">{title}</div>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def render_metric_card(
    label: str,
    value: str,
    subtext: str = "",
    accent_color: str = "#DADCE0",
) -> None:
    """Render compact light metric card with optional subtle left accent."""
    sub_html = f'<div class="metric-sub">{subtext}</div>' if subtext else ""
    border_left = (
        f"border-left: 3px solid {accent_color};" if accent_color != "#DADCE0" else ""
    )
    st.markdown(
        f"""
        <div class="metric-card" style="{border_left}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_badge(text: str, tone: str = "blue") -> str:
    """Return HTML badge using Google light tint palette."""
    cls = {
        "blue": "badge-blue",
        "green": "badge-green",
        "yellow": "badge-yellow",
        "red": "badge-red",
    }.get(tone, "badge-blue")
    return f'<span class="{cls}">{text}</span>'


def style_plotly_light(fig):
    """Apply clean Google-inspired light layout to any Plotly figure."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Roboto, sans-serif", color="#202124", size=12),
        title_font=dict(family="Plus Jakarta Sans, Roboto, sans-serif", color="#202124", size=15),
        legend=dict(bgcolor="#FFFFFF", bordercolor="#DADCE0", borderwidth=1),
        margin=dict(l=40, r=24, t=48, b=40),
    )
    fig.update_xaxes(showgrid=False, linecolor="#DADCE0", tickfont=dict(color="#5F6368"))
    fig.update_yaxes(gridcolor="#F1F3F4", linecolor="#DADCE0", tickfont=dict(color="#5F6368"))
    return fig


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
