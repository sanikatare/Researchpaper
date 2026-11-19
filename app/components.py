"""Minimal black-and-white academic styling, black top navbar CSS, and reusable UI helpers."""

from __future__ import annotations

import html
from typing import Optional

import streamlit as st

MONOCHROME_CHART_SEQUENCE = ["#171717", "#525252", "#8C8C8C", "#C2C2C2"]

MONOCHROME_ACADEMIC_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,500;0,8..60,600;0,8..60,700;1,8..60,400&display=swap');

    /* Primary Canvas & Typography */
    html, body, .stApp {
        background-color: #FFFFFF;
        color: #171717;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    /* Hide default Streamlit sidebar & header chrome so the black top navbar leads the app */
    section[data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    header[data-testid="stHeader"],
    #MainMenu,
    footer {
        display: none !important;
    }

    .main .block-container {
        max-width: 1120px;
        padding-top: 0 !important;
        padding-bottom: 3.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    h1, h2, h3, .research-title, .serif-heading {
        font-family: 'Source Serif 4', 'Libre Baskerville', Georgia, serif;
        color: #171717;
        letter-spacing: -0.015em;
    }

    p, li, label, .stMarkdown, .stText, .stCaption {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        color: #171717;
    }

    /* Full-Bleed Solid Black Top Navbar */
    .st-key-top_navbar {
        background-color: #111111 !important;
        color: #FFFFFF !important;
        box-shadow: 0 0 0 100vmax #111111;
        clip-path: inset(0 -100vmax);
        padding: 0.75rem 0 0.75rem 0 !important;
        margin-bottom: 2rem !important;
        border-bottom: 1px solid #262626;
    }
    .st-key-top_navbar p,
    .st-key-top_navbar span,
    .st-key-top_navbar div {
        color: #FFFFFF;
    }
    .navbar-brand {
        font-family: 'Source Serif 4', 'Libre Baskerville', Georgia, serif;
        font-size: 1.12rem;
        font-weight: 600;
        color: #FFFFFF !important;
        letter-spacing: -0.01em;
        white-space: nowrap;
        line-height: 1.2;
    }
    .navbar-status {
        text-align: right;
        font-size: 0.8rem;
        color: #D4D4D4 !important;
        white-space: nowrap;
        font-variant-numeric: tabular-nums;
    }

    /* Navbar Navigation Buttons */
    .st-key-top_navbar .stButton > button[kind="secondary"] {
        background-color: transparent !important;
        border: 1px solid transparent !important;
        color: #BDBDBD !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        padding: 0.36rem 0.72rem !important;
        border-radius: 6px !important;
        white-space: nowrap !important;
        min-height: 0 !important;
    }
    .st-key-top_navbar .stButton > button[kind="secondary"]:hover {
        background-color: #262626 !important;
        color: #FFFFFF !important;
        border-color: #262626 !important;
    }
    .st-key-top_navbar .stButton > button[kind="primary"] {
        background-color: #FFFFFF !important;
        border: 1px solid #FFFFFF !important;
        color: #111111 !important;
        font-size: 0.875rem !important;
        font-weight: 600 !important;
        padding: 0.36rem 0.72rem !important;
        border-radius: 6px !important;
        white-space: nowrap !important;
        min-height: 0 !important;
    }
    .st-key-top_navbar .stButton > button[kind="primary"]:hover {
        background-color: #F3F3F1 !important;
        border-color: #F3F3F1 !important;
        color: #111111 !important;
    }
    .st-key-top_navbar .stButton > button[kind="primary"] p {
        color: #111111 !important;
    }
    .st-key-top_navbar .stButton > button[kind="secondary"] p {
        color: inherit !important;
    }

    /* Primary & Secondary Page Buttons */
    .stButton > button[kind="primary"],
    .stFormSubmitButton > button[kind="primary"] {
        background-color: #171717 !important;
        border: 1px solid #171717 !important;
        color: #FFFFFF !important;
        border-radius: 6px;
        font-weight: 500;
        transition: background-color 0.15s ease;
    }
    .stButton > button[kind="primary"] p,
    .stFormSubmitButton > button[kind="primary"] p {
        color: #FFFFFF !important;
    }
    .stButton > button[kind="primary"]:hover,
    .stFormSubmitButton > button[kind="primary"]:hover {
        background-color: #2E2E2E !important;
        border-color: #2E2E2E !important;
        color: #FFFFFF !important;
    }
    .stButton > button[kind="primary"]:disabled,
    .stFormSubmitButton > button[kind="primary"]:disabled {
        background-color: #E5E5E5 !important;
        border-color: #E5E5E5 !important;
        color: #737373 !important;
    }
    .stButton > button[kind="primary"]:disabled p,
    .stFormSubmitButton > button[kind="primary"]:disabled p {
        color: #737373 !important;
    }

    .stButton > button,
    .stDownloadButton > button,
    .stFormSubmitButton > button {
        border-radius: 6px;
        border: 1px solid #E5E5E5;
        background-color: #FFFFFF;
        color: #171717;
        font-weight: 500;
        transition: background-color 0.15s ease, border-color 0.15s ease;
    }
    .stButton > button:hover,
    .stDownloadButton > button:hover,
    .stFormSubmitButton > button:hover {
        background-color: #F3F3F1;
        border-color: #D4D4D4;
        color: #171717;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem;
        border-bottom: 1px solid #E5E5E5;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #737373;
        font-weight: 500;
        font-size: 0.9rem;
        padding: 0.55rem 0.85rem;
        border-bottom: 2px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #171717 !important;
        border-bottom: 2px solid #171717 !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #171717 !important;
    }

    /* Inputs, Textareas, Selectboxes */
    .stTextInput input,
    .stTextArea textarea,
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border-color: #E5E5E5 !important;
        color: #171717 !important;
        border-radius: 6px !important;
    }

    /* Neutral Alerts */
    div[data-testid="stAlert"] {
        background-color: #F7F7F5 !important;
        border: 1px solid #E5E5E5 !important;
        color: #171717 !important;
        border-radius: 6px !important;
    }
    div[data-testid="stAlert"] * {
        color: #171717 !important;
    }

    /* Page Header */
    .research-header {
        border-bottom: 1px solid #E5E5E5;
        padding-bottom: 1.05rem;
        margin-bottom: 1.6rem;
    }
    .research-title {
        font-size: 1.95rem;
        font-weight: 600;
        color: #171717;
        line-height: 1.2;
        margin: 0;
    }
    .research-subtitle {
        font-size: 0.95rem;
        color: #737373;
        margin-top: 0.45rem;
        line-height: 1.5;
        max-width: 72ch;
    }

    /* Clean Stat Strip */
    .stat-box {
        background-color: #F7F7F5;
        border: 1px solid #E5E5E5;
        border-radius: 8px;
        padding: 1rem 1.15rem;
        margin-bottom: 0.75rem;
    }
    .stat-label {
        font-size: 0.78rem;
        font-weight: 500;
        color: #737373;
        margin-bottom: 0.35rem;
    }
    .stat-value {
        font-size: 1.38rem;
        font-weight: 600;
        color: #171717;
        font-variant-numeric: tabular-nums;
        line-height: 1.25;
    }
    .stat-sub {
        font-size: 0.78rem;
        color: #737373;
        margin-top: 0.3rem;
        line-height: 1.4;
    }

    /* Study Architecture Cards */
    .study-pillar {
        background-color: #FFFFFF;
        border: 1px solid #E5E5E5;
        border-radius: 8px;
        padding: 1.05rem 1.15rem;
        height: 100%;
    }
    .study-pillar-title {
        font-family: 'Source Serif 4', 'Libre Baskerville', Georgia, serif;
        font-size: 1.02rem;
        font-weight: 600;
        color: #171717;
        margin-bottom: 0.35rem;
    }
    .study-pillar-body {
        font-size: 0.87rem;
        color: #737373;
        line-height: 1.5;
    }

    /* Progress Step List */
    .progress-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.78rem 0;
        border-bottom: 1px solid #E5E5E5;
        gap: 1rem;
    }
    .progress-row:last-child {
        border-bottom: none;
    }
    .progress-step-name {
        font-size: 0.95rem;
        font-weight: 500;
        color: #171717;
    }
    .progress-step-detail {
        font-size: 0.82rem;
        color: #737373;
        margin-top: 0.15rem;
    }
    .progress-step-status {
        font-size: 0.85rem;
        font-weight: 500;
        color: #171717;
        white-space: nowrap;
        font-variant-numeric: tabular-nums;
    }

    /* Next Step Callout */
    .next-step-panel {
        background-color: #F7F7F5;
        border: 1px solid #E5E5E5;
        border-radius: 8px;
        padding: 1.05rem 1.25rem;
        margin-top: 1rem;
        margin-bottom: 0.75rem;
    }
    .next-step-kicker {
        font-size: 0.78rem;
        font-weight: 600;
        color: #737373;
        margin-bottom: 0.25rem;
    }
    .next-step-body {
        font-size: 0.98rem;
        color: #171717;
        line-height: 1.5;
    }

    /* Experiment Lab Chat Bubbles */
    .chat-wrap {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
        margin-bottom: 0.5rem;
    }
    .chat-row-user {
        display: flex;
        justify-content: flex-end;
    }
    .chat-bubble-user {
        max-width: 78%;
        background-color: #171717;
        color: #FFFFFF;
        padding: 0.85rem 1.1rem;
        border-radius: 10px 10px 2px 10px;
        font-size: 0.94rem;
        line-height: 1.55;
        white-space: pre-wrap;
    }
    .chat-bubble-user * {
        color: #FFFFFF !important;
    }
    .chat-meta-user {
        text-align: right;
        font-size: 0.76rem;
        color: #737373;
        margin-top: 0.25rem;
    }
    .chat-row-assistant {
        display: flex;
        justify-content: flex-start;
    }
    .chat-bubble-assistant {
        max-width: 84%;
        background-color: #F7F7F5;
        border: 1px solid #E5E5E5;
        color: #171717;
        padding: 0.95rem 1.15rem;
        border-radius: 10px 10px 10px 2px;
        font-size: 0.94rem;
        line-height: 1.6;
        white-space: pre-wrap;
    }
    .chat-meta-assistant {
        font-size: 0.76rem;
        color: #737373;
        margin-top: 0.35rem;
        font-variant-numeric: tabular-nums;
    }
    .chat-error-box {
        max-width: 84%;
        background-color: #F3F3F1;
        border: 1px solid #171717;
        color: #171717;
        padding: 0.85rem 1.05rem;
        border-radius: 8px;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    /* Empty State & Section Framing */
    .empty-state-box {
        background-color: #F7F7F5;
        border: 1px solid #E5E5E5;
        border-radius: 8px;
        padding: 1.85rem 1.5rem;
        margin: 0.85rem 0;
    }
    .empty-state-title {
        font-family: 'Source Serif 4', 'Libre Baskerville', Georgia, serif;
        font-size: 1.18rem;
        font-weight: 600;
        color: #171717;
        margin-bottom: 0.4rem;
    }
    .empty-state-desc {
        font-size: 0.92rem;
        color: #737373;
        line-height: 1.55;
        max-width: 66ch;
    }

    /* Progress Bar Monochrome */
    .stProgress > div > div > div > div {
        background-color: #171717 !important;
    }
    .stProgress > div > div > div {
        background-color: #E5E5E5 !important;
    }

    /* Hairline Section Divider */
    hr {
        border-color: #E5E5E5 !important;
        margin-top: 1.4rem !important;
        margin-bottom: 1.4rem !important;
    }
</style>
"""


def inject_theme_css() -> None:
    """Apply minimal black-and-white academic CSS and black top navbar styling."""
    st.markdown(MONOCHROME_ACADEMIC_CSS, unsafe_allow_html=True)


def navigate_to(page_name: str) -> None:
    """Switch active top-navbar page cleanly and trigger a rerun."""
    st.session_state["active_page"] = page_name
    st.rerun()


def render_page_header(title: str, subtitle: str = "") -> None:
    """Render minimal serif page header with quiet sans-serif description."""
    safe_title = html.escape(title)
    sub_html = (
        f'<div class="research-subtitle">{html.escape(subtitle)}</div>' if subtitle else ""
    )
    st.markdown(
        f'<div class="research-header"><h1 class="research-title">{safe_title}</h1>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def render_metric_card(
    label: str,
    value: str,
    subtext: str = "",
) -> None:
    """Render minimal monochrome statistic box with tabular numerals."""
    safe_label = html.escape(label)
    safe_value = html.escape(value)
    sub_html = f'<div class="stat-sub">{html.escape(subtext)}</div>' if subtext else ""
    st.markdown(
        f"""
        <div class="stat-box">
            <div class="stat-label">{safe_label}</div>
            <div class="stat-value">{safe_value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_study_pillar(title: str, body: str) -> None:
    """Render concise editorial card explaining one pillar of the research project."""
    st.markdown(
        f"""
        <div class="study-pillar">
            <div class="study-pillar-title">{html.escape(title)}</div>
            <div class="study-pillar-body">{html.escape(body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(title: str, body: str, detail: str = "") -> None:
    """Render calm, legible empty state when experimental records are unavailable."""
    detail_html = (
        f'<div class="empty-state-desc" style="margin-top: 0.65rem;">{html.escape(detail)}</div>'
        if detail
        else ""
    )
    st.markdown(
        f"""
        <div class="empty-state-box">
            <div class="empty-state-title">{html.escape(title)}</div>
            <div class="empty-state-desc">{html.escape(body)}</div>
            {detail_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_plotly_monochrome(fig):
    """Apply minimal black-and-white academic layout to any Plotly figure."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", color="#171717", size=12),
        title_font=dict(
            family="Source Serif 4, Libre Baskerville, Georgia, serif",
            color="#171717",
            size=15,
        ),
        legend=dict(bgcolor="#FFFFFF", bordercolor="#E5E5E5", borderwidth=1),
        margin=dict(l=40, r=24, t=44, b=40),
    )
    fig.update_xaxes(showgrid=False, linecolor="#E5E5E5", tickfont=dict(color="#737373"))
    fig.update_yaxes(gridcolor="#F3F3F1", linecolor="#E5E5E5", tickfont=dict(color="#737373"))
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
