"""
FinSight-Alpha — Shared design tokens and CSS injection.
Import this in every page to ensure consistent styling.
"""

# Colour palette
BG          = "#070810"
SURFACE     = "#0D0F1A"
SURFACE2    = "#13162A"
SURFACE3    = "#1A1E35"
BORDER      = "rgba(100,120,255,0.12)"
PRIMARY     = "#6C7FFF"
PRIMARY_DIM = "rgba(108,127,255,0.15)"
ACCENT      = "#F0B429"
ACCENT_DIM  = "rgba(240,180,41,0.15)"
GREEN       = "#2ECC8A"
RED         = "#F05252"
PURPLE      = "#C77DFF"
PINK        = "#FF7A7A"
TEAL        = "#4ADDA0"
TEXT        = "#E8EAF6"
TEXT_MUTED  = "#7B82A8"
TEXT_DIM    = "#4A5080"

MONO    = "Space Mono, Courier New, monospace"
SANS    = "DM Sans, Segoe UI, sans-serif"
DISPLAY = "Syne, Arial Black, sans-serif"

# Global CSS
GLOBAL_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"] {{
    background-color: {BG} !important;
    color: {TEXT} !important;
    font-family: {SANS} !important;
}}

/* Hide default Streamlit chrome */
header {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
[data-testid="stSidebar"] > div:first-child {{
    background-color: {SURFACE} !important;
    border-right: 1px solid {BORDER} !important;
}}

/* Block container padding */
div.block-container {{
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    padding-left: 1.2rem !important;
    padding-right: 1.2rem !important;
    max-width: 1100px !important;
}}

/* Typography */
h1, h2, h3, h4, h5 {{
    color: {TEXT} !important;
    font-family: {DISPLAY} !important;
    font-weight: 700 !important;
    margin-top: 0 !important;
}}

/* Metrics */
[data-testid="stMetricValue"] {{
    font-family: {MONO} !important;
    font-size: 1.6rem !important;
    color: {TEXT} !important;
    line-height: 1.2 !important;
}}
[data-testid="stMetricLabel"] {{
    color: {TEXT_MUTED} !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
}}

/* Code blocks */
code, pre {{
    font-family: {MONO} !important;
    font-size: 0.78rem !important;
    background-color: {SURFACE2} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 4px !important;
    color: {ACCENT} !important;
}}

/* Sidebar nav text */
[data-testid="stSidebarNav"] a {{
    color: {TEXT_MUTED} !important;
    font-family: {MONO} !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.04em !important;
}}
[data-testid="stSidebarNav"] a:hover,
[data-testid="stSidebarNav"] a[aria-current="page"] {{
    color: {PRIMARY} !important;
    background: {PRIMARY_DIM} !important;
    border-radius: 6px !important;
}}

/* Plotly chart backgrounds */
.js-plotly-plot .plotly .bg {{ fill: transparent !important; }}

/* Scrollbar */
::-webkit-scrollbar {{ width: 4px; }}
::-webkit-scrollbar-track {{ background: {SURFACE}; }}
::-webkit-scrollbar-thumb {{ background: {PRIMARY}; border-radius: 2px; }}

/* Animations */
@keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(10px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50%       {{ opacity: 0.4; }}
}}
.page-enter {{ animation: fadeIn 0.4s ease forwards; }}
</style>
"""


def inject_css():
    """Call st.markdown(GLOBAL_CSS, unsafe_allow_html=True) via this helper."""
    import streamlit as st
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)