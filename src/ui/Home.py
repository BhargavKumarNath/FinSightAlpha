"""
FinSight-Alpha Dashboard — Home / Overview
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
from components.theme import inject_css, PRIMARY, ACCENT, GREEN, PURPLE, SURFACE, SURFACE2, BORDER, TEXT, TEXT_MUTED, TEXT_DIM, MONO, DISPLAY
from components.ui_components import (
    hero_banner, section_title, badge_html,
    capability_cards, tech_stack_grid, live_dot_html,
)
from components.data import FILING_TYPES, CAPABILITIES, TECH_STACK

st.set_page_config(
    page_title="FinSight-Alpha",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# Sidebar branding
with st.sidebar:
    st.markdown(
        f"<div style='padding:8px 4px 16px;'>"
        f"<div style='font-family:{DISPLAY};font-size:20px;font-weight:800;'>"
        f"<span style='color:{PRIMARY};'>Fin</span>"
        f"<span style='color:{TEXT};'>Sight</span>"
        f"<span style='color:{ACCENT};font-family:{MONO};font-size:13px;margin-left:2px;'>α</span>"
        f"</div>"
        f"<div style='font-family:{MONO};font-size:9px;color:{TEXT_DIM};letter-spacing:0.1em;margin-top:2px;'>"
        f"AGENTIC RAG PLATFORM</div>"
        f"<div style='margin-top:10px;display:flex;align-items:center;gap:6px;'>"
        f"{live_dot_html(GREEN)}"
        f"<span style='font-family:{MONO};font-size:9px;color:{GREEN};letter-spacing:0.08em;'>ALL SYSTEMS NOMINAL</span>"
        f"</div></div>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        f"<div style='font-family:{MONO};font-size:9px;color:{TEXT_DIM};line-height:1.8;padding:4px 0;'>"
        f"LangGraph · Qdrant · Groq<br>"
        f"RAGAS · FastAPI · Streamlit<br><br>"
        f"<span style='color:{TEXT_DIM};'>v2.0.0 · Bhargav Kumar Nath</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

# Hero
hero_banner(
    title_colored="FinSight",
    title_plain="-Alpha",
    subtitle=(
        "An institutional-grade Agentic RAG system purpose-built for SEC filings analysis. "
        "Ingests 10-K, 10-Q, and 8-K documents to answer complex multi-hop analyst questions "
        "with fully grounded, cited reasoning and auditable decision traces."
    ),
    badges=[
        ("LangGraph Agent", PRIMARY),
        ("Hybrid RAG", ACCENT),
        ("RAGAS Eval", GREEN),
        ("Cross-Encoder", PURPLE),
    ],
    metrics=[
        ("91%", "Faithfulness",  PRIMARY),
        ("90%", "Relevancy",     ACCENT),
        ("88%", "Live MRR",      GREEN),
        ("85%", "NDCG@5",        PURPLE),
    ],
)

st.markdown("<br>", unsafe_allow_html=True)

# Capabilities
section_title("🔬", "Core Capabilities", "What makes FinSight-Alpha different from a standard RAG pipeline.")
capability_cards(CAPABILITIES)

st.markdown("<br>", unsafe_allow_html=True)

# Supported data sources
section_title("📂", "Supported Data Sources")

cols = st.columns(4)
for col, f in zip(cols, FILING_TYPES):
    col.markdown(
        f"<div style='background:{SURFACE};border:1px solid {BORDER};"
        f"border-radius:10px;padding:18px;height:130px;'>"
        f"<div style='font-size:24px;margin-bottom:8px;'>{f['icon']}</div>"
        f"<div style='font-family:{MONO};font-size:11px;color:{f['color']};font-weight:700;margin-bottom:4px;'>{f['type']}</div>"
        f"<div style='color:{TEXT_MUTED};font-size:12px;line-height:1.5;'>{f['desc']}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Technology stack
section_title("🛠️", "Technology Stack", "Every component selected for production-grade performance.")
with st.container():
    st.markdown(
        f"<div style='background:{SURFACE};border:1px solid {BORDER};border-radius:10px;padding:20px;'>",
        unsafe_allow_html=True,
    )
    tech_stack_grid(TECH_STACK)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Quick-start
section_title("🚀", "Quick Start", "Get the full pipeline running locally in three steps.")

steps = [
    ("01", "Index your filings",
     "python -m src.ingestion.sec_scraper\npython -m src.ingestion.document_processor\npython -m src.retrieval.hybrid_retriever  # builds index"),
    ("02", "Start the API",
     "uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload"),
    ("03", "Launch this dashboard",
     "streamlit run src/ui/Home.py"),
]

for step, title, cmd in steps:
    st.markdown(
        f"<div style='background:{SURFACE};border:1px solid {BORDER};border-left:3px solid {PRIMARY};"
        f"border-radius:10px;padding:18px;margin-bottom:10px;display:flex;gap:18px;align-items:flex-start;'>"
        f"<div style='font-family:{DISPLAY};font-size:28px;font-weight:800;color:{PRIMARY}40;min-width:40px;'>{step}</div>"
        f"<div style='flex:1;'>"
        f"<div style='font-family:{DISPLAY};font-size:14px;font-weight:700;color:{TEXT};margin-bottom:8px;'>{title}</div>"
        f"<code style='display:block;white-space:pre;padding:10px;background:{SURFACE2};border-radius:6px;"
        f"font-size:11px;color:{ACCENT};border:1px solid {BORDER};'>{cmd}</code>"
        f"</div></div>",
        unsafe_allow_html=True,
    )