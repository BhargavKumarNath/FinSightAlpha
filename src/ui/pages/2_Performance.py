"""
FinSight-Alpha Dashboard — Performance Metrics page
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from components.theme import inject_css, PRIMARY, ACCENT, GREEN, RED, PURPLE, TEAL, PINK
from components.theme import SURFACE, SURFACE2, BORDER, TEXT, TEXT_MUTED, TEXT_DIM, MONO, DISPLAY
from components.ui_components import section_title, stat_metric, badge_html, info_table
from components.charts import (
    ragas_trend_chart, retrieval_comparison_chart,
    latency_breakdown_chart, quality_radar_chart,
)
from components.data import RAGAS_TREND, RETRIEVAL_COMPARISON, LATENCY_BREAKDOWN, QUALITY_RADAR

st.set_page_config(page_title="Performance · FinSight-Alpha", page_icon="📊", layout="wide")
inject_css()

section_title(
    "📊", "Performance Metrics",
    "RAGAS evaluation results, retrieval benchmarks, and latency profiling across pipeline iterations.",
)

# Top KPI row
kpi_cols = st.columns(6, gap="small")
kpis = [
    ("91%",  "Faithfulness",    PRIMARY),
    ("90%",  "Answer Relevancy",ACCENT),
    ("88%",  "Live MRR",        GREEN),
    ("85%",  "NDCG@5",          PURPLE),
    ("2.1s", "Median RTT",      PINK),
    ("94%",  "Cache Hit Rate",  TEAL),
]
for col, (val, label, color) in zip(kpi_cols, kpis):
    stat_metric(col, val, label, color)

st.markdown("<br>", unsafe_allow_html=True)

# Charts row 1
col_l, col_r = st.columns(2, gap="large")

with col_l:
    st.markdown(
        f"<div style='background:{SURFACE};border:1px solid {BORDER};border-radius:10px;padding:18px;'>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(ragas_trend_chart(RAGAS_TREND), use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

with col_r:
    st.markdown(
        f"<div style='background:{SURFACE};border:1px solid {BORDER};border-radius:10px;padding:18px;'>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(retrieval_comparison_chart(RETRIEVAL_COMPARISON), use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Charts row 2
col_l2, col_r2 = st.columns(2, gap="large")

with col_l2:
    st.markdown(
        f"<div style='background:{SURFACE};border:1px solid {BORDER};border-radius:10px;padding:18px;'>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(latency_breakdown_chart(LATENCY_BREAKDOWN), use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

with col_r2:
    st.markdown(
        f"<div style='background:{SURFACE};border:1px solid {BORDER};border-radius:10px;padding:18px;'>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(quality_radar_chart(QUALITY_RADAR), use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Detailed retrieval table
section_title("🔬", "Retrieval Stack — Detailed Breakdown")

st.markdown(
    f"<div style='background:{SURFACE};border:1px solid {BORDER};border-radius:10px;padding:20px;'>",
    unsafe_allow_html=True,
)

def f1_color(val):
    try:
        v = float(val.rstrip("%")) / 100
        return GREEN if v > 0.85 else TEXT

    except Exception:
        return TEXT

rows = []
for i, r in enumerate(RETRIEVAL_COMPARISON):
    is_active = i == len(RETRIEVAL_COMPARISON) - 1
    status = badge_html("ACTIVE", GREEN) if is_active else badge_html("BASELINE", TEXT_MUTED)
    method_color = PRIMARY if is_active else TEXT
    rows.append([
        f"<span style='color:{method_color};font-family:{MONO};font-size:11px;'>{r['method']}</span>",
        f"{r['precision']*100:.0f}%",
        f"{r['recall']*100:.0f}%",
        f"<b>{r['f1']*100:.0f}%</b>" if is_active else f"{r['f1']*100:.0f}%",
        f"{r['latency']}s",
        status,
    ])

info_table(
    headers=["Method", "Precision", "Recall", "F1 Score", "Latency", "Status"],
    rows=rows,
    col_colors={
        1: TEXT_MUTED,
        2: TEXT_MUTED,
        3: lambda v: GREEN if "<b>" in str(v) else TEXT,
        4: TEXT_DIM,
    },
)

st.markdown(
    f"<div style='margin-top:14px;padding:10px 14px;background:{PRIMARY}15;border-radius:6px;"
    f"font-family:{MONO};font-size:11px;color:{PRIMARY};'>"
    f"◆ Active stack: Hybrid RRF + Cross-Encoder achieves <b>F1 0.89</b>, "
    f"a <b>+56% gain</b> over BM25-only baseline at a cost of 0.53s extra latency.</div>",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Per-phase latency detail
section_title("⏱️", "Latency Budget Analysis")

total_ms = sum(d["ms"] for d in LATENCY_BREAKDOWN)
cols = st.columns(len(LATENCY_BREAKDOWN), gap="small")
for col, d in zip(cols, LATENCY_BREAKDOWN):
    pct = d["ms"] / total_ms * 100
    node_color = d["color"]
    node_ms = d["ms"]
    node_phase = d["phase"]
    col.markdown(
        f"<div style='background:{SURFACE};border:1px solid {BORDER};border-radius:8px;"
        f"padding:12px 8px;text-align:center;border-top:3px solid {node_color};'>"
        f"<div style='font-family:{MONO};font-size:14px;font-weight:700;color:{node_color};'>{node_ms}ms</div>"
        f"<div style='color:{TEXT_MUTED};font-size:9px;font-family:{MONO};margin-top:3px;'>{node_phase}</div>"
        f"<div style='color:{TEXT_DIM};font-size:9px;margin-top:2px;'>{pct:.0f}%</div>"
        f"</div>",
        unsafe_allow_html=True,
    )