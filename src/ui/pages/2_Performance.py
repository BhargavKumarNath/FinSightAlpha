"""
FinSight-Alpha Dashboard — Performance Metrics page
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import plotly.graph_objects as go
from components.theme import inject_css, PRIMARY, ACCENT, GREEN
from components.theme import SURFACE, BORDER, TEXT_MUTED, TEXT_DIM, MONO
from components.ui_components import section_title, stat_metric
from components.charts import latency_breakdown_chart
from components.data import EVAL_RESULTS, RAGAS_SUMMARY, LATENCY_BREAKDOWN

st.set_page_config(page_title="Performance · FinSight-Alpha", page_icon="📊", layout="wide")
inject_css()

section_title(
    "📊", "Performance Metrics",
    "RAGAS evaluation results computed against the live index, and pipeline latency profiling.",
)

# Top KPI row — only metrics with a real, stored artifact behind them.
if RAGAS_SUMMARY:
    kpi_cols = st.columns(3, gap="small")
    kpis = [
        (f"{RAGAS_SUMMARY['avg_faithfulness'] * 100:.0f}%",     "Avg Faithfulness",     PRIMARY),
        (f"{RAGAS_SUMMARY['avg_answer_relevancy'] * 100:.0f}%", "Avg Answer Relevancy", ACCENT),
        (str(RAGAS_SUMMARY["num_queries"]),                     "Queries Evaluated",    GREEN),
    ]
    for col, (val, label, color) in zip(kpi_cols, kpis):
        stat_metric(col, val, label, color)
    st.markdown(
        f"<div style='font-size:10px;color:{TEXT_DIM};margin-top:8px;'>"
        f"Computed from <code>data/reports/ragas_evaluation_report.csv</code> (RAGAS Faithfulness + Answer Relevancy, "
        f"LLM-as-judge, {RAGAS_SUMMARY['num_queries']} real queries). MRR, NDCG, round-trip time, and cache hit rate "
        f"are computed live per query and are not persisted historically — see the Live Console page for real "
        f"per-query values.</div>",
        unsafe_allow_html=True,
    )
else:
    st.warning(
        "No RAGAS evaluation report found at `data/reports/ragas_evaluation_report.csv`. "
        "Run the evaluator (`src/evaluation/ragas_evaluator.py`) to generate one before this page can show real scores."
    )

st.markdown("<br>", unsafe_allow_html=True)

# Real per-query RAGAS scores (replaces a previously fabricated 5-run "trend"
# chart — there is no historical multi-run data to plot, only these
# real single-run results).
if EVAL_RESULTS:
    section_title(
        "📈", "Per-Query RAGAS Scores",
        f"{len(EVAL_RESULTS)} real evaluation queries scored against the live index.",
    )
    questions = [r["q"][:44] + ("…" if len(r["q"]) > 44 else "") for r in EVAL_RESULTS]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Faithfulness", x=questions, y=[r["faith"] for r in EVAL_RESULTS], marker=dict(color=PRIMARY)))
    fig.add_trace(go.Bar(name="Answer Relevancy", x=questions, y=[r["relev"] for r in EVAL_RESULTS], marker=dict(color=ACCENT)))
    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_MUTED, family=MONO, size=10),
        margin=dict(l=10, r=10, t=10, b=10), height=260,
        yaxis=dict(range=[0, 1], showgrid=True, gridcolor=BORDER, zeroline=False),
        xaxis=dict(showgrid=False),
        legend=dict(orientation="h", y=1.18),
    )
    st.markdown(
        f"<div style='background:{SURFACE};border:1px solid {BORDER};border-radius:10px;padding:18px;'>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# Per-phase latency — illustrative only (no historical run is stored;
# latencies are only ever computed live per query, see AgentState['latencies']).
section_title("⏱️", "Latency Budget Analysis")
st.markdown(
    f"<div style='font-size:10px;color:{TEXT_DIM};margin-bottom:10px;'>"
    f"Illustrative example shape, not a measured run — per-node latency is only ever computed live per query "
    f"and is not persisted to disk. Real per-query latencies are shown in the Live Console.</div>",
    unsafe_allow_html=True,
)

st.markdown(
    f"<div style='background:{SURFACE};border:1px solid {BORDER};border-radius:10px;padding:18px;'>",
    unsafe_allow_html=True,
)
st.plotly_chart(latency_breakdown_chart(LATENCY_BREAKDOWN), width='stretch', config={"displayModeBar": False})
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

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
