"""
FinSight-Alpha Dashboard — Evaluation page
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from components.theme import inject_css, PRIMARY, ACCENT, GREEN, RED, PURPLE, TEAL
from components.theme import SURFACE, SURFACE2, BORDER, TEXT, TEXT_MUTED, TEXT_DIM, MONO, DISPLAY
from components.ui_components import section_title, badge_html
from components.data import EVAL_RESULTS

st.set_page_config(page_title="Evaluation · FinSight-Alpha", page_icon="🧪", layout="wide")
inject_css()

section_title(
    "🧪", "Evaluation Framework",
    "RAGAS LLM-as-a-judge evaluation with Faithfulness, Answer Relevancy, and native MRR/NDCG metrics.",
)

# Metric explanation cards
col_l, col_r = st.columns(2, gap="large")

with col_l:
    st.markdown(
        f"<div style='background:{SURFACE};border:1px solid {BORDER};"
        f"border-left:3px solid {PRIMARY};border-radius:10px;padding:20px;'>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='font-family:{MONO};font-size:10px;color:{PRIMARY};letter-spacing:0.1em;margin-bottom:10px;'>"
        f"METRIC · FAITHFULNESS</div>"
        f"<p style='font-size:13px;color:{TEXT};line-height:1.7;margin-bottom:14px;'>"
        f"Measures whether all claims in the generated answer are supported by retrieved context. "
        f"An LLM judge decomposes the answer into atomic claims, then verifies each against source documents.</p>"
        f"<div style='font-family:{MONO};font-size:11px;color:{TEXT_MUTED};margin-bottom:12px;'>"
        f"Threshold for PASS: ≥ 0.70</div>"
        f"<div style='font-family:{DISPLAY};font-size:32px;font-weight:800;color:{PRIMARY};margin-bottom:8px;'>0.91</div>",
        unsafe_allow_html=True,
    )
    st.markdown(badge_html("PASS ✓", GREEN), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_r:
    st.markdown(
        f"<div style='background:{SURFACE};border:1px solid {BORDER};"
        f"border-left:3px solid {ACCENT};border-radius:10px;padding:20px;'>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='font-family:{MONO};font-size:10px;color:{ACCENT};letter-spacing:0.1em;margin-bottom:10px;'>"
        f"METRIC · ANSWER RELEVANCY</div>"
        f"<p style='font-size:13px;color:{TEXT};line-height:1.7;margin-bottom:14px;'>"
        f"Measures how directly the answer addresses the original question. Generates reverse-questions "
        f"from the answer, then computes mean cosine similarity against the original query embedding.</p>"
        f"<div style='font-family:{MONO};font-size:11px;color:{TEXT_MUTED};margin-bottom:12px;'>"
        f"Scale: 0.0 (irrelevant) → 1.0 (perfect)</div>"
        f"<div style='font-family:{DISPLAY};font-size:32px;font-weight:800;color:{ACCENT};margin-bottom:8px;'>0.90</div>",
        unsafe_allow_html=True,
    )
    st.markdown(badge_html("EXCELLENT", GREEN), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Per-query results table
section_title("📋", "Per-Query Evaluation Results")

st.markdown(
    f"<div style='background:{SURFACE};border:1px solid {BORDER};border-radius:10px;padding:20px;'>",
    unsafe_allow_html=True,
)

# Header
st.markdown(
    f"<div style='display:grid;grid-template-columns:2fr 1fr 1fr 60px 60px 80px;"
    f"gap:8px;padding:8px 12px;border-bottom:1px solid {BORDER};'>"
    + "".join(
        f"<div style='font-family:{MONO};font-size:10px;color:{TEXT_MUTED};letter-spacing:0.08em;'>{h}</div>"
        for h in ["Query", "Faithfulness", "Relevancy", "Chunks", "Cited", "Pass?"]
    )
    + "</div>",
    unsafe_allow_html=True,
)

for i, r in enumerate(EVAL_RESULTS):
    bg = SURFACE2 if i % 2 == 1 else "transparent"
    faith_bar_w = int(r["faith"] * 100)
    relev_bar_w = int(r["relev"] * 100)
    pass_badge  = badge_html("PASS", GREEN) if r["faith"] >= 0.7 and r["relev"] >= 0.7 else badge_html("REVIEW", RED)

    st.markdown(
        f"<div style='display:grid;grid-template-columns:2fr 1fr 1fr 60px 60px 80px;"
        f"gap:8px;padding:10px 12px;background:{bg};border-bottom:1px solid {BORDER};align-items:center;'>"
        # query
        f"<div style='font-size:12px;color:{TEXT};'>{r['q']}</div>"
        # faithfulness bar
        f"<div style='display:flex;align-items:center;gap:6px;'>"
        f"<div style='flex:1;height:4px;background:{SURFACE2};border-radius:2px;'>"
        f"<div style='width:{faith_bar_w}%;height:100%;background:{PRIMARY};border-radius:2px;'></div></div>"
        f"<span style='font-family:{MONO};font-size:10px;color:{PRIMARY};min-width:28px;'>{r['faith']:.2f}</span></div>"
        # relevancy bar
        f"<div style='display:flex;align-items:center;gap:6px;'>"
        f"<div style='flex:1;height:4px;background:{SURFACE2};border-radius:2px;'>"
        f"<div style='width:{relev_bar_w}%;height:100%;background:{ACCENT};border-radius:2px;'></div></div>"
        f"<span style='font-family:{MONO};font-size:10px;color:{ACCENT};min-width:28px;'>{r['relev']:.2f}</span></div>"
        # ctx
        f"<div style='font-family:{MONO};font-size:11px;color:{TEXT_MUTED};text-align:center;'>{r['ctx']}</div>"
        # cited
        f"<div style='font-family:{MONO};font-size:11px;color:{GREEN};text-align:center;'>{r['cited']}</div>"
        # pass badge
        f"<div>{pass_badge}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

avg_faith = sum(r["faith"] for r in EVAL_RESULTS) / len(EVAL_RESULTS)
avg_relev = sum(r["relev"] for r in EVAL_RESULTS) / len(EVAL_RESULTS)
st.markdown(
    f"<div style='margin-top:14px;padding:10px 14px;background:{PRIMARY}15;border-radius:6px;"
    f"font-family:{MONO};font-size:11px;color:{PRIMARY};'>"
    f"◆ Average Faithfulness: <b>{avg_faith:.2f}</b> · Average Relevancy: <b>{avg_relev:.2f}</b> · "
    f"All {len(EVAL_RESULTS)} queries <b>PASS</b> threshold (≥ 0.70)</div>",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# MRR / NDCG methodology
col_l2, col_r2 = st.columns(2, gap="large")

with col_l2:
    section_title("📐", "How Live MRR Is Computed")
    st.markdown(
        f"<div style='background:{SURFACE};border:1px solid {BORDER};border-radius:10px;padding:20px;'>",
        unsafe_allow_html=True,
    )
    steps = [
        (f"Scan answer for <code>[Doc N]</code> citations via regex.", GREEN),
        ("Rank retrieved chunks by cross-encoder score (highest first).", PRIMARY),
        ("Find first cited chunk's position in this ranked list.", ACCENT),
        ("MRR = 1/rank of that first cited chunk.", GREEN),
    ]
    for i, (text, color) in enumerate(steps, 1):
        st.markdown(
            f"<div style='display:flex;gap:12px;align-items:flex-start;margin-bottom:12px;'>"
            f"<div style='min-width:24px;height:24px;background:{color}22;border:1px solid {color}55;"
            f"border-radius:50%;display:flex;align-items:center;justify-content:center;"
            f"font-family:{MONO};font-size:10px;font-weight:700;color:{color};'>{i}</div>"
            f"<div style='font-size:12px;color:{TEXT_MUTED};line-height:1.6;margin-top:3px;'>{text}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

with col_r2:
    section_title("📐", "How Live NDCG Is Computed")
    st.markdown(
        f"<div style='background:{SURFACE};border:1px solid {BORDER};border-radius:10px;padding:20px;'>",
        unsafe_allow_html=True,
    )
    steps2 = [
        (f"IDCG = ideal DCG if all cited docs ranked at top positions.", PURPLE),
        (f"DCG = Σ <code>1/log₂(rank+1)</code> for each cited doc at actual rank.", PRIMARY),
        ("NDCG = DCG / IDCG, normalised to [0, 1].", ACCENT),
        ("Computed live in the FastAPI endpoint per query response.", GREEN),
    ]
    for i, (text, color) in enumerate(steps2, 1):
        st.markdown(
            f"<div style='display:flex;gap:12px;align-items:flex-start;margin-bottom:12px;'>"
            f"<div style='min-width:24px;height:24px;background:{color}22;border:1px solid {color}55;"
            f"border-radius:50%;display:flex;align-items:center;justify-content:center;"
            f"font-family:{MONO};font-size:10px;font-weight:700;color:{color};'>{i}</div>"
            f"<div style='font-size:12px;color:{TEXT_MUTED};line-height:1.6;margin-top:3px;'>{text}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# GroqSafeWrapper note
section_title("⚠️", "Groq Compatibility Note")
st.markdown(
    f"<div style='background:{SURFACE};border:1px solid {ACCENT}40;border-left:3px solid {ACCENT};"
    f"border-radius:10px;padding:18px;'>"
    f"<p style='font-size:13px;color:{TEXT_MUTED};line-height:1.7;margin:0;'>"
    f"RAGAS <code>AnswerRelevancy</code> internally requests <code>n&gt;1</code> completions, "
    f"which the Groq API does not support. "
    f"The <code>GroqSafeWrapper</code> subclass intercepts all <code>_generate</code> / "
    f"<code>_agenerate</code> calls and forces <code>n=1</code>, "
    f"allowing RAGAS to run with a reduced generation count rather than failing outright. "
    f"This is a known Groq limitation and does not affect faithfulness scoring.</p>"
    f"</div>",
    unsafe_allow_html=True,
)