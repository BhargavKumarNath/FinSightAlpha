"""
FinSight-Alpha Dashboard — Optimisation page
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from components.theme import inject_css, PRIMARY, ACCENT, GREEN, RED, PINK
from components.theme import SURFACE, SURFACE2, SURFACE3, BORDER, TEXT, TEXT_MUTED, TEXT_DIM, MONO, DISPLAY
from components.ui_components import section_title, badge_html, info_table
from components.charts import token_savings_chart
from components.data import (
    BUDGET_TIERS, MODEL_ROUTER_TABLE, CACHE_TOKEN_SAVINGS,
    CACHE_CONFIG_ROWS, CONTEXT_WINDOW_DETAILS,
)

st.set_page_config(page_title="Optimisation · FinSight-Alpha", page_icon="⚡", layout="wide")
inject_css()

section_title(
    "⚡", "Optimisation Layer",
    "Token budget management, semantic caching, tiered model routing, and batched embedding strategies.",
)

# Token budget tiers
st.markdown(
    f"<div style='background:{SURFACE};border:1px solid {BORDER};border-radius:10px;padding:20px;margin-bottom:20px;'>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<div style='font-family:{MONO};font-size:11px;color:{TEXT_MUTED};letter-spacing:0.08em;margin-bottom:18px;'>"
    f"TOKEN BUDGET TIER SYSTEM</div>",
    unsafe_allow_html=True,
)
for tier in BUDGET_TIERS:
    c = tier["color"]
    skips_html = "".join(badge_html(s, RED) for s in tier["skips"]) if tier["skips"] else badge_html("None", TEXT_DIM)
    st.markdown(
        f"<div style='padding:16px;background:{c}10;border:1px solid {c}35;border-radius:8px;margin-bottom:10px;'>"
        f"<div style='display:flex;flex-wrap:wrap;gap:16px;align-items:center;'>"
        # tier name + range
        f"<div style='min-width:130px;'>"
        f"<div style='font-family:{DISPLAY};font-size:18px;font-weight:800;color:{c};'>{tier['tier']}</div>"
        f"<div style='font-family:{MONO};font-size:10px;color:{TEXT_MUTED};margin-top:2px;'>{tier['range']} tokens</div>"
        f"</div>"
        # description
        f"<div style='flex:1;font-size:12px;color:{TEXT_MUTED};min-width:180px;'>{tier['desc']}</div>"
        # stats
        f"<div style='display:flex;gap:20px;'>"
        f"<div style='text-align:center;'>"
        f"<div style='font-family:{MONO};font-size:18px;font-weight:700;color:{c};'>{tier['loops']}</div>"
        f"<div style='font-family:{MONO};font-size:9px;color:{TEXT_DIM};'>MAX LOOPS</div>"
        f"</div>"
        f"<div style='text-align:center;'>"
        f"<div style='font-family:{MONO};font-size:18px;font-weight:700;color:{c};'>{tier['top_n']}</div>"
        f"<div style='font-family:{MONO};font-size:9px;color:{TEXT_DIM};'>TOP-N</div>"
        f"</div>"
        f"<div style='text-align:center;'>"
        f"<div style='font-family:{MONO};font-size:12px;font-weight:700;color:{c};'>{tier['model']}</div>"
        f"<div style='font-family:{MONO};font-size:9px;color:{TEXT_DIM};'>MODEL</div>"
        f"</div>"
        f"</div>"
        # skips
        f"<div style='display:flex;align-items:center;gap:6px;flex-wrap:wrap;'>"
        f"<span style='font-family:{MONO};font-size:9px;color:{TEXT_DIM};'>SKIPS:</span>{skips_html}"
        f"</div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )
st.markdown("</div>", unsafe_allow_html=True)

# Cache row
col_l, col_r = st.columns(2, gap="large")

with col_l:
    st.markdown(
        f"<div style='background:{SURFACE};border:1px solid {BORDER};border-radius:10px;padding:20px;'>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(token_savings_chart(CACHE_TOKEN_SAVINGS), width='stretch', config={"displayModeBar": False})
    st.markdown(
        f"<div style='font-size:10px;color:{TEXT_DIM};margin-top:-8px;'>"
        f"Illustrative example shape — no historical session log is persisted to disk. "
        f"Real cache hits/misses for a live query are printed by <code>SemanticResponseCache.get()</code> "
        f"and shown in the Live Console.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col_r:
    st.markdown(
        f"<div style='background:{SURFACE};border:1px solid {BORDER};border-radius:10px;padding:20px;'>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='font-family:{MONO};font-size:11px;color:{TEXT_MUTED};letter-spacing:0.08em;margin-bottom:14px;'>"
        f"SEMANTIC CACHE CONFIGURATION</div>",
        unsafe_allow_html=True,
    )
    for k, v, c in CACHE_CONFIG_ROWS:
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;padding:6px 0;"
            f"border-bottom:1px solid {BORDER};'>"
            f"<span style='font-size:12px;color:{TEXT_MUTED};'>{k}</span>"
            f"<span style='font-family:{MONO};font-size:11px;color:{c};'>{v}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Model router table
section_title("🤖", "Model Router — Task Dispatch Table")

st.markdown(
    f"<div style='background:{SURFACE};border:1px solid {BORDER};border-radius:10px;padding:20px;'>",
    unsafe_allow_html=True,
)

tier_badge_color = {
    "ALL": GREEN, "GREEN/YELLOW": PRIMARY, "RED": RED, "ERROR": ACCENT,
}

rows = []
for r in MODEL_ROUTER_TABLE:
    node, tier, model, reason, cost, color = r
    rows.append([
        f"<span style='font-family:{MONO};font-size:11px;color:{TEXT};'>{node}</span>",
        badge_html(tier, tier_badge_color.get(tier, TEXT_MUTED)),
        f"<span style='font-family:{MONO};font-size:11px;color:{color};'>{model}</span>",
        f"<span style='font-size:12px;color:{TEXT_MUTED};'>{reason}</span>",
        f"<span style='font-family:{MONO};font-size:11px;color:{TEXT_DIM};'>{cost}</span>",
    ])

info_table(
    headers=["Node", "Tier", "Model", "Reasoning", "Est. Cost"],
    rows=rows,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Context window details
section_title("🪟", "Dynamic Context Window", "How retrieved chunks are filtered and truncated before injection into the LLM.")

cols = st.columns(3, gap="medium")
for i, (k, v, desc, color) in enumerate(CONTEXT_WINDOW_DETAILS):
    with cols[i % 3]:
        st.markdown(
            f"<div style='background:{SURFACE};border:1px solid {BORDER};"
            f"border-top:3px solid {color};border-radius:8px;padding:14px;margin-bottom:12px;'>"
            f"<div style='font-family:{MONO};font-size:18px;font-weight:700;color:{color};'>{v}</div>"
            f"<div style='font-family:{MONO};font-size:10px;color:{TEXT};margin-top:4px;font-weight:700;'>{k}</div>"
            f"<div style='color:{TEXT_DIM};font-size:11px;margin-top:4px;'>{desc}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )