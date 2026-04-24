"""
FinSight-Alpha Dashboard — Architecture page
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from components.theme import inject_css, PRIMARY, ACCENT, GREEN, RED, PURPLE, TEAL, PINK
from components.theme import SURFACE, SURFACE2, SURFACE3, BORDER, TEXT, TEXT_MUTED, TEXT_DIM, MONO, DISPLAY
from components.ui_components import section_title, badge_html, mono_label, kv_row
from components.data import (
    PIPELINE_NODES, ROUTING_RULES, AGENT_STATE_FIELDS, OPTIMIZATION_COMPONENTS
)

st.set_page_config(page_title="Architecture · FinSight-Alpha", page_icon="🏗️", layout="wide")
inject_css()

section_title(
    "🏗️", "System Architecture",
    "LangGraph-powered Plan-Rewrite-Retrieve-Reason-Reflect loop "
    "with self-correcting hallucination detection and token-budget management.",
)

# Interactive pipeline diagram
st.markdown(
    f"<div style='background:{SURFACE};border:1px solid {BORDER};border-radius:10px;padding:22px;margin-bottom:20px;'>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<div style='font-family:{MONO};font-size:11px;color:{TEXT_MUTED};letter-spacing:0.08em;margin-bottom:16px;'>"
    f"EXECUTION PIPELINE — SELECT NODE TO INSPECT</div>",
    unsafe_allow_html=True,
)

# Node selector tabs
node_labels = [n["label"] for n in PIPELINE_NODES]
selected_label = st.radio(
    "Pipeline node",
    node_labels,
    horizontal=True,
    label_visibility="collapsed",
)
selected_node = next(n for n in PIPELINE_NODES if n["label"] == selected_label)

# Arrow-connected node strip
arrows_html = ""
for i, node in enumerate(PIPELINE_NODES):
    active = node["label"] == selected_label
    border_color = node["color"] if active else BORDER
    bg_color = f"{node['color']}22" if active else SURFACE2
    glow = f"box-shadow:0 0 12px {node['color']}55;" if active else ""
    node_color = node["color"]
    node_label = node["label"]
    arrows_html += (
        f"<div style='padding:10px 14px;background:{bg_color};border:2px solid {border_color};"
        f"border-radius:8px;min-width:90px;text-align:center;{glow};transition:all 0.2s;'>"
        f"<div style='font-family:{MONO};font-size:10px;font-weight:700;color:{node_color};letter-spacing:0.08em;'>"
        f"{node_label}</div></div>"
    )
    if i < len(PIPELINE_NODES) - 1:
        arrows_html += f"<div style='color:{TEXT_DIM};font-size:12px;padding:0 4px;align-self:center;'>▶</div>"

st.markdown(
    f"<div style='display:flex;align-items:center;flex-wrap:wrap;gap:0;overflow-x:auto;padding-bottom:4px;'>"
    f"{arrows_html}"
    f"<div style='margin-left:10px;padding:4px 10px;background:{ACCENT}22;border:1px solid {ACCENT}55;"
    f"border-radius:20px;font-family:{MONO};font-size:9px;color:{ACCENT};white-space:nowrap;'>↩ LOOP</div>"
    f"</div>",
    unsafe_allow_html=True,
)

# Detail panel for selected node
st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
c = selected_node["color"]
detail_items = [
    ("Prompt",  selected_node["prompt"]),
    ("Model",   selected_node["model"]),
    ("Input",   selected_node["input"]),
    ("Output",  selected_node["output"]),
    ("Note",    selected_node.get("note", "—")),
]
detail_html = "".join(
    f"<div style='flex:1;min-width:160px;padding:10px 12px;background:{SURFACE2};border-radius:6px;'>"
    f"<div style='font-family:{MONO};font-size:9px;color:{TEXT_DIM};text-transform:uppercase;margin-bottom:4px;'>{k}</div>"
    f"<div style='font-family:{MONO};font-size:11px;color:{TEXT};'>{v}</div>"
    f"</div>"
    for k, v in detail_items
)
st.markdown(
    f"<div style='padding:16px;background:{SURFACE3};border-radius:8px;border:1px solid {c}44;'>"
    f"<div style='font-family:{MONO};font-size:11px;color:{c};font-weight:700;margin-bottom:12px;'>"
    f"◆ {selected_node['label']} — {selected_node['desc']}</div>"
    f"<div style='display:flex;flex-wrap:wrap;gap:10px;'>{detail_html}</div>"
    f"</div>",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)  # close outer card

st.markdown("<br>", unsafe_allow_html=True)

# Two-column: routing + state schema
col_l, col_r = st.columns(2, gap="large")

with col_l:
    st.markdown(
        f"<div style='background:{SURFACE};border:1px solid {BORDER};border-radius:10px;padding:20px;'>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='font-family:{MONO};font-size:11px;color:{TEXT_MUTED};letter-spacing:0.08em;margin-bottom:16px;'>"
        f"CONDITIONAL ROUTING LOGIC</div>",
        unsafe_allow_html=True,
    )
    for rule in ROUTING_RULES:
        st.markdown(
            f"<div style='padding:10px 12px;background:{SURFACE2};border-radius:6px;"
            f"border-left:3px solid {rule['color']};margin-bottom:8px;'>"
            f"<div style='font-family:{MONO};font-size:10px;color:{TEXT_MUTED};'>{rule['from']}</div>"
            f"<div style='font-family:{MONO};font-size:10px;color:{rule['color']};margin:3px 0;'>if {rule['condition']}</div>"
            f"<div style='font-family:{MONO};font-size:10px;color:{TEXT};'>→ {rule['to']}</div>"
            f"</div>",
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
        f"AGENTSTATE SCHEMA (TypedDict)</div>",
        unsafe_allow_html=True,
    )
    rows_html = "".join(
        f"<div style='display:flex;justify-content:space-between;padding:4px 0;"
        f"border-bottom:1px solid {BORDER};font-family:{MONO};font-size:11px;'>"
        f"<span style='color:{color};'>{field}</span>"
        f"<span style='color:{TEXT_DIM};font-size:10px;'>{type_}</span>"
        f"</div>"
        for field, type_, color in AGENT_STATE_FIELDS
    )
    st.markdown(rows_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Optimisation components grid
section_title("⚙️", "Optimisation Layer Components")
cols = st.columns(3, gap="medium")
for i, comp in enumerate(OPTIMIZATION_COMPONENTS):
    with cols[i % 3]:
        st.markdown(
            f"<div style='background:{SURFACE};border:1px solid {comp['color']}30;"
            f"border-radius:10px;padding:16px;margin-bottom:12px;'>"
            f"<div style='font-family:{MONO};font-size:11px;color:{comp['color']};font-weight:700;margin-bottom:6px;'>"
            f"{comp['name']}</div>"
            f"<div style='font-size:12px;color:{TEXT};margin-bottom:6px;'>{comp['role']}</div>"
            f"<div style='font-family:{MONO};font-size:10px;color:{TEXT_DIM};line-height:1.5;'>{comp['detail']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )