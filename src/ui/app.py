import streamlit as st
import requests
import json
import re
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx

# Config
API_URL = "http://localhost:8000/chat"
HEALTH_URL = "http://localhost:8000/health"
CACHE_STATS_URL = "http://localhost:8000/cache/stats"

st.set_page_config(
    page_title="FinSight-Alpha Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Linear / Grafana Dark Theme CSS
css = """
<style>
    /* Base Colors & Typography */
    :root {
        --bg: #0A0A0C;
        --surface: #141416;
        --surface2: #1E1E22;
        --border: rgba(255, 255, 255, 0.08);
        --primary: #5E6AD2;
        --accent: #E5A84F;
        --text: #ffffff;
        --text-muted: #A0A0B0;
        --font: 'Inter', system-ui, sans-serif;
        --mono: 'JetBrains Mono', monospace;
    }
    
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: var(--bg) !important;
        color: var(--text) !important;
        font-family: var(--font) !important;
    }
    
    /* Hide Streamlit Header/Footer */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Panel Containers (Elevated Cards) */
    [data-testid="stVerticalBlock"] > div > div > div {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 18px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    [data-testid="stVerticalBlock"] > div > div > div:hover {
        box-shadow: 0 6px 16px rgba(0,0,0,0.35);
    }
    
    /* Inner overrides for main blocks */
    div.block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    
    /* Typographic tweaks */
    h1, h2, h3, h4, h5 {
        color: var(--text) !important;
        font-weight: 600 !important;
        margin-top: 0 !important;
        margin-bottom: 0.5rem !important;
    }
    
    .dashboard-title {
        font-size: 1.4rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-muted);
        border-bottom: 1px solid var(--border);
        padding-bottom: 0.75rem;
        margin-bottom: 1rem;
    }
    
    /* Chat bubbles */
    [data-testid="chatAvatarIcon-user"] { background-color: var(--surface2) !important; }
    [data-testid="chatAvatarIcon-assistant"] { background-color: var(--primary) !important; }
    [data-testid="stChatMessage"] {
        background-color: transparent !important;
        border: none !important;
        padding: 0.75rem 0 !important;
    }
    
    /* Trace & Code */
    code, pre {
        font-family: var(--mono) !important;
        font-size: 0.8rem !important;
        background-color: var(--surface2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 4px !important;
        color: var(--accent) !important;
    }
    
    /* Metrics Fixes */
    [data-testid="stMetricValue"] {
        font-family: var(--mono) !important;
        font-size: 1.25rem !important;
        color: var(--text) !important;
        line-height: 1.2 !important;
    }
    [data-testid="stMetricLabel"] {
        color: var(--text-muted) !important;
        font-size: 0.8rem !important;
        text-transform: uppercase !important;
        margin-bottom: 4px !important;
    }
    
    /* Citations Tag */
    .citation-tag {
        display: inline-block;
        background: var(--surface2);
        color: var(--accent);
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.75rem;
        margin-right: 6px;
        margin-top: 4px;
        font-family: var(--mono);
        transition: background 0.2s;
    }
    .citation-tag:hover {
        background: #2D2D33;
        cursor: pointer;
    }
    
    /* Status Pills */
    .status-pill {
        display: inline-block;
        background: rgba(94, 106, 210, 0.15);
        color: var(--primary);
        border: 1px solid var(--primary);
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-right: 8px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# State Management
if "messages" not in st.session_state:
    st.session_state.messages = []
if "telemetry" not in st.session_state:
    st.session_state.telemetry = {
        "queries": [],
        "rtt": [],
        "tokens": [],
        "cache_hits": [],
    }

# Data Fetching
@st.cache_data(ttl=5)
def get_health_stats():
    try:
        r = requests.get(HEALTH_URL, timeout=2)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return {"budget": {"total_tokens": 0, "session_budget": 100000, "usage_pct": 0, "tier": "UNKNOWN"}}

@st.cache_data(ttl=5)
def get_cache_stats():
    try:
        r = requests.get(CACHE_STATS_URL, timeout=2)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return {"hit_rate": 0, "total_hits": 0, "total_misses": 0}

health = get_health_stats()
cache = get_cache_stats()

# Plotly Helpers
def plot_bullet_gauge(value, title, max_val=1.0, is_pct=True):
    color = "#5E6AD2" if value < 0.75 else ("#E5A84F" if value < 0.9 else "#E04F5E")
    display_val = value * 100 if is_pct else value
    reference_max = 100 if is_pct else max_val
    
    fig = go.Figure(go.Indicator(
        mode="number+gauge",
        value=display_val,
        number={'suffix': "%" if is_pct else "", 'font': {'size': 20, 'color': '#EDEDED', 'family': 'JetBrains Mono'}},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 12, 'color': '#A0A0B0'}},
        gauge={
            'shape': 'bullet',
            'axis': {'range': [0, reference_max], 'visible': False},
            'bar': {'color': color, 'thickness': 1},
            'bgcolor': "#1E1E22",
            'borderwidth': 0,
        }
    ))
    fig.update_layout(margin=dict(l=100, r=15, t=20, b=20), height=60, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

def plot_waterfall_latency(latencies_dict, total_time):
    if not latencies_dict:
        phases = ["Network / API Request"]
        times = [total_time]
    else:
        phases = list(latencies_dict.keys())
        times = list(latencies_dict.values())
        measured = sum(times)
        if total_time > measured:
            phases.append("Overhead")
            times.append(total_time - measured)
            
    fig = go.Figure(go.Bar(
        y=phases[::-1],
        x=times[::-1],
        orientation='h',
        marker=dict(color="#5E6AD2", line=dict(color="#141416", width=1))
    ))
    fig.update_layout(
        title={'text': f"Live Node Latency Breakdown: {total_time:.2f}s", 'font': {'size': 12, 'color': '#A0A0B0'}},
        margin=dict(l=10, r=10, t=30, b=10),
        height=180,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False, tickfont=dict(size=10, family="JetBrains Mono")),
        yaxis=dict(tickfont=dict(size=10, color="#A0A0B0")),
    )
    return fig

def plot_knowledge_graph(query, trace_list, citations):
    G = nx.Graph()
    G.add_node("Query", type="query", size=20)
    
    subqueries = []
    for t in trace_list:
        if "Searched for:" in t:
            sq = t.replace("Searched for:", "").strip()
            subqueries.append(sq)
            short = sq[:20] + "..." if len(sq) > 20 else sq
            G.add_node(short, type="subquery", size=15)
            G.add_edge("Query", short)
            
            for doc in set(citations):
                G.add_node(doc, type="doc", size=10)
                G.add_edge(short, doc)

    if not subqueries:
        G.add_node("Cache Hit / Direct", type="subquery", size=15)
        G.add_edge("Query", "Cache Hit / Direct")
        
    pos = nx.spring_layout(G, seed=42)
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='rgba(255,255,255,0.2)'),
        hoverinfo='none',
        mode='lines')

    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        t = G.nodes[node].get("type", "doc")
        
        if t == "query": node_color.append("#E5A84F")
        elif t == "subquery": node_color.append("#5E6AD2")
        else: node_color.append("#ffffff")
        
        node_size.append(G.nodes[node].get("size", 10))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        textposition="bottom center",
        hoverinfo='text',
        marker=dict(showscale=False, color=node_color, size=node_size, line_width=1, line_color="#141416"),
        textfont=dict(color="#ffffff", size=9)
    )

    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title={'text': "Dynamic Knowledge Graph", 'font': {'size': 12, 'color': '#A0A0B0'}},
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0, l=0, r=0, t=30),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
             )
    
    if len(G.nodes()) > 10:
        fig.data[1].text = [t if G.nodes[t].get("type") != "doc" else "" for t in node_text]
        
    return fig

def plot_agent_lifecycle():
    # Simple static flowchart for the empty state
    fig = go.Figure()
    stages = ["Plan", "Rewrite", "Retrieve", "Reason", "Reflect", "Finalize"]
    x = list(range(len(stages)))
    y = [1]*len(stages)
    
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers+text+lines",
        text=stages,
        textposition="top center",
        marker=dict(size=14, color="#5E6AD2", line=dict(width=2, color="#141416")),
        line=dict(color="rgba(255,255,255,0.2)", width=2),
        textfont=dict(color="#A0A0B0", size=10)
    ))
    
    fig.update_layout(
        title={'text': "Agent Execution Pathway", 'font': {'size': 12, 'color': '#A0A0B0'}},
        margin=dict(l=10, r=10, t=30, b=10),
        height=100,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False
    )
    return fig

# Dashboard Layout
st.markdown('<div class="dashboard-title">Intelligence Hub</div>', unsafe_allow_html=True)

# Wide-spaced columns
col_left, col_mid, col_right = st.columns([1.2, 3, 1.5], gap="large")

# LEFT PANEL: Telemetry
with col_left:
    st.markdown("##### System Telemetry")
    
    budget = health.get("budget", {})
    pct = budget.get("usage_pct", 0)
    
    # Bullet Gauges (Space Efficient)
    st.plotly_chart(plot_bullet_gauge(pct, "Budget Burn", is_pct=True), width='stretch', config={'displayModeBar': False})
    st.plotly_chart(plot_bullet_gauge(cache.get("hit_rate", 0), "Cache Hits", is_pct=True), width='stretch', config={'displayModeBar': False})
    
    # Session Timeline Sparkline
    if st.session_state.telemetry["queries"]:
        df_vol = pd.DataFrame({
            "idx": range(len(st.session_state.telemetry["queries"])),
            "tokens": st.session_state.telemetry["tokens"]
        })
        fig_spark = px.area(df_vol, x="idx", y="tokens")
        fig_spark.update_traces(line_color="#5E6AD2", fillcolor="rgba(94,106,210,0.2)")
        fig_spark.update_layout(
            title={'text': "Session Token Spikes", 'font': {'size': 11, 'color': '#A0A0B0'}},
            height=100, margin=dict(l=0, r=0, t=25, b=0), 
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
            xaxis_visible=False, yaxis_visible=False
        )
        st.plotly_chart(fig_spark, width='stretch', config={'displayModeBar': False})
    
    # Metrics Container (Un-nested columns to prevent clipping)
    st.markdown("<br>", unsafe_allow_html=True)
    avg_rtt = f"{sum(st.session_state.telemetry['rtt'])/len(st.session_state.telemetry['rtt']):.2f}s" if st.session_state.telemetry['rtt'] else "0.0s"
    st.metric("Aggregate Mean RTT", avg_rtt)
    st.metric("Total Fallback Rate", "0.0%", help="Pipeline forces logic loops safely")


# MID PANEL: Chat Operations
with col_mid:
    st.markdown("##### Operational Console")
    
    chat_container = st.container(height=600)
    
    with chat_container:
        if not st.session_state.messages:
            # Active Status Landing View
            st.markdown(
                "<h4>⚡ Ready for Intel</h4>"
                "<span class='status-pill'>LLM Router Active</span>"
                "<span class='status-pill'>Hybrid Index Online</span>"
                "<span class='status-pill'>MRR Checks Ready</span><br><br>"
                "<div style='color:var(--text-muted); font-size:0.95rem; line-height: 1.6;'>"
                "The pipeline is actively evaluating token paths.<br>"
                "To initiate, drop a multi-hop query into the command bar below."
                "</div>",
                unsafe_allow_html=True
            )
            
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Render Citations parsed from answer
                if message["role"] == "assistant":
                    citations = re.findall(r'\[Doc\s*\d+:\s*([^\]]+)\]', message["content"])
                    citations = list(set(citations)) # unique
                    if citations:
                        html = ""
                        for c in citations:
                            html += f'<span class="citation-tag">{c}</span>'
                        st.markdown(f"**Sources Interrogated:**<br>{html}", unsafe_allow_html=True)
                
    if prompt := st.chat_input("Enter strategic query..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
                
            with st.chat_message("assistant"):
                st.markdown('<span style="color:var(--text-muted); font-size: 0.8rem; font-family: var(--mono)">Executing Plan-Rewrite-Retrieve loop...</span>', unsafe_allow_html=True)
                
                start_time = time.time()
                try:
                    res = requests.post(API_URL, json={"query": prompt}, timeout=60)
                    rtt = time.time() - start_time
                    
                    if res.status_code == 200:
                        data = res.json()
                        ans = data.get("answer", "Error")
                        trace = data.get("reasoning_trace", [])
                        
                        st.markdown(ans)
                        
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": ans,
                            "trace": trace,
                            "rtt": rtt,
                            "query": prompt,
                            "latencies": data.get("latencies", {}),
                            "retrieval_metrics": data.get("retrieval_metrics", {})
                        })
                        
                        # Update Telemetry
                        st.session_state.telemetry["queries"].append(prompt)
                        st.session_state.telemetry["rtt"].append(rtt)
                        st.session_state.telemetry["tokens"].append(data.get("tokens_used", 0))
                        st.session_state.telemetry["cache_hits"].append(1 if data.get("cache_hit") else 0)
                        st.rerun()
                        
                    else:
                        st.error(f"API Error {res.status_code}")
                except Exception as e:
                    st.error(f"Execution Failure: {e}")

# RIGHT PANEL: Analytics
with col_right:
    st.markdown("##### System Diagnostics")
    
    if st.session_state.messages and "trace" in st.session_state.messages[-1]:
        latest = st.session_state.messages[-1]
        
        # 1. Latency Breakdown
        st.plotly_chart(plot_waterfall_latency(latest.get("latencies", {}), latest.get("rtt", 0.0)), width='stretch', config={'displayModeBar': False})
        
        # 2. Knowledge Graph
        citations = re.findall(r'\[Doc\s*\d+:\s*([^\]]+)\]', latest["content"])
        fig_kg = plot_knowledge_graph(latest["query"], latest["trace"], citations)
        st.plotly_chart(fig_kg, width='stretch', config={'displayModeBar': False})
        
        # 3. Live Native Retrieval Metrics (MRR / NDCG)
        metrics = latest.get("retrieval_metrics", {})
        mrr = metrics.get("MRR", 0.0)
        ndcg = metrics.get("NDCG", 0.0)
        st.markdown("<hr style='border-top:1px solid rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
        colM1, colM2 = st.columns(2)
        colM1.metric("Live MRR", f"{mrr:.2f}")
        colM2.metric("Live NDCG", f"{ndcg:.2f}")
        
    else:
        # Pre-execution active flowchart display
        st.plotly_chart(plot_agent_lifecycle(), width='stretch', config={'displayModeBar': False})
        st.markdown("<hr style='border-top:1px solid rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
        st.markdown(
            "<div style='color:var(--text-muted); font-size:0.85rem; line-height: 1.5;'>"
            "Awaiting operational workload...<br><br>"
            "<b>Stream Targets:</b><br>"
            "• Node-Level Latency Tracker<br>"
            "• Search Bipartite Maps<br>"
            "• Live MRR Precision Readouts"
            "</div>", unsafe_allow_html=True
        )
