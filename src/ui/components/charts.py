"""
FinSight-Alpha — Plotly chart factory.

All charts share the same dark theme.  Each function returns a
go.Figure ready to pass to st.plotly_chart().
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import networkx as nx

from components.theme import (
    BG, SURFACE, SURFACE2, BORDER,
    PRIMARY, ACCENT, GREEN, RED, PURPLE, PINK, TEAL,
    TEXT, TEXT_MUTED, TEXT_DIM, MONO,
)

_TRANSPARENT = "rgba(0,0,0,0)"

LAYOUT_BASE = dict(
    paper_bgcolor=_TRANSPARENT,
    plot_bgcolor=_TRANSPARENT,
    font=dict(color=TEXT_MUTED, family=MONO, size=10),
    margin=dict(l=10, r=10, t=36, b=10),
    title_font=dict(size=12, color=TEXT_MUTED, family=MONO),
)

GRID_STYLE = dict(showgrid=True, gridcolor=BORDER, zeroline=False)
NO_GRID    = dict(showgrid=False, zeroline=False, showticklabels=False)

TOOLTIP_STYLE = dict(
    bgcolor=SURFACE2,
    bordercolor=BORDER,
    font=dict(color=TEXT, family=MONO, size=11),
)

# Colour helpers

def _rgba(hex_color: str, alpha: float) -> str:
    """Convert a 6-digit hex colour + alpha float to an rgba() string.
    Plotly does NOT accept 8-digit hex colours like '#RRGGBBAA'.
    """
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _apply_base(fig: go.Figure, title: str = "", height: int = 220) -> go.Figure:
    fig.update_layout(**LAYOUT_BASE, title_text=title, height=height)
    fig.update_traces(hoverlabel=TOOLTIP_STYLE)
    return fig


# Individual chart builders

def ragas_trend_chart(data: list[dict]) -> go.Figure:
    """Line chart: RAGAS metrics across pipeline iterations."""
    df = pd.DataFrame(data)
    fig = go.Figure()
    series = [
        ("faithfulness", PRIMARY, "Faithfulness"),
        ("relevancy",    ACCENT,  "Relevancy"),
        ("mrr",          GREEN,   "MRR"),
        ("ndcg",         PURPLE,  "NDCG"),
    ]
    for key, color, name in series:
        fig.add_trace(go.Scatter(
            x=df["run"], y=df[key], name=name,
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(color=color, size=5),
        ))
    fig.update_xaxes(**GRID_STYLE, tickfont=dict(size=9))
    fig.update_yaxes(**GRID_STYLE, range=[0.5, 1.0], tickfont=dict(size=9))
    return _apply_base(fig, "RAGAS Scores Across Iterations", height=220)


def retrieval_comparison_chart(data: list[dict]) -> go.Figure:
    """Horizontal grouped bar: retrieval method comparison."""
    df = pd.DataFrame(data)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["method"], x=df["f1"],
        name="F1 Score", orientation="h",
        marker=dict(color=PRIMARY),
    ))
    fig.add_trace(go.Bar(
        y=df["method"], x=df["precision"],
        name="Precision", orientation="h",
        marker=dict(color=ACCENT),
    ))
    fig.update_xaxes(range=[0, 1], **GRID_STYLE)
    fig.update_yaxes(tickfont=dict(size=9))
    fig.update_layout(barmode="group")
    return _apply_base(fig, "Retrieval Method Comparison", height=220)


def latency_breakdown_chart(data: list[dict]) -> go.Figure:
    """Horizontal bar: per-node latency in ms."""
    df = pd.DataFrame(data)
    colors = [d["color"] for d in data]
    fig = go.Figure(go.Bar(
        y=df["phase"][::-1],
        x=df["ms"][::-1],
        orientation="h",
        marker=dict(color=colors[::-1]),
        text=[f"{v}ms" for v in df["ms"][::-1]],
        textposition="outside",
        textfont=dict(size=9, color=TEXT_MUTED),
    ))
    fig.update_xaxes(**GRID_STYLE)
    fig.update_yaxes(tickfont=dict(size=9, color=TEXT_MUTED))
    return _apply_base(fig, "Pipeline Node Latency (ms)", height=240)


def quality_radar_chart(data: list[dict]) -> go.Figure:
    """Filled radar: quality dimensions."""
    df = pd.DataFrame(data)
    fig = go.Figure(go.Scatterpolar(
        r=df["value"],
        theta=df["metric"],
        fill="toself",
        fillcolor=_rgba(PRIMARY, 0.2),
        line=dict(color=PRIMARY, width=2),
        marker=dict(color=PRIMARY, size=5),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor=_TRANSPARENT,
            radialaxis=dict(
                visible=True, range=[0, 100],
                color=TEXT_DIM, gridcolor=BORDER,
            ),
            angularaxis=dict(color=TEXT_MUTED),
        ),
        **LAYOUT_BASE,
        height=240,
        title_text="Quality Radar — Final System",
    )
    return fig


def token_savings_chart(data: list[dict]) -> go.Figure:
    """Area chart: cumulative token savings from cache."""
    df = pd.DataFrame(data)
    fig = go.Figure(go.Scatter(
        x=df["query"], y=df["saved"],
        mode="lines",
        fill="tozeroy",
        fillcolor=_rgba(GREEN, 0.13),
        line=dict(color=GREEN, width=2),
        name="Tokens Saved",
    ))
    fig.update_xaxes(**GRID_STYLE, tickfont=dict(size=9))
    fig.update_yaxes(**GRID_STYLE, tickfont=dict(size=9))
    return _apply_base(fig, "Token Savings via Semantic Cache", height=200)


def session_sparkline(queries: list, tokens: list) -> go.Figure:
    """Tiny area chart for the sidebar telemetry."""
    fig = go.Figure(go.Scatter(
        x=list(range(len(tokens))), y=tokens,
        mode="lines",
        fill="tozeroy",
        fillcolor=_rgba(PRIMARY, 0.13),
        line=dict(color=PRIMARY, width=1.5),
    ))
    fig.update_xaxes(**NO_GRID)
    fig.update_yaxes(**NO_GRID)
    return _apply_base(fig, "Session Token Spikes", height=90)


def waterfall_latency(latencies: dict, total: float, height: int = 200) -> go.Figure:
    """Horizontal bar latency waterfall from live query data."""
    if not latencies:
        phases = ["Network / API"]
        times  = [total]
    else:
        phases = list(latencies.keys())
        times  = list(latencies.values())
        measured = sum(times)
        if total > measured:
            phases.append("Overhead")
            times.append(total - measured)

    fig = go.Figure(go.Bar(
        y=phases[::-1],
        x=times[::-1],
        orientation="h",
        marker=dict(color=PRIMARY, line=dict(color=SURFACE, width=1)),
        text=[f"{v:.2f}s" for v in times[::-1]],
        textposition="outside",
        textfont=dict(size=9, color=TEXT_MUTED),
    ))
    fig.update_xaxes(**GRID_STYLE)
    fig.update_yaxes(tickfont=dict(size=9, color=TEXT_MUTED))
    return _apply_base(fig, f"Live Node Latency: {total:.2f}s total", height=height)


def knowledge_graph(query: str, trace: list, citations: list, height: int = 200) -> go.Figure:
    """NetworkX spring-layout knowledge graph."""
    G = nx.Graph()
    G.add_node("Query", node_type="query", size=22)

    subqueries = [
        t.replace("Searched for:", "").strip()
        for t in trace if "Searched for:" in t
    ]

    if subqueries:
        for sq in subqueries:
            short = sq[:22] + "..." if len(sq) > 22 else sq
            G.add_node(short, node_type="subquery", size=15)
            G.add_edge("Query", short)
            for doc in set(citations):
                G.add_node(doc, node_type="doc", size=9)
                G.add_edge(short, doc)
    else:
        G.add_node("Cache Hit", node_type="subquery", size=15)
        G.add_edge("Query", "Cache Hit")

    pos = nx.spring_layout(G, seed=42)

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x, node_y, node_text, node_colors, node_sizes = [], [], [], [], []
    color_map = {"query": ACCENT, "subquery": PRIMARY, "doc": "#ffffff"}
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        nt = G.nodes[node].get("node_type", "doc")
        node_text.append(node if nt != "doc" or len(G.nodes()) <= 10 else "")
        node_colors.append(color_map[nt])
        node_sizes.append(G.nodes[node].get("size", 9))

    fig = go.Figure([
        go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(width=0.6, color="rgba(255,255,255,0.15)"),
            hoverinfo="none",
        ),
        go.Scatter(
            x=node_x, y=node_y, mode="markers+text",
            text=node_text, textposition="bottom center",
            hovertext=list(G.nodes()),
            marker=dict(
                color=node_colors, size=node_sizes,
                line=dict(width=1, color=SURFACE),
            ),
            textfont=dict(color="#ffffff", size=8),
        ),
    ])
    fig.update_xaxes(**NO_GRID)
    fig.update_yaxes(**NO_GRID)
    fig.update_layout(
        **LAYOUT_BASE, height=height,
        title_text="Dynamic Knowledge Graph",
        showlegend=False,
    )
    return fig


def agent_lifecycle_chart() -> go.Figure:
    """Static pipeline pathway shown before first query."""
    stages = ["Plan", "Rewrite", "Retrieve", "Rerank", "Reason", "Reflect", "Respond"]
    colors = [PRIMARY, PRIMARY, ACCENT, ACCENT, GREEN, GREEN, TEAL]
    fig = go.Figure(go.Scatter(
        x=list(range(len(stages))), y=[1] * len(stages),
        mode="markers+text+lines",
        text=stages,
        textposition="top center",
        marker=dict(size=14, color=colors,
                    line=dict(width=2, color=SURFACE)),
        line=dict(color="rgba(255,255,255,0.15)", width=2),
        textfont=dict(color=TEXT_MUTED, size=9),
    ))
    fig.update_xaxes(**NO_GRID)
    fig.update_yaxes(**NO_GRID, range=[0.5, 1.5])
    return _apply_base(fig, "Agent Execution Pathway", height=110)


def bullet_gauge(value: float, title: str, is_pct: bool = True) -> go.Figure:
    """Compact bullet gauge for sidebar.

    Note: does NOT use LAYOUT_BASE because LAYOUT_BASE already contains
    a 'margin' key — passing a second margin would raise:
      TypeError: update_layout() got multiple values for keyword argument 'margin'
    """
    display = value * 100 if is_pct else value
    ref_max = 100 if is_pct else 1.0
    color = PRIMARY if value < 0.75 else (ACCENT if value < 0.9 else RED)

    fig = go.Figure(go.Indicator(
        mode="number+gauge",
        value=display,
        number={"suffix": "%" if is_pct else "",
                "font": {"size": 18, "color": TEXT, "family": MONO}},
        title={"text": title, "font": {"size": 10, "color": TEXT_MUTED}},
        gauge={
            "shape": "bullet",
            "axis": {"range": [0, ref_max], "visible": False},
            "bar": {"color": color, "thickness": 1},
            "bgcolor": SURFACE2,
            "borderwidth": 0,
        },
    ))
    # Explicitly enumerate all keys — do NOT spread LAYOUT_BASE here
    fig.update_layout(
        paper_bgcolor=_TRANSPARENT,
        plot_bgcolor=_TRANSPARENT,
        font=dict(color=TEXT_MUTED, family=MONO, size=10),
        title_font=dict(size=12, color=TEXT_MUTED, family=MONO),
        margin=dict(l=110, r=20, t=20, b=20),
        height=60,
    )
    return fig