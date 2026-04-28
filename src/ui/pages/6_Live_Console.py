"""
FinSight-Alpha Dashboard — Live Console (Chat + Diagnostics)

Layout: sidebar telemetry | chat (wide) | diagnostics panel
"""

import sys, re, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import requests

from components.theme import (
    inject_css,
    PRIMARY, ACCENT, GREEN, RED, PURPLE, TEAL,
    SURFACE, SURFACE2, BORDER,
    TEXT, TEXT_MUTED, TEXT_DIM,
    MONO, DISPLAY,
)
from components.ui_components import badge_html, live_dot_html
from components.charts import waterfall_latency, knowledge_graph, agent_lifecycle_chart

API_URL    = "http://localhost:8000/chat"
HEALTH_URL = "http://localhost:8000/health"
CACHE_URL  = "http://localhost:8000/cache/stats"

st.set_page_config(
    page_title="Live Console · FinSight-Alpha",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

st.info(
    "**Note:** This is a cloud-hosted UI demo. To use the full Agentic RAG backend, please download the repository and run it locally with a GPU."
)
# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "telemetry" not in st.session_state:
    st.session_state.telemetry = {"queries": [], "rtt": [], "tokens": [], "cache_hits": []}


# API helpers
@st.cache_data(ttl=5)
def get_health():
    try:
        r = requests.get(HEALTH_URL, timeout=2)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return {"budget": {"total_tokens": 0, "usage_pct": 0, "tier": "UNKNOWN"}, "model_info": {}}


@st.cache_data(ttl=5)
def get_cache():
    try:
        r = requests.get(CACHE_URL, timeout=2)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return {"hit_rate": 0, "total_hits": 0, "total_misses": 0, "size": 0}


health = get_health()
cache  = get_cache()
budget = health.get("budget", {})

# Helper: coloured stat card
def stat_card(label: str, value: str, color: str = PRIMARY, sub: str = "") -> str:
    return (
        f"<div style='background:{SURFACE};border:1px solid {BORDER};"
        f"border-left:3px solid {color};border-radius:8px;"
        f"padding:12px 14px;margin-bottom:10px;'>"
        f"<div style='font-family:{MONO};font-size:9px;color:{TEXT_DIM};"
        f"letter-spacing:0.1em;text-transform:uppercase;margin-bottom:4px;'>{label}</div>"
        f"<div style='font-family:{DISPLAY};font-size:22px;font-weight:800;"
        f"color:{color};line-height:1;'>{value}</div>"
        + (f"<div style='font-family:{MONO};font-size:9px;color:{TEXT_DIM};margin-top:3px;'>{sub}</div>" if sub else "")
        + "</div>"
    )


def progress_bar(value_pct: float, color: str = PRIMARY) -> str:
    w = min(max(value_pct * 100, 0), 100)
    return (
        f"<div style='height:5px;background:{SURFACE2};border-radius:3px;margin-bottom:10px;'>"
        f"<div style='width:{w:.0f}%;height:100%;background:{color};"
        f"border-radius:3px;transition:width 0.4s ease;'></div></div>"
    )


# SIDEBAR — telemetry
with st.sidebar:
    st.markdown(
        f"<div style='padding:4px 0 14px;'>"
        f"<div style='font-family:{DISPLAY};font-size:18px;font-weight:800;'>"
        f"<span style='color:{PRIMARY};'>Fin</span>"
        f"<span style='color:{TEXT};'>Sight</span>"
        f"<span style='color:{ACCENT};font-family:{MONO};font-size:12px;margin-left:2px;'>α</span>"
        f"</div>"
        f"<div style='font-family:{MONO};font-size:9px;color:{TEXT_DIM};"
        f"letter-spacing:0.1em;margin-top:3px;'>LIVE CONSOLE</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div style='display:flex;align-items:center;gap:6px;margin-bottom:14px;'>"
        f"{live_dot_html(GREEN)}"
        f"<span style='font-family:{MONO};font-size:9px;color:{GREEN};"
        f"letter-spacing:0.08em;'>PIPELINE ONLINE</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # Telemetry stats
    st.markdown(
        f"<div style='font-family:{MONO};font-size:9px;color:{TEXT_DIM};"
        f"letter-spacing:0.1em;margin-bottom:10px;'>SYSTEM TELEMETRY</div>",
        unsafe_allow_html=True,
    )

    pct       = budget.get("usage_pct", 0)
    hit_rate  = cache.get("hit_rate", 0)
    tier      = str(budget.get("tier", "UNKNOWN")).upper()
    total_tok = budget.get("total_tokens", 0)
    tier_color = {"GREEN": GREEN, "YELLOW": ACCENT, "RED": RED}.get(tier, TEXT_MUTED)

    rtt_list = st.session_state.telemetry["rtt"]
    avg_rtt  = f"{sum(rtt_list)/len(rtt_list):.2f}s" if rtt_list else "—"
    q_count  = len(st.session_state.telemetry["queries"])

    # Budget card
    st.markdown(
        stat_card("Budget Burn", f"{pct*100:.1f}%", tier_color, f"{total_tok:,} tokens used"),
        unsafe_allow_html=True,
    )
    st.markdown(progress_bar(pct, tier_color), unsafe_allow_html=True)

    # Cache hit card
    st.markdown(
        stat_card("Cache Hit Rate", f"{hit_rate*100:.0f}%", GREEN,
                  f"{cache.get('total_hits',0)} hits / {cache.get('total_misses',0)} misses"),
        unsafe_allow_html=True,
    )
    st.markdown(progress_bar(hit_rate, GREEN), unsafe_allow_html=True)

    # Mean RTT
    st.markdown(stat_card("Mean RTT", avg_rtt, PRIMARY, f"over {q_count} queries"), unsafe_allow_html=True)

    # Tier badge
    st.markdown(
        f"<div style='background:{SURFACE};border:1px solid {BORDER};"
        f"border-left:3px solid {tier_color};border-radius:8px;"
        f"padding:12px 14px;margin-bottom:10px;'>"
        f"<div style='font-family:{MONO};font-size:9px;color:{TEXT_DIM};"
        f"letter-spacing:0.1em;margin-bottom:6px;'>BUDGET TIER</div>"
        f"<div style='display:flex;align-items:center;gap:8px;'>"
        f"{badge_html(tier, tier_color)}"
        f"<span style='font-family:{MONO};font-size:10px;color:{TEXT_MUTED};'>"
        f"{'Full pipeline' if tier=='GREEN' else 'Reduced mode' if tier=='YELLOW' else 'Minimal mode'}"
        f"</span></div></div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # Quick links back to other pages
    st.markdown(
        f"<div style='font-family:{MONO};font-size:9px;color:{TEXT_DIM};"
        f"letter-spacing:0.1em;margin-bottom:8px;'>NAVIGATION</div>",
        unsafe_allow_html=True,
    )
    st.page_link("Home.py",                 label="Overview",     icon="🏠")
    st.page_link("pages/1_Architecture.py", label="Architecture", icon="🏗️")
    st.page_link("pages/2_Performance.py",  label="Performance",  icon="📊")
    st.page_link("pages/3_Optimisation.py", label="Optimisation", icon="⚡")
    st.page_link("pages/4_Evaluation.py",   label="Evaluation",   icon="🧪")
    st.page_link("pages/5_Ingestion.py",    label="Ingestion",    icon="⬇️")


# MAIN — 2-column: chat | diagnostics
col_chat, col_diag = st.columns([3, 2], gap="large")

# CHAT column
with col_chat:
    st.markdown(
        f"<div style='font-family:{MONO};font-size:10px;color:{TEXT_MUTED};"
        f"letter-spacing:0.1em;margin-bottom:12px;'>"
        f"{live_dot_html(GREEN)}OPERATIONAL CONSOLE</div>",
        unsafe_allow_html=True,
    )

    chat_container = st.container(height=600)

    with chat_container:
        if not st.session_state.messages:
            st.markdown(
                f"<div style='padding:28px 12px;'>"
                f"<h3 style='color:{TEXT};margin-bottom:14px;font-family:{DISPLAY};'>⚡ Ready for Intel</h3>"
                f"<div style='display:flex;flex-wrap:wrap;gap:8px;margin-bottom:18px;'>"
                + badge_html("LLM Router Active", PRIMARY)
                + badge_html("Hybrid Index Online", ACCENT)
                + badge_html("MRR Checks Ready", GREEN)
                + f"</div>"
                f"<p style='color:{TEXT_MUTED};font-size:13px;line-height:1.8;margin-bottom:18px;'>"
                f"The pipeline executes a deterministic "
                f"<code>Plan → Rewrite → Retrieve → Reason → Reflect</code> loop "
                f"with self-correcting hallucination detection on every query.</p>"
                f"<hr style='border:none;border-top:1px solid {BORDER};margin:0 0 16px;'>"
                f"<div style='font-family:{MONO};font-size:10px;color:{TEXT_DIM};"
                f"letter-spacing:0.08em;margin-bottom:10px;'>EXAMPLE QUERIES</div>"
                f"<div style='display:flex;flex-direction:column;gap:8px;'>"
                + "".join(
                    f"<div style='padding:10px 14px;background:{SURFACE2};"
                    f"border:1px solid {BORDER};border-radius:6px;"
                    f"font-family:{MONO};font-size:11px;color:{TEXT_MUTED};'>{q}</div>"
                    for q in [
                        "What are NVIDIA's primary strategies for supply chain risk?",
                        "Compare NVIDIA data center revenue FY23 vs FY24",
                        "What risks does NVIDIA highlight in its most recent 10-K?",
                    ]
                )
                + "</div></div>",
                unsafe_allow_html=True,
            )

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    # Citations
                    citations = list(set(re.findall(r'\[Doc\s*\d+:\s*([^\]]+)\]', msg["content"])))
                    if citations:
                        cit_html = "".join(
                            f"<span style='display:inline-block;background:{SURFACE2};"
                            f"color:{ACCENT};border:1px solid {BORDER};border-radius:4px;"
                            f"padding:3px 10px;font-size:11px;margin:3px 4px 0 0;"
                            f"font-family:{MONO};'>{c}</span>"
                            for c in citations
                        )
                        st.markdown(
                            f"<div style='margin-top:10px;'>"
                            f"<span style='font-size:10px;color:{TEXT_DIM};"
                            f"font-family:{MONO};letter-spacing:0.06em;'>SOURCES CITED: </span>"
                            f"{cit_html}</div>",
                            unsafe_allow_html=True,
                        )
                    # Reasoning trace
                    if msg.get("trace"):
                        with st.expander("🔍 Reasoning trace", expanded=False):
                            for t in msg["trace"]:
                                icon = "📋" if t.startswith("Plan") else "🔎" if "Searched" in t else "💭"
                                st.markdown(
                                    f"<div style='font-family:{MONO};font-size:11px;"
                                    f"color:{TEXT_MUTED};padding:4px 0;"
                                    f"border-bottom:1px solid {BORDER};'>"
                                    f"{icon} {t}</div>",
                                    unsafe_allow_html=True,
                                )

    # Chat input
    if prompt := st.chat_input("Enter strategic financial query..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                status_ph = st.empty()
                status_ph.markdown(
                    f"<div style='padding:8px 0;font-family:{MONO};font-size:12px;"
                    f"color:{TEXT_MUTED};'>"
                    f"{live_dot_html(ACCENT)} Executing Plan → Rewrite → Retrieve loop…</div>",
                    unsafe_allow_html=True,
                )
                t0 = time.time()
                try:
                    res = requests.post(API_URL, json={"query": prompt}, timeout=120)
                    rtt = time.time() - t0
                    status_ph.empty()

                    if res.status_code == 200:
                        data  = res.json()
                        ans   = data.get("answer", "Error: no answer returned")
                        trace = data.get("reasoning_trace", [])
                        st.markdown(ans)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": ans,
                            "trace": trace,
                            "rtt": rtt,
                            "query": prompt,
                            "latencies": data.get("latencies", {}),
                            "retrieval_metrics": data.get("retrieval_metrics", {}),
                            "cache_hit": data.get("cache_hit", False),
                        })
                        st.session_state.telemetry["queries"].append(prompt)
                        st.session_state.telemetry["rtt"].append(rtt)
                        st.session_state.telemetry["tokens"].append(data.get("tokens_used", 0))
                        st.session_state.telemetry["cache_hits"].append(1 if data.get("cache_hit") else 0)
                        st.rerun()
                    else:
                        status_ph.error(f"API Error {res.status_code}: {res.text[:200]}")

                except requests.exceptions.ConnectionError:
                    status_ph.error(
                        "🚨 **Backend API Not Found (Demo Mode)**\n\n"
                        "You are currently viewing the **cloud-hosted UI demo**, which does not run the backend API. "
                        "To execute live queries, you must run the entire system locally:\n\n"
                        "**1.** Clone the repo: `git clone https://github.com/BhargavKumarNath/FinSightAlpha.git`\n"
                        "**2.** Install backend dependencies: `pip install -r requirements-backend.txt`\n"
                        "**3.** Start the AI server: `uvicorn src.main:app --host 0.0.0.0 --port 8000`\n"
                        "**4.** Start this UI locally: `streamlit run src/ui/Home.py`\n\n"
                        "*(Ensure you have a configured GPU environment if you are running the LLMs locally, or proper API keys in `.env`)*"
                    )
                except Exception as e:
                    status_ph.error(f"Execution failure: {e}")


# DIAGNOSTICS column
with col_diag:
    st.markdown(
        f"<div style='font-family:{MONO};font-size:10px;color:{TEXT_MUTED};"
        f"letter-spacing:0.1em;margin-bottom:12px;'>SYSTEM DIAGNOSTICS</div>",
        unsafe_allow_html=True,
    )

    latest = next(
        (m for m in reversed(st.session_state.messages)
         if m["role"] == "assistant" and "trace" in m),
        None,
    )

    if latest:
        metrics   = latest.get("retrieval_metrics", {})
        mrr       = metrics.get("MRR", 0.0)
        ndcg      = metrics.get("NDCG", 0.0)
        cache_hit = latest.get("cache_hit", False)

        # MRR / NDCG / RTT
        m1, m2, m3 = st.columns(3)
        m1.markdown(stat_card("Live MRR",  f"{mrr:.2f}",              PRIMARY), unsafe_allow_html=True)
        m2.markdown(stat_card("Live NDCG", f"{ndcg:.2f}",             PURPLE),  unsafe_allow_html=True)
        m3.markdown(stat_card("RTT",       f"{latest.get('rtt',0):.1f}s", ACCENT), unsafe_allow_html=True)

        if cache_hit:
            st.markdown(
                f"<div style='padding:8px 14px;background:{GREEN}15;"
                f"border:1px solid {GREEN}44;border-radius:6px;margin-bottom:10px;"
                f"font-family:{MONO};font-size:11px;color:{GREEN};'>"
                f"⚡ Semantic cache hit — full pipeline bypassed</div>",
                unsafe_allow_html=True,
            )

        # Latency waterfall
        st.markdown(
            f"<div style='background:{SURFACE};border:1px solid {BORDER};"
            f"border-radius:10px;padding:16px;margin-bottom:14px;'>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            waterfall_latency(latest.get("latencies", {}), latest.get("rtt", 0.0), height=300),
            width='stretch',
            config={"displayModeBar": False},
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Knowledge graph
        citations = re.findall(r'\[Doc\s*\d+:\s*([^\]]+)\]', latest["content"])
        st.markdown(
            f"<div style='background:{SURFACE};border:1px solid {BORDER};"
            f"border-radius:10px;padding:16px;margin-bottom:14px;'>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            knowledge_graph(
                latest.get("query", ""), latest["trace"], citations, height=320
            ),
            width='stretch',
            config={"displayModeBar": False},
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Trace summary pills
        subqueries = [t.replace("Searched for:", "").strip()
                      for t in latest["trace"] if "Searched for:" in t]
        if subqueries:
            st.markdown(
                f"<div style='font-family:{MONO};font-size:9px;color:{TEXT_DIM};"
                f"letter-spacing:0.1em;margin-bottom:8px;'>SUB-QUERIES EXECUTED</div>",
                unsafe_allow_html=True,
            )
            for sq in subqueries:
                st.markdown(
                    f"<div style='padding:6px 10px;background:{SURFACE2};"
                    f"border-left:2px solid {PRIMARY};border-radius:4px;"
                    f"font-family:{MONO};font-size:10px;color:{TEXT_MUTED};"
                    f"margin-bottom:5px;'>{sq}</div>",
                    unsafe_allow_html=True,
                )

    else:
        # Pre-query state
        st.markdown(
            f"<div style='background:{SURFACE};border:1px solid {BORDER};"
            f"border-radius:10px;padding:16px;margin-bottom:14px;'>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            agent_lifecycle_chart(),
            width='stretch',
            config={"displayModeBar": False},
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Pipeline node status cards
        st.markdown(
            f"<div style='font-family:{MONO};font-size:9px;color:{TEXT_DIM};"
            f"letter-spacing:0.1em;margin-bottom:10px;'>PIPELINE NODES</div>",
            unsafe_allow_html=True,
        )
        nodes = [
            ("PLAN",     "Query decomposition",        PRIMARY),
            ("REWRITE",  "Sub-query generation",       PRIMARY),
            ("RETRIEVE", "Hybrid BM25 + Qdrant + RRF", ACCENT),
            ("RERANK",   "Cross-encoder rescoring",    ACCENT),
            ("REASON",   "LLaMA 3.3 70B synthesis",    GREEN),
            ("REFLECT",  "Hallucination detection",    GREEN),
        ]
        for name, desc, color in nodes:
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:10px;"
                f"padding:8px 10px;background:{SURFACE};"
                f"border:1px solid {BORDER};border-radius:6px;margin-bottom:6px;'>"
                f"<div style='width:6px;height:6px;border-radius:50%;"
                f"background:{color};flex-shrink:0;'></div>"
                f"<div>"
                f"<div style='font-family:{MONO};font-size:10px;font-weight:700;"
                f"color:{color};'>{name}</div>"
                f"<div style='font-size:11px;color:{TEXT_DIM};'>{desc}</div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            f"<div style='margin-top:14px;padding:12px 14px;"
            f"background:{PRIMARY}10;border:1px solid {PRIMARY}30;"
            f"border-radius:8px;font-size:12px;color:{TEXT_MUTED};line-height:1.7;'>"
            f"Submit a query to see live latency waterfall, knowledge graph, "
            f"and MRR / NDCG retrieval metrics.</div>",
            unsafe_allow_html=True,
        )