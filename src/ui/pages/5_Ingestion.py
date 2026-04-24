"""
FinSight-Alpha Dashboard — Data Ingestion page
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from components.theme import inject_css, PRIMARY, ACCENT, GREEN, RED, PURPLE, TEAL
from components.theme import SURFACE, SURFACE2, BORDER, TEXT, TEXT_MUTED, TEXT_DIM, MONO, DISPLAY
from components.ui_components import section_title, badge_html

st.set_page_config(page_title="Ingestion · FinSight-Alpha", page_icon="📥", layout="wide")
inject_css()

section_title(
    "📥", "Data Ingestion Pipeline",
    "SEC EDGAR scraping → document parsing → semantic chunking → dual-index construction.",
)

# 4-step pipeline
steps = [
    {
        "step": "01", "name": "SEC EDGAR Scraper", "color": PRIMARY,
        "desc": (
            "Downloads 10-K, 10-Q, and 8-K filings for given tickers via "
            "<code>sec-edgar-downloader</code>. Respects SEC rate limits and "
            "preserves the original directory structure under <code>data/raw/</code>."
        ),
        "tech": "sec-edgar-downloader",
        "config": [
            ("Tickers",    "Any valid US equity ticker"),
            ("Form types", "10-K, 10-Q, 8-K (configurable list)"),
            ("Limit",      "1 most recent filing per type"),
            ("User-Agent", "Company + email (SEC requirement)"),
        ],
    },
    {
        "step": "02", "name": "Document Processor", "color": ACCENT,
        "desc": (
            "Parses raw HTML/TXT with <code>unstructured</code>. Chunks by section "
            "titles to preserve semantic context. Large files are split into 800k-char "
            "blocks first to respect spaCy's NLP memory limits."
        ),
        "tech": "unstructured.io",
        "config": [
            ("Max chunk chars",   "2,000"),
            ("Overlap",           "200 characters"),
            ("Min combine chars", "500 (merges small blocks)"),
            ("Output format",     "JSONL per filing"),
        ],
    },
    {
        "step": "03", "name": "Index Builder", "color": GREEN,
        "desc": (
            "Encodes all chunks with <code>all-MiniLM-L6-v2</code> in batches of 500 "
            "(GPU-accelerated on RTX 4070). Upserts into Qdrant cosine collection. "
            "Simultaneously tokenises corpus for BM25Okapi."
        ),
        "tech": "Qdrant + BM25Okapi",
        "config": [
            ("Embedding model",  "all-MiniLM-L6-v2 (384-dim)"),
            ("Distance metric",  "Cosine similarity"),
            ("Batch size",       "500 chunks per upsert"),
            ("BM25 tokeniser",   "Whitespace split → lowercase"),
        ],
    },
    {
        "step": "04", "name": "Persistence", "color": PURPLE,
        "desc": (
            "Qdrant vectors persisted to disk at <code>data/qdrant_db/</code>. "
            "BM25 index and full corpus metadata pickled together to "
            "<code>data/bm25_index.pkl</code> for fast reload at query time."
        ),
        "tech": "Local disk storage",
        "config": [
            ("Dense store",   "data/qdrant_db/ (Qdrant local)"),
            ("Sparse store",  "data/bm25_index.pkl"),
            ("Reload",        "load_bm25() on first search call"),
            ("Collection",    "sec_filings (recreated on rebuild)"),
        ],
    },
]

for i, s in enumerate(steps):
    c = s["color"]
    config_html = "".join(
        f"<div style='display:flex;justify-content:space-between;padding:4px 0;"
        f"border-bottom:1px solid {BORDER};'>"
        f"<span style='font-size:11px;color:{TEXT_MUTED};'>{k}</span>"
        f"<span style='font-family:{MONO};font-size:11px;color:{TEXT};'>{v}</span>"
        f"</div>"
        for k, v in s["config"]
    )
    col_desc, col_conf = st.columns([3, 2], gap="large")
    with col_desc:
        st.markdown(
            f"<div style='background:{SURFACE};border:1px solid {BORDER};border-left:3px solid {c};"
            f"border-radius:10px;padding:20px;height:100%;'>"
            f"<div style='font-family:{DISPLAY};font-size:26px;font-weight:800;color:{c}40;'>{s['step']}</div>"
            f"<div style='font-family:{DISPLAY};font-size:16px;font-weight:700;color:{c};margin-bottom:10px;'>{s['name']}</div>"
            f"<div style='font-size:13px;color:{TEXT_MUTED};line-height:1.7;margin-bottom:12px;'>{s['desc']}</div>"
            + badge_html(s["tech"], c)
            + "</div>",
            unsafe_allow_html=True,
        )
    with col_conf:
        st.markdown(
            f"<div style='background:{SURFACE};border:1px solid {BORDER};border-radius:10px;padding:18px;height:100%;'>"
            f"<div style='font-family:{MONO};font-size:10px;color:{TEXT_DIM};letter-spacing:0.08em;margin-bottom:12px;'>"
            f"CONFIGURATION</div>"
            f"{config_html}"
            f"</div>",
            unsafe_allow_html=True,
        )
    if i < len(steps) - 1:
        st.markdown(
            f"<div style='text-align:center;padding:6px 0;font-size:20px;color:{TEXT_DIM};'>↓</div>",
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# Index architecture
section_title("🗄️", "Dual Index Architecture")

col_dense, col_sparse = st.columns(2, gap="large")

with col_dense:
    st.markdown(
        f"<div style='background:{SURFACE};border:1px solid {PRIMARY}40;"
        f"border-radius:10px;padding:20px;'>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='font-family:{MONO};font-size:11px;color:{PRIMARY};font-weight:700;margin-bottom:12px;'>"
        f"DENSE INDEX (Qdrant)</div>",
        unsafe_allow_html=True,
    )
    dense_items = [
        ("Embedding model",  "all-MiniLM-L6-v2",         PRIMARY),
        ("Vector size",      "384 dimensions",            TEXT),
        ("Distance",         "Cosine similarity",         TEXT),
        ("Candidate fetch",  "fetch_k = 50 per query",    ACCENT),
        ("Device",           "CUDA (RTX 4070 8GB VRAM)",  GREEN),
        ("Storage",          "data/qdrant_db/ (local)",   TEXT_MUTED),
    ]
    for k, v, c in dense_items:
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;padding:6px 0;"
            f"border-bottom:1px solid {BORDER};'>"
            f"<span style='font-size:12px;color:{TEXT_MUTED};'>{k}</span>"
            f"<span style='font-family:{MONO};font-size:11px;color:{c};'>{v}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

with col_sparse:
    st.markdown(
        f"<div style='background:{SURFACE};border:1px solid {ACCENT}40;"
        f"border-radius:10px;padding:20px;'>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='font-family:{MONO};font-size:11px;color:{ACCENT};font-weight:700;margin-bottom:12px;'>"
        f"SPARSE INDEX (BM25)</div>",
        unsafe_allow_html=True,
    )
    sparse_items = [
        ("Algorithm",     "BM25Okapi",                   ACCENT),
        ("Tokeniser",     "Whitespace split + lowercase", TEXT),
        ("Scoring",       "get_scores() — all docs",      TEXT),
        ("Candidate fetch","Top-50 by BM25 score",        ACCENT),
        ("Persistence",   "Pickled to bm25_index.pkl",    TEXT_MUTED),
        ("Reload",        "Lazy on first search call",    GREEN),
    ]
    for k, v, c in sparse_items:
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

# RRF explanation
section_title("🔀", "Reciprocal Rank Fusion (RRF)")

st.markdown(
    f"<div style='background:{SURFACE};border:1px solid {BORDER};border-radius:10px;padding:20px;'>",
    unsafe_allow_html=True,
)
col_formula, col_detail = st.columns([1, 2], gap="large")

with col_formula:
    st.markdown(
        f"<div style='background:{SURFACE2};border-radius:8px;padding:20px;text-align:center;'>"
        f"<div style='font-family:{MONO};font-size:14px;color:{TEXT_MUTED};margin-bottom:8px;'>RRF SCORE</div>"
        f"<div style='font-family:{MONO};font-size:18px;color:{PRIMARY};font-weight:700;'>"
        f"Σ 1 / (k + rank<sub style='font-size:11px;'>i</sub>)</div>"
        f"<div style='font-family:{MONO};font-size:11px;color:{TEXT_DIM};margin-top:10px;'>k = 60 (smoothing constant)</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

with col_detail:
    st.markdown(
        f"<p style='font-size:13px;color:{TEXT_MUTED};line-height:1.7;'>"
        f"Both dense (Qdrant) and sparse (BM25) retrievers independently rank "
        f"<code>fetch_k=50</code> candidates. RRF combines these ranked lists by summing "
        f"reciprocal ranks for each document. The smoothing constant <code>k=60</code> "
        f"prevents high sensitivity to top-ranked results. The merged top-20 are then "
        f"passed to the cross-encoder reranker for precise rescoring.</p>"
        f"<div style='display:flex;gap:10px;flex-wrap:wrap;margin-top:8px;'>"
        + badge_html("BM25 top-50", ACCENT)
        + badge_html("Qdrant top-50", PRIMARY)
        + badge_html("RRF → top-20", GREEN)
        + badge_html("CrossEncoder → top-N", PURPLE)
        + "</div>",
        unsafe_allow_html=True,
    )
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# JSONL chunk format
section_title("📄", "Processed Chunk Format (JSONL)")
st.markdown(
    f"<div style='background:{SURFACE};border:1px solid {BORDER};border-radius:10px;padding:20px;'>",
    unsafe_allow_html=True,
)
st.code(
    """{
  "page_content": "Net revenue increased 122% year-over-year to $44.1B driven primarily by...",
  "metadata": {
    "source": "data/raw/sec-edgar-filings/NVDA/10-K/0001045810-24/primary-document.htm",
    "element_type": "CompositeElement"
  }
}""",
    language="json",
)
st.markdown(
    f"<div style='margin-top:10px;font-family:{MONO};font-size:11px;color:{TEXT_DIM};'>"
    f"One JSON object per line. Output path pattern: "
    f"<code>data/processed/{{TICKER}}_{{FORM}}_{{ACCESSION}}_chunks.jsonl</code></div>",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)