"""
FinSight-Alpha — Dashboard data: real artifacts where they exist, clearly
labeled illustrative placeholders where no real artifact exists yet.

Feeds scripts/precompute.py's frontend/content/generated/*.json output.
The Streamlit UI this module used to feed was retired; nothing here is
UI-specific, so it lives under src/reporting/ rather than a UI package.
Trimmed to just the exports precompute.py actually reads -- the rest of
the original data.py (RAGAS loading, illustrative latency/cache shapes,
capability/tech-stack overview rows, agent state field docs) had no other
consumer once the Streamlit pages that rendered them were deleted.

Real sources:
  - BUDGET_TIERS / MODEL_ROUTER_TABLE / CACHE_CONFIG_ROWS /
    CONTEXT_WINDOW_DETAILS  <- src.optimization.config.OptimizationConfig
                              + src.optimization.token_budget.TokenBudgetManager

FILING_TYPES, PIPELINE_NODES, and ROUTING_RULES are static architecture
description, not a measured metric -- out of scope for real-data sourcing.
"""

import contextlib
import io

from src.optimization.config import config
from src.optimization.token_budget import TokenBudgetManager
from src.reporting.colors import ACCENT, GREEN, PRIMARY, PURPLE, RED, TEAL, TEXT_MUTED


def _budget_tier_snapshot():
    """Exercise the real TokenBudgetManager at each tier boundary and read
    back its actual decisions, instead of hand-copying branch constants."""
    mgr = TokenBudgetManager()
    snapshot = {}
    with contextlib.redirect_stdout(io.StringIO()):  # suppress its per-call logging
        for tier_name, used_tokens in (
            ("GREEN", 0),
            ("YELLOW", mgr.yellow_threshold),
            ("RED", mgr.red_threshold),
        ):
            mgr.reset()
            if used_tokens:
                mgr.record_usage(input_tokens=used_tokens)
            snapshot[tier_name] = {
                "loops": mgr.get_max_iterations(),
                "top_n": mgr.get_retrieval_top_n(),
                "skip_planner": mgr.should_skip_planner(),
                "heavy_model": mgr.should_use_heavy_model(),
            }
    return snapshot, mgr.yellow_threshold, mgr.red_threshold


def _fmt_tokens(n: int) -> str:
    return f"{n // 1000}k" if n % 1000 == 0 else str(n)


_tier_snapshot, _yellow_threshold, _red_threshold = _budget_tier_snapshot()

BUDGET_TIERS = [
    {
        "tier": "GREEN",
        "range": f"0 – {_fmt_tokens(_yellow_threshold)}",
        "color": GREEN,
        "loops": _tier_snapshot["GREEN"]["loops"],
        "top_n": _tier_snapshot["GREEN"]["top_n"],
        "model": config.model_heavy,
        "desc": "Full pipeline. All nodes active.",
        "skips": [],
    },
    {
        "tier": "YELLOW",
        "range": f"{_fmt_tokens(_yellow_threshold)} – {_fmt_tokens(_red_threshold)}",
        "color": ACCENT,
        "loops": _tier_snapshot["YELLOW"]["loops"],
        "top_n": _tier_snapshot["YELLOW"]["top_n"],
        "model": config.model_heavy
        if _tier_snapshot["YELLOW"]["heavy_model"]
        else config.model_light,
        "desc": "Planner skipped. Reduced loops and retrieval.",
        "skips": ["Planner"] if _tier_snapshot["YELLOW"]["skip_planner"] else [],
    },
    {
        "tier": "RED",
        "range": f"{_fmt_tokens(_red_threshold)}+",
        "color": RED,
        "loops": _tier_snapshot["RED"]["loops"],
        "top_n": _tier_snapshot["RED"]["top_n"],
        "model": config.model_heavy
        if _tier_snapshot["RED"]["heavy_model"]
        else config.model_light,
        "desc": "Minimum viable path. Budget preservation mode.",
        "skips": (["Planner"] if _tier_snapshot["RED"]["skip_planner"] else [])
        + ([] if _tier_snapshot["RED"]["heavy_model"] else ["Heavy Model"]),
    },
]

MODEL_ROUTER_TABLE = [
    [
        "Planner",
        "ALL",
        config.model_light,
        "Simple structured task",
        "~800 tokens",
        GREEN,
    ],
    [
        "Query Rewriter",
        "ALL",
        config.model_light,
        "JSON structured output",
        "~1,200 tokens",
        GREEN,
    ],
    [
        "Reasoner (GREEN)",
        "GREEN/YELLOW",
        config.model_heavy,
        "Complex synthesis + citations",
        "~3,500 tokens",
        PRIMARY,
    ],
    [
        "Reasoner (RED)",
        "RED",
        config.model_light,
        "Budget-constrained synthesis",
        "~2,000 tokens",
        RED,
    ],
    [
        "Reflector",
        "ALL",
        config.model_light,
        "JSON hallucination check",
        "~1,000 tokens",
        GREEN,
    ],
    [
        "Responder (fallback)",
        "ERROR",
        config.model_heavy,
        "Graceful degradation",
        "~2,500 tokens",
        ACCENT,
    ],
]

CACHE_CONFIG_ROWS = [
    (
        "Similarity Threshold",
        f"{config.cache_similarity_threshold} cosine sim",
        PRIMARY,
    ),
    ("Max Cache Size", f"{config.cache_max_size} entries (LRU)", ACCENT),
    (
        "TTL",
        f"{config.cache_ttl_seconds:,} seconds ({config.cache_ttl_seconds // 3600}hr)",
        GREEN,
    ),
    ("Embedding Model", "all-MiniLM-L6-v2", PURPLE),
    ("Thread Safety", "RLock protected", GREEN),
    ("Eviction Policy", "Least-recently-used", TEXT_MUTED),
    ("Shared Model", "Yes — avoids reload across components", TEAL),
]

CONTEXT_WINDOW_DETAILS = [
    (
        "Top-K Chunks",
        str(config.context_top_k),
        "Maximum chunks injected per query",
        PRIMARY,
    ),
    (
        "Max Chunk Chars",
        str(config.context_max_chunk_chars),
        "Individual chunk character hard cap",
        ACCENT,
    ),
    (
        "Max Total Tokens",
        f"{config.context_max_total_tokens:,}",
        "Total context budget across all chunks",
        GREEN,
    ),
    (
        "Relevance Floor",
        str(config.context_relevance_floor),
        "Minimum cosine sim to include a chunk",
        PURPLE,
    ),
    (
        "Truncation Method",
        "Sentence boundary",
        "Splits at '.', '!', '?' before hard cut",
        TEAL,
    ),
    (
        "Token Estimate",
        f"{int(1 / config.tokens_per_char)} chars ≈ 1 token",
        "Fast approximation (configurable)",
        TEXT_MUTED,
    ),
]

FILING_TYPES = [
    {
        "icon": "📋",
        "type": "10-K (Annual)",
        "desc": "Full-year financials, risk factors, MD&A",
        "color": PRIMARY,
    },
    {
        "icon": "📊",
        "type": "10-Q (Quarterly)",
        "desc": "Quarterly earnings, balance sheet updates",
        "color": ACCENT,
    },
    {
        "icon": "⚡",
        "type": "8-K (Current)",
        "desc": "Material events, earnings surprises",
        "color": GREEN,
    },
    {
        "icon": "🎙️",
        "type": "Transcripts",
        "desc": "CEO/CFO earnings call verbatim text",
        "color": PURPLE,
    },
]

PIPELINE_NODES = [
    {
        "id": "plan",
        "label": "PLAN",
        "color": "#6C7FFF",
        "desc": "Decomposes query into research objectives",
        "model": "LLaMA 3.1 8B",
        "prompt": "PLANNER_PROMPT",
        "input": "Original query",
        "output": "1-3 step research plan",
        "note": "Skipped on YELLOW/RED tier",
    },
    {
        "id": "rewrite",
        "label": "REWRITE",
        "color": "#7B8FFF",
        "desc": "Generates atomic sub-queries for vector DB",
        "model": "LLaMA 3.1 8B",
        "prompt": "REWRITER_PROMPT",
        "input": "Plan + original query",
        "output": "JSON {queries:[...]}",
        "note": "Sequential dependency resolution",
    },
    {
        "id": "retrieve",
        "label": "RETRIEVE",
        "color": ACCENT,
        "desc": "Hybrid BM25 + Qdrant + RRF fusion",
        "model": "HybridRetriever",
        "prompt": "N/A",
        "input": "Sub-queries list",
        "output": "Candidate passage pool",
        "note": "fetch_k=50, RRF k=60",
    },
    {
        "id": "rerank",
        "label": "RERANK",
        "color": "#E89A20",
        "desc": "Cross-encoder ms-marco reranking",
        "model": "CrossEncoder",
        "prompt": "ms-marco-MiniLM-L-6-v2",
        "input": "Query-passage pairs",
        "output": "Reranked top-N passages",
        "note": "GPU-accelerated when available (falls back to CPU)",
    },
    {
        "id": "reason",
        "label": "REASON",
        "color": GREEN,
        "desc": "Citation-grounded synthesis via 70B LLM",
        "model": "LLaMA 3.3 70B",
        "prompt": "REASONER_PROMPT",
        "input": "Query + context chunks",
        "output": "Cited draft [Doc N]",
        "note": "Every claim must cite a doc",
    },
    {
        "id": "reflect",
        "label": "REFLECT",
        "color": "#26B87A",
        "desc": "Hallucination check + conditional loop-back",
        "model": "LLaMA 3.1 8B",
        "prompt": "REFLECTOR_PROMPT",
        "input": "Draft + context",
        "output": "JSON {is_grounded, needs_more_info}",
        "note": "Routes to: Responder | Rewriter | Reasoner",
    },
]

ROUTING_RULES = [
    {
        "from": "reflector",
        "condition": "is_grounded=True OR loops≥max",
        "to": "responder",
        "color": GREEN,
    },
    {
        "from": "reflector",
        "condition": "needs_more_info=True",
        "to": "query_rewriter",
        "color": ACCENT,
    },
    {
        "from": "reflector",
        "condition": "is_grounded=False",
        "to": "reasoner",
        "color": RED,
    },
    {
        "from": "responder",
        "condition": "error in state",
        "to": "graceful_degradation",
        "color": TEXT_MUTED,
    },
]
