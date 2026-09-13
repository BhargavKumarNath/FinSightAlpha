"""
FinSight-Alpha — Dashboard data: real artifacts where they exist, clearly
labeled illustrative placeholders where no real artifact exists yet.

Real sources:
  - EVAL_RESULTS / RAGAS_SUMMARY  <- data/reports/ragas_evaluation_report.csv
  - BUDGET_TIERS / MODEL_ROUTER_TABLE / CACHE_CONFIG_ROWS /
    CONTEXT_WINDOW_DETAILS         <- src.optimization.config.OptimizationConfig
                                      + src.optimization.token_budget.TokenBudgetManager

LATENCY_BREAKDOWN and CACHE_TOKEN_SAVINGS have no stored historical run to
load (per-query latencies/cache savings are only ever computed live, see
src/agents/langgraph_agent.py's `latencies` state field and
src/optimization/response_cache.py) — kept as illustrative example shapes,
explicitly labeled as such by the pages that render them. Real live values
for a given query are shown in pages/6_Live_Console.py.

Everything else here (FILING_TYPES, CAPABILITIES, TECH_STACK, PIPELINE_NODES,
ROUTING_RULES, AGENT_STATE_FIELDS) is static architecture description, not
a measured metric — out of scope for real-data sourcing.
"""

import ast
import contextlib
import csv
import io
import re
import sys
from pathlib import Path

from components.theme import PRIMARY, ACCENT, GREEN, RED, PURPLE, PINK, TEAL, TEXT_MUTED

# Make the backend `src` package importable from the standalone UI. Only
# lightweight, dependency-free modules (src.optimization.*) are imported
# below — this does not pull in torch/sentence-transformers/etc., so the
# UI-only requirements.txt install still works.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.optimization.config import config
from src.optimization.token_budget import TokenBudgetManager

# Real: RAGAS evaluation results
_RAGAS_REPORT_PATH = _REPO_ROOT / "data" / "reports" / "ragas_evaluation_report.csv"

# Mirrors the citation regex src/main.py uses to compute retrieval_metrics
# ("total_cited"), so this dashboard counts citations the same way the
# live backend does.
_CITATION_RE = re.compile(r"\[Doc\s*(\d+):")


def _load_ragas_report():
    """Load real per-query RAGAS results, or None if no report exists yet.

    Never fabricated as a substitute — callers must handle None explicitly
    (e.g. render "no evaluation run yet" rather than a fake number).
    """
    if not _RAGAS_REPORT_PATH.exists():
        return None

    rows = []
    with open(_RAGAS_REPORT_PATH, newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            try:
                contexts = ast.literal_eval(raw.get("retrieved_contexts", "[]"))
            except (ValueError, SyntaxError):
                contexts = []
            cited = _CITATION_RE.findall(raw.get("response", ""))
            rows.append({
                "q": raw["question"],
                "faith": float(raw["faithfulness"]),
                "relev": float(raw["answer_relevancy"]),
                "ctx": len(contexts),
                "cited": len(set(cited)),
            })
    return rows


EVAL_RESULTS = _load_ragas_report()  # None if data/reports/ragas_evaluation_report.csv is missing

if EVAL_RESULTS:
    RAGAS_SUMMARY = {
        "num_queries": len(EVAL_RESULTS),
        "avg_faithfulness": sum(r["faith"] for r in EVAL_RESULTS) / len(EVAL_RESULTS),
        "avg_answer_relevancy": sum(r["relev"] for r in EVAL_RESULTS) / len(EVAL_RESULTS),
    }
else:
    RAGAS_SUMMARY = None

# Note: with only 3 real evaluated queries and 2 metrics (Faithfulness,
# Answer Relevancy), there is no real historical "trend across runs" or
# "6-dimension quality radar" to show (RAGAS_TREND / QUALITY_RADAR were
# previously fabricated placeholders with no backing artifact — removed).
# There is also no real ablation study comparing BM25-only / Dense-only /
# Hybrid / +Rerank configurations anywhere in this repo (RETRIEVAL_COMPARISON
# was likewise a fabricated benchmark — removed).

# Illustrative only — no stored historical run exists for these (latencies
# and cache savings are computed live per-query, not persisted to disk).
LATENCY_BREAKDOWN = [
    {"phase": "Planner",  "ms": 420,  "color": "#6C7FFF"},
    {"phase": "Rewriter", "ms": 680,  "color": "#7B8FFF"},
    {"phase": "Retriever","ms": 1840, "color": ACCENT},
    {"phase": "Reranker", "ms": 720,  "color": "#E89A20"},
    {"phase": "Reasoner", "ms": 2100, "color": GREEN},
    {"phase": "Reflector","ms": 610,  "color": "#26B87A"},
    {"phase": "Responder","ms": 180,  "color": TEAL},
]

CACHE_TOKEN_SAVINGS = [
    {"query": "Q1", "saved": 0},
    {"query": "Q2", "saved": 3200},
    {"query": "Q3", "saved": 5100},
    {"query": "Q4", "saved": 4800},
    {"query": "Q5", "saved": 6200},
    {"query": "Q6", "saved": 0},
    {"query": "Q7", "saved": 7100},
    {"query": "Q8", "saved": 5900},
]

# Real: derived from OptimizationConfig + TokenBudgetManager's actual tier

def _budget_tier_snapshot():
    """Exercise the real TokenBudgetManager at each tier boundary and read
    back its actual decisions, instead of hand-copying branch constants."""
    mgr = TokenBudgetManager()
    snapshot = {}
    with contextlib.redirect_stdout(io.StringIO()):  # suppress its per-call logging
        for tier_name, used_tokens in (
            ("GREEN", 0), ("YELLOW", mgr.yellow_threshold), ("RED", mgr.red_threshold),
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
        "tier": "GREEN", "range": f"0 – {_fmt_tokens(_yellow_threshold)}", "color": GREEN,
        "loops": _tier_snapshot["GREEN"]["loops"], "top_n": _tier_snapshot["GREEN"]["top_n"],
        "model": config.model_heavy,
        "desc": "Full pipeline. All nodes active.",
        "skips": [],
    },
    {
        "tier": "YELLOW", "range": f"{_fmt_tokens(_yellow_threshold)} – {_fmt_tokens(_red_threshold)}", "color": ACCENT,
        "loops": _tier_snapshot["YELLOW"]["loops"], "top_n": _tier_snapshot["YELLOW"]["top_n"],
        "model": config.model_heavy if _tier_snapshot["YELLOW"]["heavy_model"] else config.model_light,
        "desc": "Planner skipped. Reduced loops and retrieval.",
        "skips": ["Planner"] if _tier_snapshot["YELLOW"]["skip_planner"] else [],
    },
    {
        "tier": "RED", "range": f"{_fmt_tokens(_red_threshold)}+", "color": RED,
        "loops": _tier_snapshot["RED"]["loops"], "top_n": _tier_snapshot["RED"]["top_n"],
        "model": config.model_heavy if _tier_snapshot["RED"]["heavy_model"] else config.model_light,
        "desc": "Minimum viable path. Budget preservation mode.",
        "skips": (["Planner"] if _tier_snapshot["RED"]["skip_planner"] else [])
                 + ([] if _tier_snapshot["RED"]["heavy_model"] else ["Heavy Model"]),
    },
]

MODEL_ROUTER_TABLE = [
    ["Planner",           "ALL",          config.model_light,  "Simple structured task",        "~800 tokens",   GREEN],
    ["Query Rewriter",    "ALL",          config.model_light,  "JSON structured output",         "~1,200 tokens", GREEN],
    ["Reasoner (GREEN)",  "GREEN/YELLOW", config.model_heavy,  "Complex synthesis + citations",  "~3,500 tokens", PRIMARY],
    ["Reasoner (RED)",    "RED",          config.model_light,  "Budget-constrained synthesis",   "~2,000 tokens", RED],
    ["Reflector",         "ALL",          config.model_light,  "JSON hallucination check",       "~1,000 tokens", GREEN],
    ["Responder (fallback)", "ERROR",     config.model_heavy,  "Graceful degradation",           "~2,500 tokens", ACCENT],
]

# Real: mirrors 3_Optimisation.py's former hardcoded cache-config table
CACHE_CONFIG_ROWS = [
    ("Similarity Threshold", f"{config.cache_similarity_threshold} cosine sim",     PRIMARY),
    ("Max Cache Size",        f"{config.cache_max_size} entries (LRU)",             ACCENT),
    ("TTL",                   f"{config.cache_ttl_seconds:,} seconds ({config.cache_ttl_seconds // 3600}hr)", GREEN),
    ("Embedding Model",       "all-MiniLM-L6-v2",                                   PURPLE),
    ("Thread Safety",         "RLock protected",                                    GREEN),
    ("Eviction Policy",       "Least-recently-used",                                TEXT_MUTED),
    ("Shared Model",          "Yes — avoids reload across components",              TEAL),
]

# Real: mirrors 3_Optimisation.py's former hardcoded context-window table
CONTEXT_WINDOW_DETAILS = [
    ("Top-K Chunks",       str(config.context_top_k),                              "Maximum chunks injected per query",       PRIMARY),
    ("Max Chunk Chars",    str(config.context_max_chunk_chars),                    "Individual chunk character hard cap",     ACCENT),
    ("Max Total Tokens",   f"{config.context_max_total_tokens:,}",                 "Total context budget across all chunks",  GREEN),
    ("Relevance Floor",    str(config.context_relevance_floor),                    "Minimum cosine sim to include a chunk",    PURPLE),
    ("Truncation Method",  "Sentence boundary",                                    "Splits at '.', '!', '?' before hard cut", TEAL),
    ("Token Estimate",     f"{int(1 / config.tokens_per_char)} chars ≈ 1 token",   "Fast approximation (configurable)",       TEXT_MUTED),
]

# Overview — filing types & capabilities
FILING_TYPES = [
    {"icon": "📋", "type": "10-K (Annual)",     "desc": "Full-year financials, risk factors, MD&A",    "color": PRIMARY},
    {"icon": "📊", "type": "10-Q (Quarterly)",  "desc": "Quarterly earnings, balance sheet updates",   "color": ACCENT},
    {"icon": "⚡", "type": "8-K (Current)",     "desc": "Material events, earnings surprises",          "color": GREEN},
    {"icon": "🎙️", "type": "Transcripts",       "desc": "CEO/CFO earnings call verbatim text",         "color": PURPLE},
]

CAPABILITIES = [
    {"icon": "🧠", "title": "Multi-Hop Reasoning",   "color": PRIMARY,
     "desc": "Decomposes complex analyst questions into sequential research sub-plans with dependency resolution."},
    {"icon": "🔍", "title": "Hybrid Search",          "color": ACCENT,
     "desc": "BM25 sparse retrieval fused with Qdrant dense vectors via Reciprocal Rank Fusion (RRF)."},
    {"icon": "⚖️", "title": "Cross-Encoder Reranking","color": GREEN,
     "desc": "ms-marco-MiniLM-L-6-v2 reranker re-scores and reorders all candidate passages."},
    {"icon": "🔁", "title": "Self-Correction Loop",   "color": PURPLE,
     "desc": "Reflector node detects hallucinations and triggers targeted retrieval retries autonomously."},
    {"icon": "📎", "title": "Citation Tracking",      "color": PINK,
     "desc": "Every claim in the final answer is attributed to a specific [Doc N] source chunk."},
    {"icon": "💰", "title": "Token Budget Manager",   "color": TEAL,
     "desc": "3-tier system (GREEN/YELLOW/RED) dynamically downscales models under rate pressure."},
]

TECH_STACK = [
    ("LangGraph",            "Agent Orchestration",  PRIMARY),
    ("Qdrant",               "Vector Database",      ACCENT),
    ("BM25 (rank-bm25)",     "Sparse Retrieval",     GREEN),
    ("Groq API",             "LLM Inference",        PURPLE),
    ("LLaMA 3.3 70B",        "Reasoning Model",      PRIMARY),
    ("LLaMA 3.1 8B",         "Planning Model",       TEAL),
    ("CrossEncoder",         "Passage Reranker",     ACCENT),
    ("FastAPI",              "Backend Server",       PINK),
    ("Streamlit",            "UI Framework",         RED),
    ("RAGAS",                "Evaluation Suite",     GREEN),
    ("sentence-transformers","Embeddings",           PURPLE),
    ("ParserRegistry",       "Doc Parsing",          TEXT_MUTED),
]

# Architecture
PIPELINE_NODES = [
    {"id": "plan",     "label": "PLAN",     "color": "#6C7FFF",
     "desc": "Decomposes query into research objectives",
     "model": "LLaMA 3.1 8B", "prompt": "PLANNER_PROMPT",
     "input": "Original query", "output": "1-3 step research plan",
     "note": "Skipped on YELLOW/RED tier"},
    {"id": "rewrite",  "label": "REWRITE",  "color": "#7B8FFF",
     "desc": "Generates atomic sub-queries for vector DB",
     "model": "LLaMA 3.1 8B", "prompt": "REWRITER_PROMPT",
     "input": "Plan + original query", "output": "JSON {queries:[...]}",
     "note": "Sequential dependency resolution"},
    {"id": "retrieve", "label": "RETRIEVE", "color": ACCENT,
     "desc": "Hybrid BM25 + Qdrant + RRF fusion",
     "model": "HybridRetriever", "prompt": "N/A",
     "input": "Sub-queries list", "output": "Candidate passage pool",
     "note": "fetch_k=50, RRF k=60"},
    {"id": "rerank",   "label": "RERANK",   "color": "#E89A20",
     "desc": "Cross-encoder ms-marco reranking",
     "model": "CrossEncoder", "prompt": "ms-marco-MiniLM-L-6-v2",
     "input": "Query-passage pairs", "output": "Reranked top-N passages",
     "note": "GPU-accelerated when available (falls back to CPU)"},
    {"id": "reason",   "label": "REASON",   "color": GREEN,
     "desc": "Citation-grounded synthesis via 70B LLM",
     "model": "LLaMA 3.3 70B", "prompt": "REASONER_PROMPT",
     "input": "Query + context chunks", "output": "Cited draft [Doc N]",
     "note": "Every claim must cite a doc"},
    {"id": "reflect",  "label": "REFLECT",  "color": "#26B87A",
     "desc": "Hallucination check + conditional loop-back",
     "model": "LLaMA 3.1 8B", "prompt": "REFLECTOR_PROMPT",
     "input": "Draft + context", "output": "JSON {is_grounded, needs_more_info}",
     "note": "Routes to: Responder | Rewriter | Reasoner"},
]

ROUTING_RULES = [
    {"from": "reflector", "condition": "is_grounded=True OR loops≥max", "to": "responder",     "color": GREEN},
    {"from": "reflector", "condition": "needs_more_info=True",           "to": "query_rewriter","color": ACCENT},
    {"from": "reflector", "condition": "is_grounded=False",              "to": "reasoner",      "color": RED},
    {"from": "responder", "condition": "error in state",                 "to": "graceful_degradation", "color": TEXT_MUTED},
]

AGENT_STATE_FIELDS = [
    ("messages",        "Sequence[BaseMessage]", PRIMARY),
    ("original_query",  "str",                  "white"),
    ("plan",            "str",                  "white"),
    ("sub_queries",     "List[str]",             ACCENT),
    ("context_chunks",  "List[Dict]",            ACCENT),
    ("draft_answer",    "str",                  "white"),
    ("reflection",      "str",                  TEXT_MUTED),
    ("is_grounded",     "bool",                 GREEN),
    ("loop_count",      "int",                  "white"),
    ("latencies",       "Dict[str, float]",      PRIMARY),
    ("error",           "str",                  RED),
]

# Real: DynamicContextWindow / SemanticResponseCache / TokenBudgetManager
# numeric details sourced from config directly (previously hand-typed and
# had drifted: floor/chunk/cap were 0.25/800/2000 vs. the real 0.15/1500/6000).
OPTIMIZATION_COMPONENTS = [
    {"name": "SemanticResponseCache",   "color": PRIMARY,
     "role": "Cosine-sim query cache",
     "detail": f"Threshold: {config.cache_similarity_threshold} · LRU eviction · "
               f"{config.cache_ttl_seconds // 3600}hr TTL · thread-safe (RLock)"},
    {"name": "TokenBudgetManager",      "color": ACCENT,
     "role": "3-tier graceful degradation",
     "detail": "GREEN/YELLOW/RED · per-call logging · reset() for new sessions"},
    {"name": "ModelRouter",             "color": GREEN,
     "role": "Task-based LLM dispatcher",
     "detail": f"{config.model_light} for planning · {config.model_heavy} for reasoning · RED forces light model everywhere"},
    {"name": "DynamicContextWindow",    "color": PURPLE,
     "role": "Top-K relevance filtering",
     "detail": f"Cosine floor: {config.context_relevance_floor} · max chunk: {config.context_max_chunk_chars} chars · "
               f"cap: {config.context_max_total_tokens:,} tokens"},
    {"name": "QueryBatcher",            "color": PINK,
     "role": "Batched embedding computation",
     "detail": "Single encode() call for eval suite · Jaccard dedup built-in"},
    {"name": "_EmbeddingCache (LRU)",   "color": TEAL,
     "role": "Query embedding memoization",
     "detail": f"OrderedDict · {config.retriever_cache_maxsize} entry cap · O(1) lookup and eviction"},
]
