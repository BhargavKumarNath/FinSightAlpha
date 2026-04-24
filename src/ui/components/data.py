"""
FinSight-Alpha — Static mock / demo data for all dashboard pages.

In production these would be fetched from the FastAPI backend.
Centralised here so every page imports from one place.
"""

from components.theme import PRIMARY, ACCENT, GREEN, RED, PURPLE, PINK, TEAL, TEXT_MUTED

# Performance / RAGAS
RAGAS_TREND = [
    {"run": "Run 1", "faithfulness": 0.71, "relevancy": 0.78, "mrr": 0.62, "ndcg": 0.58},
    {"run": "Run 2", "faithfulness": 0.79, "relevancy": 0.82, "mrr": 0.71, "ndcg": 0.67},
    {"run": "Run 3", "faithfulness": 0.83, "relevancy": 0.85, "mrr": 0.79, "ndcg": 0.74},
    {"run": "Run 4", "faithfulness": 0.87, "relevancy": 0.88, "mrr": 0.84, "ndcg": 0.80},
    {"run": "Run 5", "faithfulness": 0.91, "relevancy": 0.90, "mrr": 0.88, "ndcg": 0.85},
]

RETRIEVAL_COMPARISON = [
    {"method": "BM25 Only",   "precision": 0.54, "recall": 0.61, "f1": 0.57, "latency": 0.08},
    {"method": "Dense Only",  "precision": 0.68, "recall": 0.72, "f1": 0.70, "latency": 0.31},
    {"method": "Hybrid RRF",  "precision": 0.79, "recall": 0.83, "f1": 0.81, "latency": 0.38},
    {"method": "+ Rerank",    "precision": 0.91, "recall": 0.87, "f1": 0.89, "latency": 0.61},
]

LATENCY_BREAKDOWN = [
    {"phase": "Planner",  "ms": 420,  "color": "#6C7FFF"},
    {"phase": "Rewriter", "ms": 680,  "color": "#7B8FFF"},
    {"phase": "Retriever","ms": 1840, "color": ACCENT},
    {"phase": "Reranker", "ms": 720,  "color": "#E89A20"},
    {"phase": "Reasoner", "ms": 2100, "color": GREEN},
    {"phase": "Reflector","ms": 610,  "color": "#26B87A"},
    {"phase": "Responder","ms": 180,  "color": TEAL},
]

QUALITY_RADAR = [
    {"metric": "Faithfulness",  "value": 91},
    {"metric": "Relevancy",     "value": 90},
    {"metric": "Groundedness",  "value": 87},
    {"metric": "Coherence",     "value": 93},
    {"metric": "Completeness",  "value": 84},
    {"metric": "Precision",     "value": 91},
]

# Cache / optimisation
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

BUDGET_TIERS = [
    {
        "tier": "GREEN",  "range": "0 – 60k",   "color": GREEN,
        "loops": 6, "top_n": 5, "model": "LLaMA 3.3 70B",
        "desc": "Full pipeline. All nodes active.",
        "skips": [],
    },
    {
        "tier": "YELLOW", "range": "60k – 85k", "color": ACCENT,
        "loops": 3, "top_n": 3, "model": "LLaMA 3.3 70B",
        "desc": "Planner skipped. Reduced loops and retrieval.",
        "skips": ["Planner"],
    },
    {
        "tier": "RED",    "range": "85k+",       "color": RED,
        "loops": 2, "top_n": 2, "model": "LLaMA 3.1 8B",
        "desc": "Minimum viable path. Budget preservation mode.",
        "skips": ["Planner", "Heavy Model"],
    },
]

MODEL_ROUTER_TABLE = [
    ["Planner",           "ALL",          "LLaMA 3.1 8B",    "Simple structured task",          "~800 tokens",   GREEN],
    ["Query Rewriter",    "ALL",          "LLaMA 3.1 8B",    "JSON structured output",           "~1,200 tokens", GREEN],
    ["Reasoner (GREEN)",  "GREEN/YELLOW", "LLaMA 3.3 70B",   "Complex synthesis + citations",   "~3,500 tokens", PRIMARY],
    ["Reasoner (RED)",    "RED",          "LLaMA 3.1 8B",    "Budget-constrained synthesis",    "~2,000 tokens", RED],
    ["Reflector",         "ALL",          "LLaMA 3.1 8B",    "JSON hallucination check",        "~1,000 tokens", GREEN],
    ["Responder (fallback)", "ERROR",     "LLaMA 3.3 70B",   "Graceful degradation",            "~2,500 tokens", ACCENT],
]

# Evaluation results
EVAL_RESULTS = [
    {"q": "NVIDIA supply chain risk mitigation strategies",           "faith": 0.91, "relev": 0.89, "ctx": 8,  "cited": 6},
    {"q": "If NVIDIA data center revenue drops 10% from $47B",       "faith": 0.96, "relev": 0.94, "ctx": 5,  "cited": 3},
    {"q": "NVIDIA secret plans to acquire AMD (hallucination test)",  "faith": 0.88, "relev": 0.71, "ctx": 7,  "cited": 2},
    {"q": "Year-over-year GPU revenue comparison FY23 vs FY24",      "faith": 0.87, "relev": 0.90, "ctx": 9,  "cited": 7},
    {"q": "Key risk factors in NVIDIA's most recent 10-K",           "faith": 0.93, "relev": 0.92, "ctx": 11, "cited": 8},
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
    ("Unstructured",         "Doc Parsing",          TEXT_MUTED),
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
     "note": "GPU-accelerated on RTX 4070"},
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

OPTIMIZATION_COMPONENTS = [
    {"name": "SemanticResponseCache",   "color": PRIMARY,
     "role": "Cosine-sim query cache",
     "detail": "Threshold: 0.92 · LRU eviction · 1hr TTL · thread-safe (RLock)"},
    {"name": "TokenBudgetManager",      "color": ACCENT,
     "role": "3-tier graceful degradation",
     "detail": "GREEN/YELLOW/RED · per-call logging · reset() for new sessions"},
    {"name": "ModelRouter",             "color": GREEN,
     "role": "Task-based LLM dispatcher",
     "detail": "8B for planning · 70B for reasoning · RED forces 8B everywhere"},
    {"name": "DynamicContextWindow",    "color": PURPLE,
     "role": "Top-K relevance filtering",
     "detail": "Cosine floor: 0.25 · max chunk: 800 chars · cap: 2000 tokens"},
    {"name": "QueryBatcher",            "color": PINK,
     "role": "Batched embedding computation",
     "detail": "Single encode() call for eval suite · Jaccard dedup built-in"},
    {"name": "_EmbeddingCache (LRU)",   "color": TEAL,
     "role": "Query embedding memoization",
     "detail": "OrderedDict · 256 entry cap · O(1) lookup and eviction"},
]