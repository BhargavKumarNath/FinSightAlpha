"""Run the full LangGraph agent on one question and capture everything that matters.

Index location is chosen by env vars read when the agent module is imported,
so they are set here before the import.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("FINSIGHT_COLLECTION", "filings5")
os.environ.setdefault("FINSIGHT_QDRANT_PATH", "data/qdrant_db_p3")
os.environ.setdefault("FINSIGHT_BM25_DIR", "data/p3")

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from src.agents import langgraph_agent as ag  # noqa: E402
from src.benchmark.llm import DailyLimitError  # noqa: E402
from src.optimization.config import config  # noqa: E402
from src.optimization.token_budget import BudgetTier  # noqa: E402

_GRAPH = None


def run(question: str, tier: str = "green", context_window: bool = False) -> dict:
    """Run once. `tier` pins the budget tier; `context_window` toggles DynamicContextWindow."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = ag.build_graph()
    bm = ag.get_budget_manager()
    bm.reset()
    bm.forced_tier = BudgetTier(tier)
    config.use_context_window = context_window
    state = {"messages": [HumanMessage(content=question)], "original_query": question,
             "context_chunks": [], "loop_count": 0, "latencies": {}, "error": ""}
    t0 = time.time()
    err = ""
    final: dict = {}
    try:
        final = _GRAPH.invoke(state)
    except Exception as e:  # infrastructure failure: recorded, never dropped
        err = f"{type(e).__name__}: {str(e)[:300]}"
    dt = time.time() - t0
    import re
    if re.search(r"per day|TPD|RPD", err + str(final.get('error', ''))):
        raise DailyLimitError(err or str(final.get('error')))
    answer = ""
    for m in reversed(final.get("messages", [])):
        if isinstance(m, AIMessage) and m.content and m.content.strip():
            answer = m.content
            break
    calls = bm.all_calls()
    return {
        "answer": answer,
        "error": err or final.get("error", ""),
        "chunks": [{"text": c["text"], "label": c["source"], "score": c["score"]}
                   for c in final.get("context_chunks", [])],
        "latency_s": dt,
        "loops": final.get("loop_count", 0),
        "node_latencies": final.get("latencies", {}),
        "input_tokens": sum(c["input"] for c in calls),
        "output_tokens": sum(c["output"] for c in calls),
        "n_calls": len(calls),
        "calls": [{"label": c["label"], "input": c["input"], "output": c["output"]} for c in calls],
        "tier": tier,
        "context_window": context_window,
    }
