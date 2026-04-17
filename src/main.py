"""
FinSight-Alpha FastAPI Server — Token-Optimized.

Integrates semantic response cache at the API layer:
  - Near-identical queries skip the LLM pipeline entirely
  - Budget stats exposed via /health endpoint
  - Cache stats exposed via /cache/stats endpoint
"""

import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
sys.path.append(str(Path(__file__).resolve().parent.parent))
from langchain_core.messages import HumanMessage, AIMessage
from src.agents.langgraph_agent import build_graph, get_budget_manager, get_model_router
from src.optimization.response_cache import SemanticResponseCache
from src.optimization.token_budget import TokenBudgetManager
from src.retrieval.hybrid_retriever import HybridRetriever

app = FastAPI(
    title="FinSight-Alpha API",
    description="Institutional-Grade Financial Agentic RAG API",
    version="2.0.0"
)

# --- Initialize Components ---
print("Loading Agent Graph & Qdrant/BM25 Indices...")
agent_app = build_graph()
response_cache = SemanticResponseCache()
budget_manager = get_budget_manager()
model_router = get_model_router()

# Share the retriever's embedding model with the cache to avoid
# loading a second copy of all-MiniLM-L6-v2 (~80MB)
if HybridRetriever._shared_embedding_model is not None:
    response_cache.set_model(HybridRetriever._shared_embedding_model)
    print("  [Optimization] Shared embedding model with response cache")

print("System Ready (with token optimization layer)")


# --- Request/Response Models ---
class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    reasoning_trace: List[str]
    cache_hit: bool = False
    tokens_used: Optional[int] = None


# --- Endpoints ---
@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    """
    Accepts a user query, checks semantic cache first, then runs the
    LangGraph agent if no cache hit. Returns the answer with trace.
    """
    try:
        # Step 1: Check semantic response cache
        cached = response_cache.get(request.query)
        if cached is not None:
            response_text, trace = cached
            return QueryResponse(
                answer=response_text,
                reasoning_trace=trace,
                cache_hit=True,
                tokens_used=0,
            )

        # Step 2: Record pre-call token state
        tokens_before = budget_manager.total_tokens

        # Step 3: Run the full agent pipeline
        initial_state = {"messages": [HumanMessage(content=request.query)]}
        final_state = agent_app.invoke(initial_state)

        messages = final_state.get("messages", [])

        # Extract reasoning trace
        reasoning_trace = []
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    action = f"Used Tool: `{tc['name']}` | Args: {tc['args']}"
                    reasoning_trace.append(action)

        # Extract final answer
        final_answer = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and msg.content.strip():
                if '{"type": "function"' not in msg.content:
                    final_answer = msg.content
                    break

        if not final_answer:
            final_answer = "The agent could not formulate a definitive answer"

        # Step 4: Cache the result for future similar queries
        tokens_used = budget_manager.total_tokens - tokens_before
        response_cache.put(
            query=request.query,
            response=final_answer,
            trace=reasoning_trace,
            token_cost=tokens_used,
        )

        return QueryResponse(
            answer=final_answer,
            reasoning_trace=reasoning_trace,
            cache_hit=False,
            tokens_used=tokens_used,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check with model routing and budget info."""
    return {
        "status": "healthy",
        "model_info": model_router.info(),
        "budget": budget_manager.stats(),
    }


@app.get("/cache/stats")
async def cache_stats():
    """Return semantic cache performance statistics."""
    return response_cache.stats()


@app.post("/cache/clear")
async def cache_clear():
    """Flush the semantic response cache."""
    response_cache.clear()
    return {"status": "cache cleared"}


@app.post("/budget/reset")
async def budget_reset():
    """Reset the token budget for a new session."""
    budget_manager.reset()
    return {"status": "budget reset", "budget": budget_manager.stats()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
