"""
FinSight-Alpha LangGraph Agent — Production-Grade RAG Architecture.

Advanced Plan-Rewrite-Retrieve-Reason-Reflect Loop:
- Explicit Multi-Hop Query Decomposition
- Hybrid Search + Cross-Encoder Reranking
- Dynamic Context Windowing with Token Budget Awareness
- Citation Tracking
- Post-Generation Hallucination Self-Correction
"""

import os
import json
import operator
import time
import traceback
from typing import Annotated, TypedDict, Sequence, List, Dict, Any
from dotenv import load_dotenv

from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage
)
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# Import retriever
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.retrieval.hybrid_retriever import HybridRetriever

# Import optimization layer
from src.optimization.config import config
from src.optimization.token_budget import TokenBudgetManager, BudgetTier
from src.optimization.model_router import ModelRouter
from src.optimization.context_window import DynamicContextWindow

# Environment Setup
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is missing. Please add it to your .env file.")
if os.getenv("HF_TOKEN"):
    os.environ["HUGGINGFACE_HUB_TOKEN"] = os.getenv("HF_TOKEN")

# Reducer
def add_latencies(left: dict, right: dict) -> dict:
    if left is None: left = {}
    if right is None: return left
    res = left.copy()
    for k, v in right.items():
        res[k] = res.get(k, 0.0) + v
    return res

# Advanced State Definition
class AgentState(TypedDict):
    """State for production-grade RAG pipeline."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    original_query: str
    plan: str
    sub_queries: List[str]
    context_chunks: Annotated[List[Dict[str, Any]], operator.add]
    draft_answer: str
    reflection: str
    is_grounded: bool
    loop_count: int
    latencies: Annotated[Dict[str, float], add_latencies]
    error: str


# Optimization Layer Initialization
print("Initializing Production Retriever & Reranker...")
retriever = HybridRetriever()

budget_manager = TokenBudgetManager()
model_router = ModelRouter(budget_manager=budget_manager)

# Dynamic Context Window — shares the retriever's embedding model
context_window = DynamicContextWindow(
    embedding_model=HybridRetriever._shared_embedding_model,
    top_k=config.context_top_k,
    max_chunk_chars=config.context_max_chunk_chars,
    max_total_tokens=config.context_max_total_tokens,
    relevance_floor=config.context_relevance_floor,
)


# System Prompts

PLANNER_PROMPT = (
    "You are a Senior Financial Strategist. Analyze the user's query and formulate a 1-3 step "
    "retrieval plan to answer it completely. Output ONLY the numbered list. No filler."
)

REWRITER_PROMPT = (
    "You are a skilled Query Decomposition Agent. The user wants to answer a complex, multi-hop question. "
    "Your job is to break the main query into separate atomic search queries optimized for a semantic vector database. "
    "CRITICAL RULES:\n"
    "1. SEQUENTIAL DEPENDENCIES: If identifying an entity is required before answering a subsequent question, branch the questions sequentially! (e.g. ['Company that acquired Figma in 2022', 'CEO of Adobe', 'Open-source framework created by Adobe']).\n"
    "2. Generate NATURAL LANGUAGE queries only. NEVER generate SQL, code, or boolean expressions. Expand abbreviations automatically.\n"
    "3. Focus on entities, years, financial metrics, and specific risk factors.\n"
    "Previous plan: {plan}\n"
    "Feedback from reflection (if any): {reflection}\n\n"
    "Return JSON exact schema: {{\"queries\": [\"query 1\", \"query 2\"]}}"
)

REASONER_PROMPT = (
    "You are an elite Financial Analyst. Answer the user's query using ONLY the provided retrieved chunks. "
    "CRITICAL RULES:\n"
    "1. You MUST cite the exact [Doc X: source_file] for every claim you make.\n"
    "2. Do not mix sources without citing both.\n"
    "3. If specific data (dollar amounts, percentages, dates) appears in the chunks, ALWAYS include it in your answer.\n"
    "4. If the documents genuinely do not contain the answer, explicitly state so — but first carefully re-read ALL chunks.\n"
    "5. Do not fabricate information.\n"
    "6. Provide specific numbers and details whenever they appear in the context.\n"
    "\nUser Query: {original_query}\n\n"
    "--- Retrieved Context ---\n{context}"
)

REFLECTOR_PROMPT = (
    "You are a strict Hallucination Checker & Evaluator. You are reviewing a draft answer against the provided context. "
    "1. Is the answer directly addressing the user's query based ONLY on the context?\n"
    "2. Are there any fabricated facts or numbers not present in the context?\n"
    "3. Does the draft have missing information that requires another search?\n"
    "\nContext:\n{context}\n\nDraft Answer:\n{draft}\n\n"
    "Output JSON strictly with keys: 'is_grounded' (boolean), 'needs_more_info' (boolean), 'feedback' (string)."
)

# Nodes

def planner_node(state: AgentState):
    """Supervisor formulates a high-level research strategy."""
    if state.get("error"): return {"latencies": {}}
    start_t = time.time()
    print("\n[Node] Planner...")
    try:
        messages = state.get("messages", [])
        query = state.get("original_query", "")
        if not query: query = messages[0].content
        loop_count = state.get("loop_count", 0)

        if budget_manager.should_skip_planner():
            return {"plan": "Synthesize directly.", "original_query": query, "loop_count": loop_count, "latencies": {"planner": time.time() - start_t}}

        planner_llm = model_router.get_planner_llm()
        response = planner_llm.invoke([SystemMessage(content=PLANNER_PROMPT), HumanMessage(content=query)])
        plan = response.content
        print(f"  Plan: {plan.replace(chr(10), ' | ')}")
        budget_manager.record_usage(budget_manager.estimate_tokens(PLANNER_PROMPT + query), budget_manager.estimate_tokens(plan), "planner")
        return {"plan": plan, "original_query": query, "loop_count": loop_count, "latencies": {"planner": time.time() - start_t}}
    except Exception as e:
        err = f"Planner Error: {str(e)}"
        print(f"  [CRASH] {err}")
        return {"error": err, "latencies": {"planner": time.time() - start_t}}


def query_rewriter_node(state: AgentState):
    """Decomposes the main query into atomic search queries."""
    if state.get("error"): return {"latencies": {}}
    start_t = time.time()
    print("\n[Node] Query Rewriter...")
    try:
        plan = state.get("plan", "")
        query = state.get("original_query", "")
        reflection = state.get("reflection", "None")

        prompt = REWRITER_PROMPT.format(plan=plan, reflection=reflection)
        llm = model_router.get_planner_llm()
        
        response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=f"Original Query: {query}")], response_format={"type": "json_object"})
        
        try:
            parsed = json.loads(response.content)
            queries = parsed.get("queries", [query])
        except Exception:
            print("  [WARN] Failed to parse JSON queries. Using original.")
            queries = [query]

        print(f"  Generated Sub-Queries: {queries}")
        budget_manager.record_usage(budget_manager.estimate_tokens(prompt + query), budget_manager.estimate_tokens(response.content), "query_rewriter")
        return {"sub_queries": queries, "latencies": {"query_rewriter": time.time() - start_t}}
    except Exception as e:
        err = f"Rewriter Error: {str(e)}"
        print(f"  [CRASH] {err}")
        return {"error": err, "latencies": {"query_rewriter": time.time() - start_t}}


def retriever_node(state: AgentState):
    """Executes multi-query retrieval and formats chunks with citations."""
    if state.get("error"): return {"latencies": {}}
    start_t = time.time()
    print("\n[Node] Retriever...")
    try:
        sub_queries = state.get("sub_queries", [])
        if not sub_queries: sub_queries = [state.get("original_query", "")]
        top_n = budget_manager.get_retrieval_top_n()
        existing_context = state.get("context_chunks", [])
        new_chunks = []
        doc_id_counter = len(existing_context) + 1
        existing_texts = {c["text"] for c in existing_context}

        for sq in sub_queries:
            print(f"  Searching: `{sq}`")
            results = retriever.search(sq, top_n=top_n)
            for r in results:
                text = r["text"]
                if text not in existing_texts:
                    existing_texts.add(text)
                    source = r.get("metadata", {}).get("source", "unknown")
                    short_fname = source.split("/")[-1].split("\\")[-1]
                    new_chunks.append({"doc_id": doc_id_counter, "text": text, "source": short_fname, "score": r.get("score", 0.0)})
                    doc_id_counter += 1

        print(f"  Collected {len(new_chunks)} new chunks for reasoning.")
        new_chunks.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return {"context_chunks": new_chunks, "latencies": {"retriever": time.time() - start_t}}
    except Exception as e:
        err = f"Retriever Error: {str(e)}"
        print(f"  [CRASH] {err}")
        return {"error": err, "latencies": {"retriever": time.time() - start_t}}


def _format_context(context_chunks: List[Dict[str, Any]]) -> str:
    """
    Format context chunks into a clean, readable string for the LLM.
    Uses actual newlines (not escaped) so the model can parse document boundaries.
    """
    if not context_chunks:
        return "No relevant documents found."

    parts = []
    for c in context_chunks:
        header = f"[Doc {c['doc_id']}: {c['source']}] (Relevance: {c['score']:.2f})"
        parts.append(f"{header}\n{c['text']}")

    return "\n\n---\n\n".join(parts)


def reasoner_node(state: AgentState):
    """Synthesizes the final answer using strict citations."""
    if state.get("error"): return {"latencies": {}}
    start_t = time.time()
    print("\n[Node] Reasoner...")
    try:
        query = state.get("original_query", "")
        context_chunks = state.get("context_chunks", [])

        # Format context with proper newlines and clear boundaries
        context_str = _format_context(context_chunks)

        prompt = REASONER_PROMPT.format(original_query=query, context=context_str)
        llm = model_router.get_agent_llm()
        response = llm.invoke([SystemMessage(content=prompt)])
        draft = response.content
        preview = draft.replace('\n', ' ')[:100]
        print(f"  Draft: {preview}...")
        budget_manager.record_usage(budget_manager.estimate_tokens(prompt), budget_manager.estimate_tokens(draft), "reasoner")
        return {"draft_answer": draft, "latencies": {"reasoner": time.time() - start_t}}
    except Exception as e:
        err = f"Reasoner Error: {str(e)}"
        print(f"  [CRASH] {err}")
        return {"error": err, "latencies": {"reasoner": time.time() - start_t}}


def reflector_node(state: AgentState):
    """Evaluates the draft for hallucinations and coverage."""
    if state.get("error"): return {"loop_count": state.get("loop_count", 0), "latencies": {}}
    start_t = time.time()
    print("\n[Node] Reflector...")
    try:
        draft = state.get("draft_answer", "")
        context_chunks = state.get("context_chunks", [])
        loop_count = state.get("loop_count", 0) + 1

        # Format context with proper newlines
        context_str = "\n\n".join([f"[Doc {c['doc_id']}]: {c['text'][:400]}..." for c in context_chunks])
        prompt = REFLECTOR_PROMPT.format(context=context_str, draft=draft)
        
        llm = model_router.get_planner_llm()
        response = llm.invoke([SystemMessage(content=prompt)], response_format={"type": "json_object"})
        
        try:
            parsed = json.loads(response.content)
            is_grounded = parsed.get("is_grounded", True)
            needs_more_info = parsed.get("needs_more_info", False)
            feedback = parsed.get("feedback", "")
        except Exception:
            print("  [WARN] Failed to parse Reflection. Assuming grounded.")
            is_grounded = True
            needs_more_info = False
            feedback = "Parse failed."

        print(f"  Grounded: {is_grounded} | Needs More Info: {needs_more_info}")
        budget_manager.record_usage(budget_manager.estimate_tokens(prompt), budget_manager.estimate_tokens(response.content), "reflector")

        action = "needs_retrieval" if needs_more_info else ("needs_rewrite" if not is_grounded else "good")
        return {
            "is_grounded": True if action == "good" else False,
            "reflection": f"Action={action}, Reason={feedback}",
            "loop_count": loop_count,
            "latencies": {"reflector": time.time() - start_t}
        }
    except Exception as e:
        err = f"Reflector Error: {str(e)}"
        print(f"  [CRASH] {err}")
        return {"error": err, "loop_count": state.get("loop_count", 0), "latencies": {"reflector": time.time() - start_t}}


def responder_node(state: AgentState):
    """Packages the verified draft into the final AIMessage or triggers Graceful Degradation."""
    start_t = time.time()
    print("\n[Node] Responder (Finalizing)...")
    
    if state.get("error"):
        print(f"  [ROUTER] Bypassing to Direct Fallback mode due to: {state['error']}")
        query = state.get("original_query", "")
        try:
            llm = model_router.get_agent_llm()
            msg = "You are a helpful Financial assistant. Provide the best possible direct answer to the user's query globally, even without documentation."
            response = llm.invoke([SystemMessage(content=msg), HumanMessage(content=query)])
            draft = response.content + f"\n\n*(System Note: Graceful Degradation active. The Agentic framework bypassed execution due to an internal error: {state['error']}.)*"
            budget_manager.record_usage(budget_manager.estimate_tokens(msg+query), budget_manager.estimate_tokens(draft), "responder_fallback")
        except Exception as e:
            draft = f"CRITICAL FAILURE: Pipeline crashed and fallback LLM also failed to respond. Details: {str(e)}"
        return {"messages": [AIMessage(content=draft)], "latencies": {"responder_fallback": time.time() - start_t}}
    
    draft = state.get("draft_answer", "No answer could be formulated.")
    return {"messages": [AIMessage(content=draft)], "latencies": {"responder": time.time() - start_t}}


# Callbacks for Conditional Edges

def route_reflection(state: AgentState):
    """Determines whether to loop back based on reflection feedback and budget."""
    if state.get("error"):
        print("  [Router] Error detected. Forwarding to Responder.")
        return "responder"

    is_grounded = state.get("is_grounded", False)
    reflection = state.get("reflection", "")
    loop_count = state.get("loop_count", 0)
    
    max_loops = budget_manager.get_max_iterations() // 2  # e.g. 6//2 = 3 full Reflect loops
    
    if is_grounded or loop_count >= max_loops:
        if loop_count >= max_loops and not is_grounded:
            print(f"  [Router] Max multi-hop loops ({max_loops}) reached. Forcing response.")
        else:
            print("  [Router] Draft accepted.")
        return "responder"
    
    if "needs_retrieval" in reflection:
        print("  [Router] Reflector requested more context -> Query Rewriter")
        return "query_rewriter"
    
    print("  [Router] Reflector flagged hallucination -> Reasoner")
    return "reasoner"


# Graph Construction

def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("query_rewriter", query_rewriter_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("reasoner", reasoner_node)
    workflow.add_node("reflector", reflector_node)
    workflow.add_node("responder", responder_node)

    workflow.set_entry_point("planner")
    
    workflow.add_edge("planner", "query_rewriter")
    workflow.add_edge("query_rewriter", "retriever")
    workflow.add_edge("retriever", "reasoner")
    workflow.add_edge("reasoner", "reflector")
    
    workflow.add_conditional_edges(
        "reflector",
        route_reflection,
        {
            "responder": "responder",
            "query_rewriter": "query_rewriter",
            "reasoner": "reasoner"
        }
    )
    
    workflow.add_edge("responder", END)

    return workflow.compile()


def get_budget_manager() -> TokenBudgetManager:
    return budget_manager


def get_model_router() -> ModelRouter:
    return model_router


# CLI Execution
if __name__ == "__main__":
    import asyncio
    
    agent_app = build_graph()

    print("\n" + "="*50)
    print("Production RAG Pipeline CLI")
    print("="*50)

    test_query = (
        "Identify the current CEO of the company that acquired Figma in 2022, and then list the primary programming language used in the open-source web framework that this CEO's company originally created."
    )

    initial_state = {
        "messages": [HumanMessage(content=test_query)],
        "original_query": test_query,
        "loop_count": 0,
        "context_chunks": [],
        "latencies": {},
        "error": ""
    }

    print(f"\nEvaluating: {test_query}\n")

    try:
        final_state = agent_app.invoke(initial_state)
        
        print("\n" + "="*50)
        print("FINAL ANSWER with Citations:")
        print("="*50)
        
        messages = final_state.get("messages", [])
        if messages and isinstance(messages[-1], AIMessage):
            print(messages[-1].content)
        
        print("\n--- Telemetry ---")
        stats = budget_manager.stats()
        print(f"Total tokens used: {stats['total_tokens']}")
        print(f"Multi-hop loops: {final_state.get('loop_count')}")
        print(f"Latencies: {final_state.get('latencies')}")
        if final_state.get("error"):
            print(f"Encountered Error: {final_state['error']}")
        
    finally:
        retriever.close()