"""
FinSight-Alpha LangGraph Agent — Token-Optimized.

Integrates all optimization modules:
  - Compressed system prompts (zero-filler, instruction-dense)
  - Dynamic context windowing (top-K relevant chunks, truncated)
  - Token budget manager (graceful degradation under pressure)
  - Tiered model routing (8B planner, 70B reasoning)
  - Message history pruning (summarize old tool outputs)
"""

import os
import re
from typing import Annotated, TypedDict, Sequence
from dotenv import load_dotenv
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
)
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Import retriever
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.retrieval.hybrid_retriever import HybridRetriever

# Import optimization layer
from src.optimization.config import config
from src.optimization.context_window import DynamicContextWindow
from src.optimization.token_budget import TokenBudgetManager, BudgetTier
from src.optimization.model_router import ModelRouter

# Environment Setup
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is missing. Please add it to your .env file.")
if os.getenv("HF_TOKEN"):
    os.environ["HUGGINGFACE_HUB_TOKEN"] = os.getenv("HF_TOKEN")


# State Definition
class AgentState(TypedDict):
    """
    maintains the state of our agentic loop.
    `add_messages` automatically appends new messages to the existing list.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    plan: str


# Optimization Layer Initialization
print("Initializing Retriever for the Agent (This loads BM25 into memory)...")
retriever = HybridRetriever()

# Shared optimization instances (created once, used across all requests)
budget_manager = TokenBudgetManager()
model_router = ModelRouter(budget_manager=budget_manager)
context_window = DynamicContextWindow()

# Share the retriever's embedding model with context_window to avoid
# loading a second copy of all-MiniLM-L6-v2 (~80MB)
context_window.set_model(retriever.embedding_model)


# Compressed System Prompts
# These are instruction-dense with zero filler. Every token earns its place.

PLANNER_PROMPT = (
    "Output a 1-3 step research plan as a numbered list. "
    "Do NOT answer the question, provide data, or output JSON/code."
)

AGENT_PROMPT_TEMPLATE = (
    "Financial analyst. Tools: search_financial_docs (SEC filings), "
    "financial_calculator (math).\n"
    "Plan: {plan}\n"
    "Rules: Use tools to retrieve data. Cite sources in final answer. "
    "State clearly if data is unavailable. No fabrication."
)

AGENT_PROMPT_MINIMAL = (
    "Answer using retrieved SEC filing data. Cite sources. No fabrication."
)

# Stripped version for hallucination recovery — absolute minimum prompt
RECOVERY_PROMPT = "Search SEC filings to answer this question. Use search_financial_docs tool."


# Tool Definitions
@tool
def search_financial_docs(query: str) -> str:
    """
    Search the SEC filings database for financial information, risks, and strategies. Input should be a specific search query.
    """
    print(f"\n[Tool Execution] Searching for: `{query}`")

    # Budget-aware retrieval: fewer results under pressure
    top_n = budget_manager.get_retrieval_top_n()
    results = retriever.search(query, top_n=top_n)

    # Apply dynamic context windowing: score, filter, truncate
    selected = context_window.select_chunks(query, results)

    # Format compactly — short source names, relevance scores
    output = context_window.format_context(selected)

    print(f"[Tool Execution] search_financial_docs: {len(results)} retrieved → "
          f"{len(selected)} selected (windowed)")
    return output


@tool
def financial_calculator(expression: str) -> str:
    """
    Evaluates mathematical expressions for financial analysis
    Input must be a valid Python mathematical expression
    """
    print(f"\n[Tool Execution] Calculating: `{expression}`")
    try:
        allowed_names = {"__builtins__": None}
        result = eval(expression, allowed_names, {})
        print(f"[Tool Execution] financial_calculator result: {result}")
        return str(result)
    except Exception as e:
        print(f"[Tool Execution] financial_calculator ERROR: {e}")
        return f"Calculation Error: {e}"


tools = [search_financial_docs, financial_calculator]

# Regex pattern to detect hallucinated tool calls in text output
_HALLUCINATED_TOOL_CALL_PATTERN = re.compile(
    r'\{"type"\s*:\s*"function"|"name"\s*:\s*"(search_financial_docs|financial_calculator)"',
)


# --- Message History Pruning ---
def _prune_message_history(messages: list, max_tool_chars: int = 600) -> list:
    """
    Compress old tool outputs in message history to reduce context bloat.

    For ToolMessages beyond the most recent 2, truncate their content
    to max_tool_chars. This prevents unbounded context growth across
    multiple agent iterations while preserving the most recent results.
    """
    tool_indices = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]

    if len(tool_indices) <= 2:
        return messages  # Nothing to prune

    # Indices of older tool messages (all except last 2)
    old_tool_indices = set(tool_indices[:-2])
    pruned = []

    for i, msg in enumerate(messages):
        if i in old_tool_indices and isinstance(msg, ToolMessage):
            content = msg.content or ""
            if len(content) > max_tool_chars:
                truncated = content[:max_tool_chars] + "\n[...truncated for context budget...]"
                # Create new ToolMessage with truncated content
                pruned.append(ToolMessage(
                    content=truncated,
                    tool_call_id=msg.tool_call_id,
                    name=getattr(msg, 'name', None),
                ))
            else:
                pruned.append(msg)
        else:
            pruned.append(msg)

    return pruned


# --- Node Definitions ---

def planner_node(state: AgentState):
    """
    Supervisor node. Creates a brief research plan.
    Uses the light (8B) model — planning is a trivial task.
    Skipped entirely under YELLOW/RED budget tiers.
    """
    print("\n[Node] Supervisor Planning...")

    # Budget check: skip planner to save ~800 tokens under pressure
    if budget_manager.should_skip_planner():
        tier = budget_manager.get_tier()
        print(f"  [Budget] Skipping planner (tier={tier.value}) — using default plan")
        return {"plan": "Search SEC filings for relevant information, then synthesize answer."}

    messages = state.get("messages", [])
    if not messages:
        raise ValueError("No messages provided to the graph")

    original_query = messages[0].content
    system_prompt = SystemMessage(content=PLANNER_PROMPT)

    # Use light model for planning (8B — ~10× cheaper than 70B)
    planner_llm = model_router.get_planner_llm()
    response = planner_llm.invoke([system_prompt, HumanMessage(content=original_query)])

    plan_text = response.content
    print(f"\n--- Research Plan ---\n{plan_text}\n---------------------")

    # Record token usage
    input_est = budget_manager.estimate_tokens(PLANNER_PROMPT + original_query)
    output_est = budget_manager.estimate_tokens(plan_text)
    budget_manager.record_usage(input_est, output_est, call_label="planner")

    return {"plan": plan_text}


def agent_node(state: AgentState):
    """
    Main reasoning engine. Executes tools and synthesizes answers.
    Uses the heavy (70B) model for reasoning, with budget-aware degradation.
    """
    print("\n[Node] Agent Reasoning...")

    plan = state.get("plan", "Search SEC filings and answer the question.")
    messages = state["messages"]

    # Count iterations for safety
    ai_message_count = sum(1 for m in messages if isinstance(m, AIMessage))
    max_iterations = budget_manager.get_max_iterations()
    print(f"  [State] Agent iteration: {ai_message_count + 1}/{max_iterations}")

    has_tool_results = any(isinstance(m, ToolMessage) for m in messages)
    print(f"  [State] Has tool results: {has_tool_results}")

    # Select prompt based on budget tier
    tier = budget_manager.get_tier()
    if tier == BudgetTier.RED:
        prompt_text = AGENT_PROMPT_MINIMAL
    else:
        prompt_text = AGENT_PROMPT_TEMPLATE.format(plan=plan)

    system_prompt = SystemMessage(content=prompt_text)

    # Prune message history to reduce context bloat
    pruned_messages = _prune_message_history(list(messages))
    full_messages = [system_prompt] + pruned_messages

    # Get budget-appropriate model with tools
    llm_with_tools = model_router.get_agent_llm(tools=tools)
    llm_plain = model_router.get_agent_llm()  # Without tools for forced final answer

    # Max Iteration Guard
    if ai_message_count >= max_iterations:
        print(f"  [WARN] Max iterations ({max_iterations}) reached. Forcing final answer.")
        response = llm_plain.invoke(full_messages)

        # Record usage
        input_est = budget_manager.estimate_tokens(
            prompt_text + "".join(m.content or "" for m in pruned_messages)
        )
        output_est = budget_manager.estimate_tokens(response.content or "")
        budget_manager.record_usage(input_est, output_est, call_label="agent_final_forced")
        return {"messages": [response]}

    # Tool Call Strategy
    if not has_tool_results:
        print("  [Strategy] First pass - forcing tool call via tool_choice='required'")
        response = llm_with_tools.invoke(full_messages, tool_choice="required")
    else:
        print("  [Strategy] Follow-up pass — tool_choice='auto'")
        response = llm_with_tools.invoke(full_messages)

    # Hallucination Guard (single retry, not cascading 3×)
    if not response.tool_calls and not has_tool_results:
        if _HALLUCINATED_TOOL_CALL_PATTERN.search(response.content or ""):
            print("  [WARN] Hallucinated tool call detected. Single retry with simplified prompt.")
            simple_messages = [
                SystemMessage(content=RECOVERY_PROMPT),
                messages[0],
            ]
            response = llm_with_tools.invoke(simple_messages, tool_choice="required")

    # Record token usage
    msg_text = "".join(m.content or "" for m in pruned_messages)
    input_est = budget_manager.estimate_tokens(prompt_text + msg_text)
    output_est = budget_manager.estimate_tokens(response.content or "")
    label = f"agent_iter_{ai_message_count + 1}"
    budget_manager.record_usage(input_est, output_est, call_label=label)

    # Log Routing Decision
    if response.tool_calls:
        for tc in response.tool_calls:
            print(f"  [Tool Call Detected] {tc['name']}(args={tc['args']})")
    else:
        answer_preview = (response.content or "")[:120]
        print(f"  [Final Answer] {answer_preview}...")

    return {"messages": [response]}


# --- Graph Routing & Compilation ---
def route_after_agent(state: AgentState):
    """
    Determines whether to loop back to a tool or finish.
    """
    last_message = state["messages"][-1]

    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        tool_names = [tc['name'] for tc in last_message.tool_calls]
        print(f"[Router] -> tools (calls: {tool_names})")
        return "tools"

    print("[Router] -> END (no tool calls, agent is done)")
    return END


def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))

    # Define Edges (The flow)
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "agent")

    # Conditional routing from agent
    workflow.add_conditional_edges(
        "agent", route_after_agent, {"tools": "tools", END: END}
    )

    # Edge from tools back to agent
    workflow.add_edge("tools", "agent")

    return workflow.compile()


# --- Module-level accessors for integration ---
def get_budget_manager() -> TokenBudgetManager:
    """Get the shared budget manager instance."""
    return budget_manager


def get_model_router() -> ModelRouter:
    """Get the shared model router instance."""
    return model_router


if __name__ == "__main__":
    agent_app = build_graph()

    print("\n" + "="*50)
    print("Welcome to FinSight-Alpha Agent CLI")
    print("="*50)

    test_query = (
        "What are NVIDIA's main risks regarding the GPU supply chain, "
        "and if their revenue drops by 15% from a hypothetical $60 Billion, "
        "what would the new revenue be?"
    )

    initial_state = {
        "messages": [HumanMessage(content=test_query)],
        "plan": "",
    }

    final_message = None

    print("\n--- Running Agent ---\n")

    try:
        for event in agent_app.stream(initial_state):
            for node_name, node_state in event.items():
                print(f"\n[Event] Node: {node_name}")

                if isinstance(node_state, dict):
                    if "plan" in node_state:
                        print("\n--- Plan ---")
                        print(node_state["plan"])

                    if "messages" in node_state:
                        last_msg = node_state["messages"][-1]
                        if isinstance(last_msg, AIMessage):
                            final_message = last_msg
                            print("\n--- AI Response ---")
                            print(last_msg.content)

        print("\n" + "="*50)
        print("FINAL ANSWER:")
        print("="*50)

        if final_message:
            print(final_message.content)
        else:
            print("No final message generated.")

        # Print optimization stats
        print("\n--- Token Budget Stats ---")
        stats = budget_manager.stats()
        print(f"  Total tokens: {stats['total_tokens']}")
        print(f"  Budget used: {stats['usage_pct']:.1%}")
        print(f"  Tier: {stats['tier']}")
        print(f"  LLM calls: {stats['call_count']}")

    finally:
        retriever.close()