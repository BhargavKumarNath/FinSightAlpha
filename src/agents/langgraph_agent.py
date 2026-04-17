import os
import json
import operator
import re
from typing import Annotated, TypedDict, Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Import our custom Retriever
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.retrieval.hybrid_retriever import HybridRetriever

# Load Environment Variables (Groq API Key)
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is missing. Please add it to your .env file.")

# (Hugging Face API Key)
load_dotenv()
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

# Tool Definitions
print("Initializing Retriever for the Agent (This loads BM25 into memory)...")
retriever = HybridRetriever()


@tool
def search_financial_docs(query: str) -> str:
    """
    Search the SEC filings database for financial information, risks, and strategies. Input should be a specific search query.
    """
    print(f"\n[Tool Execution] Searching for: `{query}`")
    results = retriever.search(query, top_n=5)

    formatted_results = []
    for i, res in enumerate(results, 1):
        formatted_results.append(
            f"Result {i} (Source: {res['metadata'].get('source')}):\n{res['text']}\n"
        )

    output = "\n".join(formatted_results)
    print(f"[Tool Execution] search_financial_docs returned {len(results)} result(s)")
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

# LLM Setup
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# Constants
MAX_AGENT_ITERATIONS = 6

# Regex pattern to detect hallucinated tool calls in text output
_HALLUCINATED_TOOL_CALL_PATTERN = re.compile(
    r'\{"type"\s*:\s*"function"|"name"\s*:\s*"(search_financial_docs|financial_calculator)"',
)

# Node Definitions

def planner_node(state: AgentState):
    """
    Acts as the `Supervisor`. Reads the user's question and creates a brief
    research plan. Does NOT answer the question or call tools.
    """
    print("\n[Node] Supervisor Planning...")
    messages = state.get("messages", [])
    if not messages:
        raise ValueError("No messages provided to the graph")

    original_query = messages[0].content

    system_prompt = SystemMessage(content=(
        "You are a financial research planning assistant. "
        "Given a user query about SEC filings, output ONLY a brief action plan "
        "as a numbered list of 1-3 short steps.\n\n"
        "Rules:\n"
        "- Do NOT answer the question.\n"
        "- Do NOT provide any financial data, numbers, or analysis.\n"
        "- Do NOT output JSON, function calls, or code.\n"
        "- ONLY output a short numbered plan of what to research.\n\n"
        "Example:\n"
        "1. Search SEC filings for NVIDIA supply chain risk factors.\n"
        "2. Summarize the key mitigation strategies found.\n"
    ))

    response = llm.invoke([system_prompt, HumanMessage(content=original_query)])
    plan_text = response.content
    print(f"\n--- Research Plan ---\n{plan_text}\n---------------------")

    return {"plan": plan_text}

def agent_node(state: AgentState):
    """
    The main reasoning engine. Executes tools according to the plan,
    then synthesizes a final answer from the retrieved context.
    """
    print("\n[Node] Agent Reasoning...")

    plan = state.get("plan", "Answer the user's question using the available tools.")
    messages = state["messages"]

    # Count previous agent iterations for safety
    ai_message_count = sum(1 for m in messages if isinstance(m, AIMessage))
    print(f"  [State] Agent iteration: {ai_message_count + 1}/{MAX_AGENT_ITERATIONS}")

    # Detect if tools have already been called (ToolMessages exist in history)
    has_tool_results = any(isinstance(m, ToolMessage) for m in messages)
    print(f"  [State] Has tool results: {has_tool_results}")

    # Build system prompt — keep it clean and concise to avoid confusing tool calling
    system_prompt = SystemMessage(content=(
        "You are an expert financial analyst. You have access to two tools:\n"
        "- search_financial_docs: Search SEC filings for financial information\n"
        "- financial_calculator: Evaluate math expressions\n\n"
        f"Research plan: {plan}\n\n"
        "Instructions:\n"
        "- Use search_financial_docs to retrieve relevant SEC filing excerpts.\n"
        "- Use financial_calculator for any numerical computations.\n"
        "- After receiving tool results, write a final answer citing sources.\n"
        "- If documents do not contain the answer, state that clearly.\n"
        "- Do NOT fabricate information."
    ))

    full_messages = [system_prompt] + list(messages)

    # Max Iteration Guard
    if ai_message_count >= MAX_AGENT_ITERATIONS:
        print(f"  [WARN] Max iterations ({MAX_AGENT_ITERATIONS}) reached. Forcing final answer.")
        # Use LLM without tools to force a text-only synthesized answer
        response = llm.invoke(full_messages)
        return {"messages": [response]}

    # Tool Call Strategy
    if not has_tool_results:
        # First invocation: force the LLM to call a tool
        print("  [Strategy] First pass - forcing tool call via tool_choice='required'")
        response = llm_with_tools.invoke(full_messages, tool_choice="required")
    else:
        # Subsequent invocations: let the LLM decide (auto)
        print("  [Strategy] Follow-up pass — tool_choice='auto'")
        response = llm_with_tools.invoke(full_messages)

    # Hallucination Guard
    if not response.tool_calls and not has_tool_results:
        # Check if the LLM wrote tool calls as text instead of native API calls
        if _HALLUCINATED_TOOL_CALL_PATTERN.search(response.content or ""):
            print("  [WARN] Detected hallucinated tool call in text output. Retrying with forced tool_choice...")
            response = llm_with_tools.invoke(full_messages, tool_choice="required")

            # If still hallucinating after retry, strip the text and try once more
            if not response.tool_calls:
                print("  [WARN] Second retry — simplifying prompt...")
                simple_messages = [
                    SystemMessage(content="Search the SEC filings to answer this question. You MUST use the search_financial_docs tool."),
                    messages[0],  # Original user query
                ]
                response = llm_with_tools.invoke(simple_messages, tool_choice="required")

    # Log Routing Decision
    if response.tool_calls:
        for tc in response.tool_calls:
            print(f"  [Tool Call Detected] {tc['name']}(args={tc['args']})")
    else:
        answer_preview = (response.content or "")[:120]
        print(f"  [Final Answer] {answer_preview}...")

    return {"messages": [response]}


# Graph Routing & Compilation
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

    finally:
        retriever.close()