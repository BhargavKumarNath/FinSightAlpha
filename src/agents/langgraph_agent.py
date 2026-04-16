import os
import json
import operator
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
    return "\n".join(formatted_results)

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
        return str(result)
    except Exception as e:
        return f"Calculation Error: {e}"

tools = [search_financial_docs, financial_calculator]

# Node Definitions
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
llm_with_tools = llm.bind_tools(tools)

def planner_node(state: AgentState):
    """
    Acts as the `Supervisor`. Reads the user's question and creates a research plan.    
    """
    print("\n[Node] Supervisor Planning...")
    original_query = state["messages"][0].content

    system_prompt = SystemMessage(content=
                                  "You are a Senior Financial Analyst. Your job is to create a step-by-step research plan "
                                  "to answer the user's query. Output only the numbered steps. Do not answer the question.")
    
    response = llm.invoke([system_prompt, HumanMessage(content=original_query)])
    print(f"\n--- Research Plan ---\n{response.content}\n---------------------")

    return {"plan": response.content}

def agent_node(state: AgentState):
    """
    The main reasoning engine. Executes steps of the plan, calls tools, and synthesizes answers.
    """
    print("\n[Node] Agent Reasoning...")

    system_prompt = SystemMessage(content=
                                  "You are an expert Institutional Financial Assistant. "
                                    "Use the provided tools to execute the research plan and answer the user's query.\n"
                                    f"Research Plan:\n{state.get('plan', 'No plan provided.')}\n\n"
                                    "Rules:\n"
                                    "1. ALWAYS cite your sources (e.g., 'According to the 10-K...').\n"
                                    "2. If performing math, use the financial_calculator tool.\n"
                                    "3. If the retrieved documents do not contain the answer, say so. Do not hallucinate."
                                  )
    
    # Prepend the system prompt dynamically to the message history
    messages = [system_prompt] + state["messages"]
    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}

# Graph Routing & Compilation
def route_after_agent(state: AgentState):
    """
    Determines whether to loop back to a tool or finish
    """
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"
    # Otherwise, it has finished reasoning
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
        "plan_created": False
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