import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
sys.path.append(str(Path(__file__).resolve().parent.parent))
from langchain_core.messages import HumanMessage, AIMessage
from src.agents.langgraph_agent import build_graph
app = FastAPI(
    title="FinSight-Alpha API",
    description="Institutional-Grade Financial Agentic RAG API",
    version="1.0.0"
)

print("Loading Agent Graph & Qdrant/BM25 Indices...")
agent_app = build_graph()
print("System Ready")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    reasoning_trace: List[str]

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    """
    Accepts a user query, runs it through the LandGraph agent, and returns the final answer along with the tool execution trace.
    """
    try:
        initial_state = {"messages": [HumanMessage(content=request.query)]}
        final_state = agent_app.invoke(initial_state)

        messages = final_state.get("messages", [])

        # 1. Extract the reasoning trace (tool calls)
        reasoning_trace = []
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    action = f"Used Tool: `{tc['name']}` | Args: {tc['args']}"
                    reasoning_trace.append(action)
        
        # 2. Extract the Final Answer
        final_answer = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and msg.content.strip():
                if '{"type": "function"' not in msg.content:
                    final_answer = msg.content
                    break
        
        if not final_answer:
            final_answer = "The agent could not formulate a definitive answer"
        
        return QueryResponse(
            answer=final_answer,
            reasoning_trace=reasoning_trace
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "llama-3.3-70b-versatile"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
