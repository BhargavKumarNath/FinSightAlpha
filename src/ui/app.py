import streamlit as st
import requests

API_URL = "http://localhost:8000/chat"

st.set_page_config(
    page_title="FinSight-Alpha | Institutional RAG",
    page_icon="📈",
    layout="centered"
)

st.title("📈 FinSight-Alpha")
st.markdown("**Autonomous Financial Research Agent** (Powered by Llama-3.3, LangGraph, & Hybrid RAG)")

# Initialise session state for char history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "trace" in message and message["trace"]:
            with st.expander("🔍 View Agent Reasoning Trace"):
                for step in message["trace"]:
                    st.code(step)

# User Input
if prompt := st.chat_input("Ask a complex financial question (e.g., YoY revenue impacts, supply chain risks)..."):
    # 1. Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 2. Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Agent is researching and reasoning..."):
            try:
                # Call the FastAPI backend (add timeout)
                response = requests.post(API_URL, json={"query": prompt}, timeout=30)

                if response.status_code == 200:
                    try:
                        data = response.json()
                    except ValueError:
                        raise ValueError("Invalid JSON response from API")

                    answer = data.get("answer", "No answer provided.")
                    trace = data.get("reasoning_trace")

                    # Display the final answer
                    st.markdown(answer)
                    
                    # Display the trace safely
                    if isinstance(trace, list):
                        with st.expander("🔍 View Agent Reasoning Trace"):
                            for step in trace:
                                st.code(str(step))
                    
                    # Save the assistant response to session state
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "trace": trace
                    })
                    
                else:
                    error_msg = f"API Error: {response.status_code} - {response.text}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
                    
            except requests.exceptions.ConnectionError:
                error_msg = "Could not connect to the API. Is the FastAPI server running on port 8000?"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            
            except requests.exceptions.Timeout:
                error_msg = "The request timed out. The backend might be overloaded."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
