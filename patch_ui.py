import sys

with open('src/ui/app.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Fix metric CSS size
old_css_metric = '''    /* Metrics */
    [data-testid="stMetricValue"] {
        font-family: var(--mono) !important;
        font-size: 1.5rem !important;
        color: var(--text) !important;
    }'''

new_css_metric = '''    /* Metrics */
    [data-testid="stMetricValue"] {
        font-family: var(--mono) !important;
        font-size: 1.15rem !important;
        color: var(--text) !important;
    }'''
text = text.replace(old_css_metric, new_css_metric)


# 2. Fix plot_gauge size
old_gauge_layout = '''fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=140, paper_bgcolor="rgba(0,0,0,0)", font={'color': "#EDEDED"})'''
new_gauge_layout = '''fig.update_layout(margin=dict(l=35, r=35, t=30, b=15), height=170, paper_bgcolor="rgba(0,0,0,0)", font={'color': "#EDEDED"})'''
text = text.replace(old_gauge_layout, new_gauge_layout)

# 3. Add welcome screen to operational console
old_chat_loop = '''    with chat_container:
        for message in st.session_state.messages:'''

new_chat_loop = '''    with chat_container:
        if not st.session_state.messages:
            st.markdown(
                "#### 📊 Welcome to FinSight-Alpha\\n"
                "<div style='color:var(--text-muted); font-size:0.9rem; margin-bottom:20px;'>"
                "This is a real-time FAANG-grade RAG intelligent platform executing a deterministic <code>Plan-Rewrite-Retrieve-Reason-Reflect</code> loop."
                "</div>\\n"
                "**System Architecture**:\\n"
                "• **Models:** LlaMA 3 / Gemini Pro under Token Budget Router\\n"
                "• **Index:** Hybrid BM25 & Qdrant with `ms-marco` Cross-Encoder Reranking\\n"
                "• **Evaluator:** Native explicit Ground-Truth tracking (MRR & NDCG)\\n\\n"
                "---\\n*Please enter a strategic financial inquiry below to begin.*", 
                unsafe_allow_html=True
            )
            
        for message in st.session_state.messages:'''

text = text.replace(old_chat_loop, new_chat_loop)

# 4. Enhance empty diagnostics state
old_empty_diag = '''<div style="color:var(--text-muted); font-size:0.8rem">Awaiting query execution...</div>'''
new_empty_diag = '''<div style="color:var(--text-muted); font-size:0.85rem">Awaiting query execution...<br><br><b>Up Next:</b><br>- Dynamic Latency Waterfall<br>- Graph execution tracing<br>- Bipartite Knowledge Maps<br>- Implicit Live MRR/NDCG metric computation</div>'''
text = text.replace(old_empty_diag, new_empty_diag)

with open('src/ui/app.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Patch applied")
