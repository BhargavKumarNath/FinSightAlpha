import sys

with open('src/ui/app.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. replace plot_waterfall_latency definition
old_plot_def = '''def plot_waterfall_latency(total_time):
    # Simulated strict phase breakdown since backend does not log per-step ms
    phases = ["Plan", "Rewrite", "Retrieve (Hyb+RRF)", "Rerank (Cross)", "Reason", "Reflect", "Finalize"]
    weights = [0.05, 0.1, 0.35, 0.15, 0.25, 0.08, 0.02]
    
    times = [total_time * w for w in weights]'''

new_plot_def = '''def plot_waterfall_latency(latencies_dict, total_time):
    if not latencies_dict:
        phases = ["Network / API Request"]
        times = [total_time]
    else:
        phases = list(latencies_dict.keys())
        times = list(latencies_dict.values())
        measured = sum(times)
        if total_time > measured:
            phases.append("Overhead")
            times.append(total_time - measured)'''
text = text.replace(old_plot_def, new_plot_def)


# 2. replace st.session_state.messages.append
old_append = '''                        st.session_state.messages.append({"role": "assistant", "content": ans, "trace": trace, "rtt": rtt, "query": prompt})'''
new_append = '''                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": ans,
                            "trace": trace,
                            "rtt": rtt,
                            "query": prompt,
                            "latencies": data.get("latencies", {}),
                            "retrieval_metrics": data.get("retrieval_metrics", {})
                        })'''
text = text.replace(old_append, new_append)


# 3. replace st.plotly_chart call
old_chart_call = '''st.plotly_chart(plot_waterfall_latency(latest["rtt"]), width='stretch', config={'displayModeBar': False})'''
new_chart_call = '''st.plotly_chart(plot_waterfall_latency(latest.get("latencies", {}), latest.get("rtt", 0.0)), width='stretch', config={'displayModeBar': False})'''
text = text.replace(old_chart_call, new_chart_call)


# 4. replace metrics display
old_metrics = '''# 4. Synthesized Retrieval Metrics
        num_sub = sum(1 for t in latest["trace"] if "Searched for:" in t)
        docs_cited = len(set(citations))
        # Simulated Hit Rate: Docs cited vs arbitrary retreived pool size
        simulated_mr = (docs_cited / max((num_sub * 3), 1)) if num_sub else (1.0 if latest.get("cache_hit") else 0.0)
        st.metric("Synthesized Retrieval Quality (NDCG ≈ Hits/Sub)", f"{simulated_mr:.2f}", 
                  help="Approximation derived from cited docs vs generated subqueries.")'''

new_metrics = '''# 4. Live Native Retrieval Metrics (MRR / NDCG)
        metrics = latest.get("retrieval_metrics", {})
        mrr = metrics.get("MRR", 0.0)
        ndcg = metrics.get("NDCG", 0.0)
        colM1, colM2 = st.columns(2)
        colM1.metric("Live MRR", f"{mrr:.2f}", help="Mean Reciprocal Rank calculated by matching LLM Citations against Cross-Encoder sorted candidates.")
        colM2.metric("Live NDCG", f"{ndcg:.2f}", help="Normalized Discounted Cumulative Gain against theoretical ideal citing.")'''
text = text.replace(old_metrics, new_metrics)

with open('src/ui/app.py', 'w', encoding='utf-8') as f:
    f.write(text)
print("done")
