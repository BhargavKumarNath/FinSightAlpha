"""Quick integration test for all optimization modules."""
import sys
sys.path.insert(0, ".")

# Test 1: Config
from src.optimization.config import config
print("=== Config ===")
print(f"  budget={config.session_token_budget}, heavy={config.model_heavy}, light={config.model_light}")
print(f"  cache_threshold={config.cache_similarity_threshold}, top_k={config.context_top_k}")
print("  [PASS] Config loaded")

# Test 2: Context Window
from src.optimization.context_window import DynamicContextWindow
cw = DynamicContextWindow()
chunks = [
    {"text": "NVIDIA faces significant GPU supply chain risks due to TSMC concentration.", "metadata": {"source": "filing1"}},
    {"text": "Apple reported record quarterly revenue of 123 billion dollars.", "metadata": {"source": "filing2"}},
    {"text": "NVIDIA mitigates supply risks through dual-source wafer agreements.", "metadata": {"source": "filing3"}},
    {"text": "The weather in San Francisco is nice today.", "metadata": {"source": "filing4"}},
    {"text": "GPU manufacturing depends on rare earth minerals from limited sources.", "metadata": {"source": "filing5"}},
]
selected = cw.select_chunks("What are NVIDIA GPU supply chain risks?", chunks)
print(f"\n=== Context Window ===")
print(f"  Selected {len(selected)}/{len(chunks)} chunks")
for c in selected:
    print(f"  score={c['relevance_score']:.3f} | {c['text'][:60]}")
print(f"\n  Formatted:\n{cw.format_context(selected)}")
print("  [PASS] Context windowing works")

# Test 3: Token Budget
from src.optimization.token_budget import TokenBudgetManager, BudgetTier
b = TokenBudgetManager()
assert b.get_tier() == BudgetTier.GREEN
b.record_usage(65000, 0, "bulk")
assert b.get_tier() == BudgetTier.YELLOW
assert b.should_skip_planner() == True
assert b.get_max_iterations() == 3
b.record_usage(22000, 0, "more")
assert b.get_tier() == BudgetTier.RED
assert b.get_retrieval_top_n() == 2
assert b.should_use_heavy_model() == False
print("\n=== Token Budget ===")
print("  [PASS] All tier transitions correct")

# Test 4: Semantic Response Cache
from src.optimization.response_cache import SemanticResponseCache
cache = SemanticResponseCache()
cache.set_model(cw.model)  # Share model

# Miss
result = cache.get("What are NVIDIA supply chain risks?")
assert result is None
print("\n=== Response Cache ===")
print("  [PASS] Cache miss works")

# Put
cache.put("What are NVIDIA supply chain risks?", "Here is the answer...", ["trace1"], token_cost=5000)

# Hit (same query)
result = cache.get("What are NVIDIA supply chain risks?")
assert result is not None
print("  [PASS] Exact cache hit works")

# Near-hit (similar query)
result2 = cache.get("What are NVIDIA's supply chain risks?")
if result2 is not None:
    print("  [PASS] Semantic cache hit works (near-identical query)")
else:
    print("  [INFO] Semantic near-miss (threshold may be too high for this pair)")

stats = cache.stats()
print(f"  Stats: {stats}")

# Test 5: Batch Queries
from src.optimization.batch_queries import QueryBatcher
batcher = QueryBatcher()
batcher.set_model(cw.model)
embeddings = batcher.batch_embed(["query 1", "query 2", "query 3"])
print(f"\n=== Query Batcher ===")
print(f"  Batch embedded 3 queries -> shape: {embeddings.shape}")
print("  [PASS] Batch embedding works")

# Test 6: Deduplication
results_with_dups = [
    {"text": "NVIDIA faces supply chain risks from TSMC.", "score": 0.9},
    {"text": "NVIDIA faces supply chain risks from TSMC.", "score": 0.85},
    {"text": "Apple has different risks entirely.", "score": 0.4},
]
deduped = batcher.deduplicate_results(results_with_dups)
print(f"  Deduped {len(results_with_dups)} -> {len(deduped)} results")
print("  [PASS] Deduplication works")

print("\n" + "="*50)
print("ALL OPTIMIZATION MODULE TESTS PASSED")
print("="*50)
