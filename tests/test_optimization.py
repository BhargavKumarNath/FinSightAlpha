"""Integration tests for all optimization modules."""
import sys
import pytest
from src.optimization.config import config
from src.optimization.context_window import DynamicContextWindow
from src.optimization.token_budget import TokenBudgetManager, BudgetTier
from src.optimization.response_cache import SemanticResponseCache
from src.optimization.batch_queries import QueryBatcher


def test_config():
    assert config.session_token_budget == 100_000
    assert config.model_heavy == "llama-3.3-70b-versatile"
    assert config.model_light == "llama-3.1-8b-instant"
    assert config.cache_similarity_threshold == 0.92
    assert config.context_top_k == 8


def test_context_window():
    cw = DynamicContextWindow()
    chunks = [
        {"text": "NVIDIA faces significant GPU supply chain risks due to TSMC concentration.", "metadata": {"source": "filing1"}},
        {"text": "Apple reported record quarterly revenue of 123 billion dollars.", "metadata": {"source": "filing2"}},
        {"text": "NVIDIA mitigates supply risks through dual-source wafer agreements.", "metadata": {"source": "filing3"}},
        {"text": "The weather in San Francisco is nice today.", "metadata": {"source": "filing4"}},
        {"text": "GPU manufacturing depends on rare earth minerals from limited sources.", "metadata": {"source": "filing5"}},
    ]
    selected = cw.select_chunks("What are NVIDIA GPU supply chain risks?", chunks)
    assert len(selected) > 0
    formatted = cw.format_context(selected)
    assert "NVIDIA" in formatted


def test_token_budget():
    b = TokenBudgetManager()
    assert b.get_tier() == BudgetTier.GREEN
    assert b.get_retrieval_top_n() == 8
    
    b.record_usage(65000, 0, "bulk")
    assert b.get_tier() == BudgetTier.YELLOW
    assert b.should_skip_planner() is True
    assert b.get_max_iterations() == 3
    assert b.get_retrieval_top_n() == 5
    
    b.record_usage(22000, 0, "more")
    assert b.get_tier() == BudgetTier.RED
    assert b.get_retrieval_top_n() == 4
    assert b.should_use_heavy_model() is False


def test_semantic_response_cache():
    cw = DynamicContextWindow()
    cache = SemanticResponseCache()
    cache.set_model(cw.model)

    # Miss
    result = cache.get("What are NVIDIA supply chain risks?")
    assert result is None

    # Put
    cache.put("What are NVIDIA supply chain risks?", "Here is the answer...", ["trace1"], token_cost=5000)

    # Hit (same query)
    result = cache.get("What are NVIDIA supply chain risks?")
    assert result is not None
    assert result[0] == "Here is the answer..."


def test_batch_queries():
    cw = DynamicContextWindow()
    batcher = QueryBatcher()
    batcher.set_model(cw.model)
    embeddings = batcher.batch_embed(["query 1", "query 2", "query 3"])
    assert embeddings.shape[0] == 3


def test_deduplication():
    batcher = QueryBatcher()
    results_with_dups = [
        {"text": "NVIDIA faces supply chain risks from TSMC.", "score": 0.9},
        {"text": "NVIDIA faces supply chain risks from TSMC.", "score": 0.85},
        {"text": "Apple has different risks entirely.", "score": 0.4},
    ]
    deduped = batcher.deduplicate_results(results_with_dups)
    assert len(deduped) == 2

