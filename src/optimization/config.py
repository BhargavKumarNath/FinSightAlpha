"""
Centralised configuration for all token optimization parameters.

All defaults are tuned for Groq free-tier rate limits.
Override via environment variables prefixed with FINSIGHT_OPT_.
"""

import os
from dataclasses import dataclass, field


@dataclass
class OptimizationConfig:
    """Single source of truth for all optimization knobs."""

    # Semantic Response Cache
    cache_max_size: int = 500           # Max cached responses
    cache_ttl_seconds: int = 3600       # 1 hour TTL
    cache_similarity_threshold: float = 0.92  # Cosine sim threshold for cache hit

    # Dynamic Context Window
    context_top_k: int = 8              # Max chunks injected per query
    context_max_chunk_chars: int = 1500  # Truncate individual chunks
    context_max_total_tokens: int = 6000 # Hard cap on total context tokens
    context_relevance_floor: float = 0.15  # Min similarity to include chunk

    # Token Budget Manager
    session_token_budget: int = 100_000  # Total tokens per session
    budget_yellow_pct: float = 0.60      # Yellow tier threshold
    budget_red_pct: float = 0.85         # Red tier threshold
    tokens_per_char: float = 0.25        # Estimation: 4 chars ≈ 1 token

    # Model Router
    model_heavy: str = "llama-3.3-70b-versatile"   # Reasoning/synthesis
    model_light: str = "llama-3.1-8b-instant"      # Planning/classification
    model_temperature: float = 0.0

    # Retriever Cache
    retriever_cache_maxsize: int = 256   # Max cached query embeddings
    retriever_result_cache_ttl: int = 300  # 5 min result cache TTL

    # Query Batching
    batch_window_ms: int = 100           # Batching time window
    batch_max_size: int = 8              # Max queries per batch

    def __post_init__(self):
        """Override defaults from environment variables."""
        prefix = "FINSIGHT_OPT_"
        for fld in self.__dataclass_fields__:
            env_key = f"{prefix}{fld.upper()}"
            env_val = os.environ.get(env_key)
            if env_val is not None:
                field_type = type(getattr(self, fld))
                try:
                    setattr(self, fld, field_type(env_val))
                except (ValueError, TypeError):
                    pass  # Skip invalid env overrides silently


# Module-level singleton — import and use directly
config = OptimizationConfig()
