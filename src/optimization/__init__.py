"""
Token Optimization Layer for FinSight-Alpha.

Provides modular, composable optimizations for LLM token efficiency
and rate-limit resilience without altering the RAG architecture.
"""

from src.optimization.config import OptimizationConfig
from src.optimization.context_window import DynamicContextWindow
from src.optimization.response_cache import SemanticResponseCache
from src.optimization.token_budget import TokenBudgetManager, BudgetTier
from src.optimization.model_router import ModelRouter
from src.optimization.batch_queries import QueryBatcher

__all__ = [
    "OptimizationConfig",
    "DynamicContextWindow",
    "SemanticResponseCache",
    "TokenBudgetManager",
    "BudgetTier",
    "ModelRouter",
    "QueryBatcher",
]
