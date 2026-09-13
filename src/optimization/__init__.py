"""
Token Optimization Layer for FinSight-Alpha.

Provides modular, composable optimizations for LLM token efficiency
and rate-limit resilience without altering the RAG architecture.

Submodule re-exports below are lazy (PEP 562 module __getattr__), not
eager. `context_window` and `response_cache` import `sentence-transformers`
at module level; every actual call site in this repo already imports its
specific submodule directly (e.g. `from src.optimization.config import
config`), so eagerly importing everything here just to populate this
package's __all__ forced sentence-transformers/torch onto anything that
imports so much as `src.optimization.config` — including the standalone
UI (`src/ui/`), whose requirements.txt deliberately excludes those heavy
backend dependencies.
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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

_LAZY_ATTRS = {
    "OptimizationConfig": ("src.optimization.config", "OptimizationConfig"),
    "DynamicContextWindow": ("src.optimization.context_window", "DynamicContextWindow"),
    "SemanticResponseCache": ("src.optimization.response_cache", "SemanticResponseCache"),
    "TokenBudgetManager": ("src.optimization.token_budget", "TokenBudgetManager"),
    "BudgetTier": ("src.optimization.token_budget", "BudgetTier"),
    "ModelRouter": ("src.optimization.model_router", "ModelRouter"),
    "QueryBatcher": ("src.optimization.batch_queries", "QueryBatcher"),
}


def __getattr__(name: str):
    """Import the owning submodule only when its symbol is actually accessed
    as `src.optimization.<name>` (PEP 562). Submodule-direct imports like
    `from src.optimization.config import config` never go through this."""
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        value = getattr(importlib.import_module(module_name), attr_name)
        globals()[name] = value  # cache on this module for subsequent access
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
