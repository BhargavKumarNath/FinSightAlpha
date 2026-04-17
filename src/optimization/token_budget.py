"""
Token Budget Manager.

Tracks per-session token consumption and enforces graceful degradation
as rate limits approach. Three tiers: GREEN (full), YELLOW (reduced),
RED (minimal). Prevents hard rate-limit failures by progressively
deprioritizing non-critical LLM calls.
"""

import time
import threading
from enum import Enum
from typing import Dict, Any

from src.optimization.config import config


class BudgetTier(Enum):
    """Budget pressure tiers with associated behavior policies."""
    GREEN = "green"    # 0-60%: Full pipeline
    YELLOW = "yellow"  # 60-85%: Skip planner, reduce iterations
    RED = "red"        # 85-100%: Single-pass, cheap model, minimal context


class TokenBudgetManager:
    """
    Tracks token usage per session and returns the appropriate budget tier.

    Usage:
        budget = TokenBudgetManager()
        tier = budget.get_tier()
        if tier == BudgetTier.RED:
            # Use cheap model, skip planner, reduce context
        budget.record_usage(input_tokens=500, output_tokens=200)
    """

    def __init__(
        self,
        session_budget: int = None,
        yellow_pct: float = None,
        red_pct: float = None,
    ):
        self.session_budget = session_budget or config.session_token_budget
        self.yellow_threshold = int((yellow_pct or config.budget_yellow_pct) * self.session_budget)
        self.red_threshold = int((red_pct or config.budget_red_pct) * self.session_budget)

        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._call_count = 0
        self._session_start = time.time()
        self._lock = threading.Lock()

        # Per-call log for debugging
        self._call_log = []

    @property
    def total_tokens(self) -> int:
        return self._total_input_tokens + self._total_output_tokens

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.session_budget - self.total_tokens)

    @property
    def usage_pct(self) -> float:
        return self.total_tokens / self.session_budget if self.session_budget > 0 else 1.0

    def get_tier(self) -> BudgetTier:
        """Return the current budget tier based on cumulative usage."""
        used = self.total_tokens
        if used >= self.red_threshold:
            return BudgetTier.RED
        elif used >= self.yellow_threshold:
            return BudgetTier.YELLOW
        return BudgetTier.GREEN

    def record_usage(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        call_label: str = "",
    ) -> BudgetTier:
        """
        Record token usage from an LLM call. Returns the new tier.

        Args:
            input_tokens: Tokens in the prompt.
            output_tokens: Tokens in the response.
            call_label: Human-readable label (e.g., 'planner', 'agent_iter_2').
        """
        with self._lock:
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens
            self._call_count += 1
            self._call_log.append({
                "call": self._call_count,
                "label": call_label,
                "input": input_tokens,
                "output": output_tokens,
                "cumulative": self.total_tokens,
                "tier": self.get_tier().value,
            })

        tier = self.get_tier()
        print(f"  [Budget] +{input_tokens + output_tokens} tokens "
              f"({call_label}) | total={self.total_tokens}/{self.session_budget} "
              f"({self.usage_pct:.0%}) | tier={tier.value}")
        return tier

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count from text length."""
        return int(len(text) * config.tokens_per_char)

    def can_afford(self, estimated_tokens: int) -> bool:
        """Check if an operation's estimated cost fits the remaining budget."""
        return self.remaining_tokens >= estimated_tokens

    def should_skip_planner(self) -> bool:
        """Planner is skipped under YELLOW and RED to save ~800 tokens."""
        return self.get_tier() != BudgetTier.GREEN

    def get_max_iterations(self) -> int:
        """Return max agent iterations for the current tier."""
        tier = self.get_tier()
        if tier == BudgetTier.RED:
            return 2
        elif tier == BudgetTier.YELLOW:
            return 3
        return 6  # Default (GREEN)

    def get_retrieval_top_n(self) -> int:
        """Return top_n for retrieval based on current tier."""
        tier = self.get_tier()
        if tier == BudgetTier.RED:
            return 2
        elif tier == BudgetTier.YELLOW:
            return 3
        return 5  # Default (GREEN)

    def should_use_heavy_model(self) -> bool:
        """Heavy model is reserved for GREEN and YELLOW tiers."""
        return self.get_tier() != BudgetTier.RED

    def stats(self) -> Dict[str, Any]:
        """Return budget usage statistics."""
        return {
            "total_tokens": self.total_tokens,
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
            "remaining_tokens": self.remaining_tokens,
            "usage_pct": round(self.usage_pct, 3),
            "tier": self.get_tier().value,
            "call_count": self._call_count,
            "session_seconds": round(time.time() - self._session_start, 1),
            "call_log": self._call_log[-10:],  # Last 10 calls
        }

    def reset(self) -> None:
        """Reset budget for a new session."""
        with self._lock:
            self._total_input_tokens = 0
            self._total_output_tokens = 0
            self._call_count = 0
            self._session_start = time.time()
            self._call_log.clear()
