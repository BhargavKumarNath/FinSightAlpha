"""
Tiered Model Router.

Routes LLM calls to the cheapest model capable of handling the task.
Planning and classification use a fast 8B model; reasoning and
synthesis use the full 70B model. Under budget pressure (RED tier),
all calls fall back to the 8B model.
"""

from typing import Optional
from langchain_groq import ChatGroq
from langchain_core.tools import BaseTool
from src.optimization.config import config
from src.optimization.token_budget import TokenBudgetManager, BudgetTier

class ModelRouter:
    """
    Manages model instances and routes tasks to the appropriate model.

    Model mapping (GREEN/YELLOW tier):
      - Planner:   llama-3.1-8b-instant    (fast, cheap)
      - Agent:     llama-3.3-70b-versatile  (powerful, reasoning)

    Under RED tier:
      - All tasks: llama-3.1-8b-instant     (survive rate limits)
    """

    def __init__(
        self,
        budget_manager: Optional[TokenBudgetManager] = None,
        heavy_model: str = None,
        light_model: str = None,
    ):
        self.budget = budget_manager
        self._heavy_name = heavy_model or config.model_heavy
        self._light_name = light_model or config.model_light
        self._temperature = config.model_temperature

        # Lazy-initialized model instances (created once, reused)
        self._heavy_llm: Optional[ChatGroq] = None
        self._light_llm: Optional[ChatGroq] = None

    @property
    def heavy_llm(self) -> ChatGroq:
        """The powerful reasoning model (70B)."""
        if self._heavy_llm is None:
            self._heavy_llm = ChatGroq(
                model=self._heavy_name,
                temperature=self._temperature,
            )
        return self._heavy_llm

    @property
    def light_llm(self) -> ChatGroq:
        """The fast classification model (8B)."""
        if self._light_llm is None:
            self._light_llm = ChatGroq(
                model=self._light_name,
                temperature=self._temperature,
            )
        return self._light_llm

    def get_planner_llm(self) -> ChatGroq:
        """
        Get the LLM for planning tasks.
        Always uses the light model — planning is a trivial task.
        """
        return self.light_llm

    def get_agent_llm(self, tools: list = None) -> ChatGroq:
        """
        Get the LLM for agent reasoning tasks.
        Uses heavy model under GREEN/YELLOW, falls back to light under RED.

        Args:
            tools: Optional list of tools to bind to the model.

        Returns:
            ChatGroq instance (with tools bound if provided).
        """
        if self.budget and not self.budget.should_use_heavy_model():
            print(f"  [ModelRouter] RED budget tier — using light model: {self._light_name}")
            llm = self.light_llm
        else:
            llm = self.heavy_llm

        if tools:
            return llm.bind_tools(tools)
        return llm

    def get_current_model_name(self, task: str = "agent") -> str:
        """Return the model name that would be used for the given task."""
        if task == "planner":
            return self._light_name

        if self.budget and not self.budget.should_use_heavy_model():
            return self._light_name
        return self._heavy_name

    def info(self) -> dict:
        """Return current routing configuration."""
        tier = self.budget.get_tier().value if self.budget else "unknown"
        return {
            "heavy_model": self._heavy_name,
            "light_model": self._light_name,
            "current_tier": tier,
            "planner_model": self._light_name,
            "agent_model": self.get_current_model_name("agent"),
        }
