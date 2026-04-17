"""
RAGAS Evaluator — Token-Optimized.

Evaluates the FinSight-Alpha Agent using RAGAS metrics.
Optimized with:
  - Batched query embeddings (single encode() call for all test queries)
  - Budget manager reset between evaluation runs
  - Shared embedding model across components
"""

import os
import sys
import pandas as pd
from pathlib import Path
from datasets import Dataset
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.llms.base import LangchainLLMWrapper
from ragas.embeddings.base import LangchainEmbeddingsWrapper
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.agents.langgraph_agent import build_graph, get_budget_manager
from src.optimization.batch_queries import QueryBatcher


class GroqSafeWrapper(ChatGroq):
    """ChatGroq wrapper that forces n=1 for RAGAS compatibility.
    
    RAGAS evaluation often requests n>1 completions for metrics like
    AnswerRelevancy. Groq's API doesn't support n>1, so we intercept
    and force n=1, letting RAGAS handle the reduced generation count.
    """
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs["n"] = 1
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs["n"] = 1
        return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)


class RagasEvaluator:
    """
    Evaluates the FinSight-Alpha Agent using RAGAS metrics to prove institutional
    grade accuracy and prevent hallucinations.
    """
    def __init__(self):
        print("Initializing the Agent Graph for Evaluation...")
        self.agent_app = build_graph()
        self.budget_manager = get_budget_manager()

        # Query batcher for efficient embedding computation
        self.batcher = QueryBatcher()

        # Evaluation LLM wrapped for RAGAS (uses LangchainLLMWrapper for
        # compatibility with the legacy evaluate() pipeline)
        print("Initializing evaluation LLM...")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self.eval_llm = LangchainLLMWrapper(
                GroqSafeWrapper(model="llama-3.3-70b-versatile", temperature=0)
            )

        print("Loading local embedding model for evaluation metrics...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self.eval_embeddings = LangchainEmbeddingsWrapper(
                HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            )

    def run_agent_and_collect_data(self, test_questions: list[str]) -> pd.DataFrame:
        """
        Runs the agent against a set of questions, capturing the generated answers
        and the exact contexts retrieved by the tools.

        Optimization: Pre-computes all query embeddings in a single batch call
        before iterating through queries.
        """
        data = {
            "question": [],
            "answer": [],
            "contexts": [],
        }

        # Pre-compute all query embeddings in a single batch
        print(f"\n[Optimization] Batching {len(test_questions)} query embeddings...")
        _ = self.batcher.batch_embed(test_questions)

        for i, question in enumerate(test_questions, 1):
            print(f"\n{'='*60}")
            print(f"[{i}/{len(test_questions)}] Evaluating Query: `{question}`")
            print("="*60)

            # Reset budget for each evaluation query to prevent
            # degradation across the test suite
            self.budget_manager.reset()

            initial_state = {"messages": [HumanMessage(content=question)]}
            final_state = self.agent_app.invoke(initial_state)

            messages = final_state.get("messages", [])

            # Extract the final answer - find last AIMessage with actual content
            # (skip AIMessages that only have tool_calls and no text)
            final_answer = ""
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content and msg.content.strip():
                    # Skip if the content is just hallucinated JSON tool calls
                    if '{"type": "function"' not in msg.content:
                        final_answer = msg.content
                        break

            if not final_answer:
                # Fallback: use the very last message content
                final_answer = messages[-1].content if messages else "No answer generated."
                print(f"  [WARN] Could not find clean final answer, using fallback.")

            # Extract ALL ToolMessage contents as retrieved contexts
            retrieved_contexts = []
            for msg in messages:
                if isinstance(msg, ToolMessage) and msg.content:
                    retrieved_contexts.append(msg.content)

            # Validation logging
            if not retrieved_contexts:
                print(f"  [WARN] No tool contexts retrieved for this query!")
                retrieved_contexts = ["No external context retrieved."]
            else:
                # Show a preview of what was retrieved
                total_chars = sum(len(c) for c in retrieved_contexts)
                print(f"  [OK] Retrieved {len(retrieved_contexts)} context(s) ({total_chars:,} chars total)")

            print(f"  [Answer Preview] {final_answer[:150]}...")

            # Print budget stats for this query
            stats = self.budget_manager.stats()
            print(f"  [Budget] Tokens used: {stats['total_tokens']} | "
                  f"Calls: {stats['call_count']} | Tier: {stats['tier']}")

            data["question"].append(question)
            data["answer"].append(final_answer)
            data["contexts"].append(retrieved_contexts)

        return pd.DataFrame(data)

    def evaluate_performance(self, df: pd.DataFrame):
        """
        Converts the results to a HuggingFace Dataset and runs RAGAS scoring.
        """
        print("\n" + "=" * 60)
        print("Pre-Evaluation Validation")
        print("=" * 60)

        # Validate that contexts were actually retrieved
        total = len(df)
        with_context = sum(
            1 for ctx_list in df["contexts"]
            if ctx_list and ctx_list != ["No external context retrieved."]
        )
        print(f"  Queries with retrieved contexts: {with_context}/{total}")

        if with_context == 0:
            print("  [ERROR] No queries have retrieved contexts!")
            print("  [ERROR] Faithfulness scores will be meaningless.")
            print("  [ERROR] Check agent tool calling pipeline before trusting results.\n")
        else:
            print(f"  [OK] {with_context}/{total} queries have real contexts.\n")

        print("=" * 60)
        print("Running RAGAS 'LLM-as-a-Judge' Evaluation...")
        print("=" * 60)

        dataset = Dataset.from_pandas(df)

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = evaluate(
                dataset=dataset,
                metrics=[
                    Faithfulness(llm=self.eval_llm),
                    AnswerRelevancy(llm=self.eval_llm, embeddings=self.eval_embeddings),
                ],
            )

        eval_df = result.to_pandas()

        if "question" not in eval_df.columns:
            eval_df = eval_df.copy()
            eval_df.insert(0, "question", df["question"].values)

        print("\n--- Evaluation Results ---")
        display_df = eval_df[["question", "faithfulness", "answer_relevancy"]]
        print(display_df.to_markdown(index=False))

        print("\nOverall System Score:")
        avg_faith = eval_df['faithfulness'].mean()
        avg_relev = eval_df['answer_relevancy'].mean()
        print(f"Average Faithfulness:     {avg_faith:.2f}")
        print(f"Average Answer Relevancy: {avg_relev:.2f}")

        # Quality gate
        if avg_faith >= 0.7:
            print("\n[PASS] Faithfulness PASSED (>= 0.70)")
        else:
            print(f"\n[WARN] Faithfulness BELOW threshold ({avg_faith:.2f} < 0.70)")

        # Save report
        os.makedirs("data/reports", exist_ok=True)
        eval_df.to_csv("data/reports/ragas_evaluation_report.csv", index=False)
        print("\nDetailed report saved to data/reports/ragas_evaluation_report.csv")


if __name__ == "__main__":
    evaluator = RagasEvaluator()
    test_suite = [
        "What are NVIDIA's primary strategies for mitigating supply chain constraints?",
        "If NVIDIA's reported data center revenue drops by 10% from $47 billion, what is the new figure?",
        "What does the 10-K say about NVIDIA's secret plans to acquire AMD?"
    ]

    results_df = evaluator.run_agent_and_collect_data(test_suite)
    evaluator.evaluate_performance(results_df)
