"""
Query Batching for Evaluation & Bulk Retrieval.

Batches embedding computation for multiple queries into single encode()
calls, and supports batched Qdrant searches. Primarily benefits the
evaluation pipeline where multiple queries are processed sequentially.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

from src.optimization.config import config


class QueryBatcher:
    """
    Batches multiple queries for efficient embedding and retrieval.

    Usage:
        batcher = QueryBatcher(embedding_model)
        embeddings = batcher.batch_embed(["query1", "query2", "query3"])
        # Use embeddings with retriever.search() to avoid re-encoding
    """

    def __init__(
        self,
        embedding_model: Optional[SentenceTransformer] = None,
        batch_size: int = None,
    ):
        self._model = embedding_model
        self.batch_size = batch_size or config.batch_max_size

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    def set_model(self, model: SentenceTransformer):
        """Inject shared embedding model."""
        self._model = model

    def batch_embed(
        self,
        queries: List[str],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode multiple queries in a single batched call.

        Args:
            queries: List of query strings.
            normalize: Whether to L2-normalize embeddings.

        Returns:
            numpy array of shape (len(queries), embedding_dim).
        """
        if not queries:
            return np.array([])

        print(f"  [Batcher] Encoding {len(queries)} queries in one batch call")
        embeddings = self.model.encode(
            queries,
            batch_size=self.batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return embeddings

    def batch_search(
        self,
        retriever,
        queries: List[str],
        top_n: int = 5,
    ) -> List[List[Dict[str, Any]]]:
        """
        Batch-search multiple queries, reusing pre-computed embeddings
        where the retriever supports it.

        Args:
            retriever: HybridRetriever instance.
            queries: List of query strings.
            top_n: Results per query.

        Returns:
            List of result lists, one per query.
        """
        # Pre-compute all embeddings in one batch
        embeddings = self.batch_embed(queries)

        results = []
        for i, query in enumerate(queries):
            # Pass pre-computed embedding to avoid re-encoding
            result = retriever.search(
                query,
                top_n=top_n,
                query_embedding=embeddings[i],
            )
            results.append(result)

        return results

    def deduplicate_results(
        self,
        results: List[Dict[str, Any]],
        overlap_threshold: float = 0.9,
    ) -> List[Dict[str, Any]]:
        """
        Remove near-duplicate chunks from search results.

        Uses character-level Jaccard similarity as a fast proxy.
        """
        if len(results) <= 1:
            return results

        unique = [results[0]]
        for candidate in results[1:]:
            is_dup = False
            cand_set = set(candidate.get("text", "").split())
            for kept in unique:
                kept_set = set(kept.get("text", "").split())
                if not cand_set or not kept_set:
                    continue
                intersection = len(cand_set & kept_set)
                union = len(cand_set | kept_set)
                jaccard = intersection / union if union > 0 else 0
                if jaccard >= overlap_threshold:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(candidate)

        if len(unique) < len(results):
            print(f"  [Batcher] Deduped {len(results)} -> {len(unique)} results")
        return unique
