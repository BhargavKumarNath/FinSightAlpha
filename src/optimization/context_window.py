"""
Dynamic Context Windowing.

Selects only the top-K most semantically relevant chunks per query
and truncates them to fit within a strict token budget. Eliminates
the wasteful practice of injecting all retrieved chunks at full length.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

from src.optimization.config import config


class DynamicContextWindow:
    """
    Filters and truncates retrieved chunks to maximize information density
    within a fixed token budget.

    Pipeline:
      1. Embed query + all candidate chunks
      2. Score each chunk by cosine similarity to the query
      3. Filter chunks below relevance floor
      4. Take top-K by score
      5. Truncate each chunk to max_chunk_chars at sentence boundaries
      6. Enforce total token budget cap
    """

    def __init__(
        self,
        embedding_model: Optional[SentenceTransformer] = None,
        top_k: int = None,
        max_chunk_chars: int = None,
        max_total_tokens: int = None,
        relevance_floor: float = None,
    ):
        # Use shared embedding model if provided, else load a lightweight one
        self._model = embedding_model
        self.top_k = top_k or config.context_top_k
        self.max_chunk_chars = max_chunk_chars or config.context_max_chunk_chars
        self.max_total_tokens = max_total_tokens or config.context_max_total_tokens
        self.relevance_floor = relevance_floor or config.context_relevance_floor

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load embedding model only when needed."""
        if self._model is None:
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    def set_model(self, model: SentenceTransformer):
        """Inject a shared embedding model to avoid duplicate loading."""
        self._model = model

    def select_chunks(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        query_embedding: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Select and truncate the most relevant chunks for a given query.

        Args:
            query: The user's search query.
            chunks: List of dicts with 'text' and 'metadata' keys.
            query_embedding: Pre-computed query embedding (avoids re-encoding).

        Returns:
            Filtered, truncated list of chunk dicts with added 'relevance_score'.
        """
        if not chunks:
            return []

        # Step 1: Encode query (reuse if provided)
        if query_embedding is None:
            query_embedding = self.model.encode(query, normalize_embeddings=True)
        elif not _is_normalized(query_embedding):
            query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)

        # Step 2: Encode all chunk texts in a single batch call
        texts = [c.get("text", "") for c in chunks]
        chunk_embeddings = self.model.encode(
            texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False
        )

        # Step 3: Compute cosine similarities (dot product on normalized vectors)
        similarities = np.dot(chunk_embeddings, query_embedding)

        # Step 4: Filter by relevance floor and sort descending
        scored = [
            (chunks[i], float(similarities[i]))
            for i in range(len(chunks))
            if similarities[i] >= self.relevance_floor
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Step 5: Take top-K
        selected = scored[: self.top_k]

        # Step 6: Truncate chunks and enforce total token budget
        result = []
        total_tokens = 0

        for chunk, score in selected:
            text = chunk.get("text", "")
            truncated = self._truncate_at_sentence(text, self.max_chunk_chars)
            chunk_tokens = self._estimate_tokens(truncated)

            # Stop adding if we'd exceed total budget
            if total_tokens + chunk_tokens > self.max_total_tokens:
                # Try fitting a shorter version
                remaining_tokens = self.max_total_tokens - total_tokens
                remaining_chars = int(remaining_tokens / config.tokens_per_char)
                if remaining_chars > 100:  # Only worth including if >100 chars
                    truncated = self._truncate_at_sentence(text, remaining_chars)
                    chunk_tokens = self._estimate_tokens(truncated)
                else:
                    break

            result.append({
                "text": truncated,
                "metadata": chunk.get("metadata", {}),
                "relevance_score": score,
            })
            total_tokens += chunk_tokens

        return result

    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format selected chunks into a compact context string for the LLM."""
        if not chunks:
            return "No relevant documents found."

        parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("metadata", {}).get("source", "unknown")
            # Extract just the filename from the full path for brevity
            source_short = source.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] if source else "unknown"
            score = chunk.get("relevance_score", 0)
            parts.append(f"[{i}|{source_short}|rel:{score:.2f}] {chunk['text']}")

        return "\n\n".join(parts)

    @staticmethod
    def _truncate_at_sentence(text: str, max_chars: int) -> str:
        """Truncate text at the last sentence boundary within max_chars."""
        if len(text) <= max_chars:
            return text

        truncated = text[:max_chars]

        # Find last sentence-ending punctuation
        for sep in [". ", ".\n", "! ", "? "]:
            last_sep = truncated.rfind(sep)
            if last_sep > max_chars * 0.5:  # Only if we keep >50% of the text
                return truncated[: last_sep + 1].strip()

        # Fallback: truncate at last space
        last_space = truncated.rfind(" ")
        if last_space > max_chars * 0.5:
            return truncated[:last_space].strip() + "..."

        return truncated.strip() + "..."

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count using the 4-chars-per-token heuristic."""
        return int(len(text) * config.tokens_per_char)


def _is_normalized(vec: np.ndarray, tol: float = 0.01) -> bool:
    """Check if a vector is approximately unit-normalized."""
    return abs(np.linalg.norm(vec) - 1.0) < tol
