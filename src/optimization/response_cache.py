"""
Semantic Response Cache.

Caches LLM responses keyed by query embedding similarity. Near-identical
queries (cosine sim > threshold) return cached responses, bypassing the
entire LLM pipeline.
"""

import time
import threading
import numpy as np
from typing import Optional, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer

from src.optimization.config import config


class CacheEntry:
    """Single cached response entry."""
    __slots__ = ("query_text", "query_embedding", "response", "trace",
                 "timestamp", "hit_count", "token_cost")

    def __init__(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        response: str,
        trace: list,
        token_cost: int = 0,
    ):
        self.query_text = query_text
        self.query_embedding = query_embedding
        self.response = response
        self.trace = trace
        self.timestamp = time.time()
        self.hit_count = 0
        self.token_cost = token_cost  # Tokens saved per cache hit


class SemanticResponseCache:
    """
    Thread-safe semantic cache that matches queries by embedding similarity.

    Lookup flow:
      1. Encode incoming query
      2. Compute cosine similarity against all cached query embeddings
      3. If max similarity > threshold: return cached response (cache hit)
      4. Otherwise: return None (cache miss, caller runs full pipeline)
      5. After pipeline completes, caller stores the result via put()

    Eviction: LRU by timestamp when max_size is exceeded.
    """

    def __init__(
        self,
        embedding_model: Optional[SentenceTransformer] = None,
        max_size: int = None,
        ttl_seconds: int = None,
        similarity_threshold: float = None,
    ):
        self._model = embedding_model
        self.max_size = max_size or config.cache_max_size
        self.ttl = ttl_seconds or config.cache_ttl_seconds
        self.threshold = similarity_threshold or config.cache_similarity_threshold
        self._entries: Dict[str, CacheEntry] = {}  # keyed by query_text
        self._lock = threading.RLock()

        # Stats
        self.total_hits = 0
        self.total_misses = 0
        self.total_tokens_saved = 0

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    def set_model(self, model: SentenceTransformer):
        """Inject shared embedding model."""
        self._model = model

    def get(self, query: str) -> Optional[Tuple[str, list]]:
        """
        Look up a semantically similar cached response.

        Returns:
            (response_text, reasoning_trace) if cache hit, else None.
        """
        if not self._entries:
            self.total_misses += 1
            return None

        query_emb = self.model.encode(query, normalize_embeddings=True)

        with self._lock:
            self._evict_expired()

            if not self._entries:
                self.total_misses += 1
                return None

            # Batch compare against all cached embeddings
            keys = list(self._entries.keys())
            cached_embs = np.array([
                self._entries[k].query_embedding for k in keys
            ])
            similarities = np.dot(cached_embs, query_emb)

            best_idx = int(np.argmax(similarities))
            best_sim = float(similarities[best_idx])

            if best_sim >= self.threshold:
                entry = self._entries[keys[best_idx]]
                entry.hit_count += 1
                entry.timestamp = time.time()  # Refresh for LRU
                self.total_hits += 1
                self.total_tokens_saved += entry.token_cost
                print(f"  [Cache HIT] sim={best_sim:.3f} | "
                      f"matched='{entry.query_text[:60]}...' | "
                      f"tokens_saved={entry.token_cost}")
                return (entry.response, entry.trace)

        self.total_misses += 1
        return None

    def put(
        self,
        query: str,
        response: str,
        trace: list = None,
        token_cost: int = 0,
    ) -> None:
        """
        Store a query-response pair in the cache.

        Args:
            query: Original query text.
            response: LLM response text.
            trace: Reasoning trace (tool calls).
            token_cost: Estimated tokens this response consumed.
        """
        query_emb = self.model.encode(query, normalize_embeddings=True)

        with self._lock:
            # Evict LRU if at capacity
            while len(self._entries) >= self.max_size:
                self._evict_lru()

            self._entries[query] = CacheEntry(
                query_text=query,
                query_embedding=query_emb,
                response=response,
                trace=trace or [],
                token_cost=token_cost,
            )

    def _evict_expired(self) -> None:
        """Remove entries older than TTL."""
        now = time.time()
        expired = [
            k for k, v in self._entries.items()
            if (now - v.timestamp) > self.ttl
        ]
        for k in expired:
            del self._entries[k]

    def _evict_lru(self) -> None:
        """Remove the least recently used entry."""
        if not self._entries:
            return
        oldest_key = min(self._entries, key=lambda k: self._entries[k].timestamp)
        del self._entries[oldest_key]

    def stats(self) -> Dict[str, Any]:
        """Return cache performance statistics."""
        total = self.total_hits + self.total_misses
        return {
            "size": len(self._entries),
            "max_size": self.max_size,
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "hit_rate": self.total_hits / total if total > 0 else 0.0,
            "total_tokens_saved": self.total_tokens_saved,
        }

    def clear(self) -> None:
        """Flush all cached entries."""
        with self._lock:
            self._entries.clear()
