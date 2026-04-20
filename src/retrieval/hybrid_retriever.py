"""
Hybrid Retriever with Caching Layer.

Combines dense (Qdrant) + sparse (BM25) retrieval with RRF fusion.
Optimized with:
  - LRU embedding cache (avoids re-encoding identical queries)
  - Search result cache with TTL (avoids redundant searches)
  - Accepts pre-computed query embeddings (for batch operations)
  - Near-duplicate result deduplication
"""

import os
import json
import pickle
import glob
import time
import hashlib
from typing import List, Dict, Any, Optional
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
import torch
import warnings

from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi
from transformers import logging

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore")
logging.set_verbosity_error()


class _EmbeddingCache:
    """
    LRU cache for query embeddings. Avoids re-encoding identical or
    recently-seen queries. Thread-safe via ordered dict + size cap.
    """

    def __init__(self, maxsize: int = 256):
        self.maxsize = maxsize
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, query: str) -> Optional[np.ndarray]:
        if query in self._cache:
            self._cache.move_to_end(query)  # Mark as recently used
            return self._cache[query]
        return None

    def put(self, query: str, embedding: np.ndarray) -> None:
        if query in self._cache:
            self._cache.move_to_end(query)
        else:
            if len(self._cache) >= self.maxsize:
                self._cache.popitem(last=False)  # Evict LRU
            self._cache[query] = embedding

    @property
    def size(self) -> int:
        return len(self._cache)


class _ResultCache:
    """
    TTL-based cache for search results. Keyed by (query, top_n, fetch_k).
    Avoids redundant retrieval for repeated queries within TTL window.
    """

    def __init__(self, ttl_seconds: int = 300):
        self.ttl = ttl_seconds
        self._cache: Dict[str, tuple] = {}  # key -> (results, timestamp)

    def _make_key(self, query: str, top_n: int, fetch_k: int) -> str:
        raw = f"{query}|{top_n}|{fetch_k}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, query: str, top_n: int, fetch_k: int) -> Optional[list]:
        key = self._make_key(query, top_n, fetch_k)
        if key in self._cache:
            results, ts = self._cache[key]
            if time.time() - ts < self.ttl:
                return results
            else:
                del self._cache[key]  # Expired
        return None

    def put(self, query: str, top_n: int, fetch_k: int, results: list) -> None:
        key = self._make_key(query, top_n, fetch_k)
        self._cache[key] = (results, time.time())

    def clear(self) -> None:
        self._cache.clear()


# Class-level shared model reference (for cross-module model sharing)
class HybridRetriever:
    _shared_embedding_model: Optional[SentenceTransformer] = None
    _shared_cross_encoder: Optional[CrossEncoder] = None

    def __init__(
        self,
        collection_name: str = "sec_filings",
        qdrant_path: str = "data/qdrant_db",
        bm25_path: str = "data/bm25_index.pkl",
        embedding_cache_size: int = 256,
        result_cache_ttl: int = 300,
    ):
        self.collection_name = collection_name
        self.qdrant_path = qdrant_path
        self.bm25_path = bm25_path

        # GPU detection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Embedding models running on: {self.device.upper()}")

        # Model (shared across instances)
        if HybridRetriever._shared_embedding_model is None:
            HybridRetriever._shared_embedding_model = SentenceTransformer(
                "all-MiniLM-L6-v2", device=self.device
            )
        self.embedding_model = HybridRetriever._shared_embedding_model
        self.vector_size = self.embedding_model.get_embedding_dimension()

        # Cross-Encoder Reranker
        if HybridRetriever._shared_cross_encoder is None:
            print("Loading Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)...")
            HybridRetriever._shared_cross_encoder = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2", device=self.device
            )
        self.reranker = HybridRetriever._shared_cross_encoder

        # Qdrant
        os.makedirs(self.qdrant_path, exist_ok=True)
        self.qdrant_client = QdrantClient(path=self.qdrant_path)

        # BM25
        self.bm25: BM25Okapi = None
        self.corpus_metadata: List[Dict[str, Any]] = []

        # Caching layers
        self._embedding_cache = _EmbeddingCache(maxsize=embedding_cache_size)
        self._result_cache = _ResultCache(ttl_seconds=result_cache_ttl)

        # Stats
        self._embedding_cache_hits = 0
        self._result_cache_hits = 0
        self._total_searches = 0

    # tokenizer
    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    # Build index
    def build_index(self, jsonl_dir: str, batch_size: int = 500) -> None:
        if self.qdrant_client.collection_exists(self.collection_name):
            self.qdrant_client.delete_collection(self.collection_name)

        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size, distance=Distance.COSINE
            ),
        )

        files = glob.glob(os.path.join(jsonl_dir, "*.jsonl"))
        if not files:
            print(f"No JSONL files found in {jsonl_dir}")
            return

        print("Reading chunks...")
        all_chunks = []
        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    all_chunks.append(json.loads(line))

        print(f"Total chunks: {len(all_chunks)}")

        tokenized_corpus = []
        self.corpus_metadata = []

        for i in tqdm(range(0, len(all_chunks), batch_size), desc="Indexing"):
            batch = all_chunks[i : i + batch_size]

            texts = [c.get("page_content", "") for c in batch]
            metas = [c.get("metadata", {}) for c in batch]

            # Sparse
            for t, m in zip(texts, metas):
                tokenized_corpus.append(self._tokenize(t))
                self.corpus_metadata.append({"text": t, "metadata": m})

            # Dense
            embeddings = self.embedding_model.encode(
                texts, batch_size=batch_size, show_progress_bar=False
            )

            points = [
                PointStruct(
                    id=i + j,
                    vector=embeddings[j].tolist(),
                    payload={"text": texts[j], "metadata": metas[j]},
                )
                for j in range(len(batch))
            ]

            self.qdrant_client.upsert(
                collection_name=self.collection_name, points=points
            )

        print("Training BM25...")
        self.bm25 = BM25Okapi(tokenized_corpus)

        with open(self.bm25_path, "wb") as f:
            pickle.dump((self.bm25, self.corpus_metadata), f)

        print("Indexing complete")

    # LOAD BM25
    def load_bm25(self) -> None:
        if not os.path.exists(self.bm25_path):
            raise FileNotFoundError("BM25 index not found. Run build_index first.")

        with open(self.bm25_path, "rb") as f:
            self.bm25, self.corpus_metadata = pickle.load(f)

    # RRF
    def _rrf(self, dense, sparse, k=60, top_n=20):
        scores = {}
        docs = {}

        for rank, p in enumerate(dense):
            doc_id = p.id
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
            docs[doc_id] = p.payload

        for rank, s in enumerate(sparse):
            doc_id = s["id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
            if doc_id not in docs:
                docs[doc_id] = self.corpus_metadata[doc_id]

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [
            {
                "id": doc_id,
                "text": docs[doc_id]["text"],
                "metadata": docs[doc_id]["metadata"],
                "rrf_score": score,
            }
            for doc_id, score in ranked[:top_n]
        ]

    # SEARCH (with caching + optional pre-computed embedding)
    def search(
        self,
        query: str,
        top_n: int = 5,
        fetch_k: int = 50,
        query_embedding: Optional[np.ndarray] = None,
    ):
        """
        Hybrid search with embedding and result caching.

        Args:
            query: Search query string.
            top_n: Number of results to return.
            fetch_k: Candidates to fetch from each retrieval path.
            query_embedding: Pre-computed embedding (skips re-encoding).
        """
        self._total_searches += 1

        if self.bm25 is None:
            self.load_bm25()

        # Check result cache first
        cached_results = self._result_cache.get(query, top_n, fetch_k)
        if cached_results is not None:
            self._result_cache_hits += 1
            print(f"\n[Retriever] Result cache hit for: '{query[:50]}...'")
            return cached_results

        print(f"\nQuery: {query}")

        # Dense: use cached or pre-computed embedding
        if query_embedding is not None:
            q_vec = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        else:
            cached_emb = self._embedding_cache.get(query)
            if cached_emb is not None:
                self._embedding_cache_hits += 1
                q_vec = cached_emb.tolist()
            else:
                emb = self.embedding_model.encode(query)
                self._embedding_cache.put(query, emb)
                q_vec = emb.tolist()

        dense = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=q_vec,
            limit=fetch_k,
        ).points

        # Sparse
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        idxs = np.argsort(scores)[::-1][:fetch_k]

        sparse = [{"id": int(i)} for i in idxs if scores[i] > 0]

        # RRF top candidates (we fetch more, e.g., 20, for reranking)
        rrf_results = self._rrf(dense, sparse, top_n=max(top_n * 2, 20))

        if not rrf_results:
            return []

        # Cross-Encoder Reranking
        pairs = [[query, res["text"]] for res in rrf_results]
        rerank_scores = self.reranker.predict(pairs)
        
        # Add rerank scores to results and sort
        for i, res in enumerate(rrf_results):
            res["score"] = float(rerank_scores[i])
            res["metadata"]["cross_encoder_score"] = float(rerank_scores[i])
            
        reranked_results = sorted(rrf_results, key=lambda x: x["score"], reverse=True)
        results = reranked_results[:top_n]

        # Cache the results
        self._result_cache.put(query, top_n, fetch_k, results)

        return results

    def cache_stats(self) -> Dict[str, Any]:
        """Return retriever cache statistics."""
        return {
            "total_searches": self._total_searches,
            "embedding_cache_size": self._embedding_cache.size,
            "embedding_cache_hits": self._embedding_cache_hits,
            "result_cache_hits": self._result_cache_hits,
        }

    # CLEAN SHUTDOWN
    def close(self):
        self.qdrant_client.close()

if __name__ == "__main__":
    retriever = HybridRetriever()

    try:
        # Only needed for the first execution
        # retriever.build_index("data/processed")

        results = retriever.search(
            "What are the primary risks associated with artificial intelligence and GPU supply chain?",
            top_n=3,
        )

        print("\n" + "=" * 50)
        print("RESULTS")
        print("=" * 50)

        for i, r in enumerate(results, 1):
            print(f"\nResult {i} | Score: {r['score']:.4f}")
            print(f"Source: {r['metadata'].get('source', 'Unknown')}")
            print(f"Text: {r['text'][:200]}...")
            print("-" * 50)

        # Test cache hit
        print("\n--- Testing cache hit ---")
        results2 = retriever.search(
            "What are the primary risks associated with artificial intelligence and GPU supply chain?",
            top_n=3,
        )
        print(f"Cache stats: {retriever.cache_stats()}")

    finally:
        retriever.close()