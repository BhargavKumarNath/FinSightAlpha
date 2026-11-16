"""Retrieval configurations for the Phase 3 comparison.

All four modes reuse the production `HybridRetriever` components (same models,
same Qdrant collection, same BM25 index, same RRF function) so the comparison
isolates the retrieval strategy and nothing else.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from src.benchmark.corpus import BM25_PATH, COLLECTION, QDRANT_PATH
from src.retrieval.hybrid_retriever import HybridRetriever

Mode = Literal["dense", "sparse", "hybrid", "hybrid_rerank"]
MODES: tuple[Mode, ...] = ("dense", "sparse", "hybrid", "hybrid_rerank")


class RetrievalBench:
    def __init__(self) -> None:
        self.r = HybridRetriever(collection_name=COLLECTION, qdrant_path=QDRANT_PATH,
                                 bm25_path=str(BM25_PATH), bm25_dir="data/p3")
        self.r.load_bm25(COLLECTION)

    def close(self) -> None:
        self.r.close()

    def retrieve(self, query: str, mode: Mode, top_n: int = 8, fetch_k: int = 50) -> list[dict]:
        """Return ranked [{'id', 'text', 'score'}]; `id` is the corpus chunk index."""
        r = self.r
        q_vec = r.embedding_model.encode(query).tolist()
        dense = r.qdrant_client.query_points(collection_name=COLLECTION, query=q_vec,
                                             limit=fetch_k).points
        scores = r.bm25.get_scores(r._tokenize(query))
        idxs = np.argsort(scores)[::-1][:fetch_k]
        sparse = [{"id": int(i)} for i in idxs if scores[i] > 0]
        if mode == "dense":
            return [{"id": int(p.id), "text": p.payload["text"], "score": float(p.score)}
                    for p in dense[:top_n]]
        if mode == "sparse":
            return [{"id": s["id"], "text": r.corpus_metadata[s["id"]]["text"],
                     "score": float(scores[s["id"]])} for s in sparse[:top_n]]
        pool = max(top_n * 2, 20)
        rrf = r._rrf(dense, sparse, top_n=pool)
        if mode == "hybrid":
            return [{"id": x["id"], "text": x["text"], "score": x["rrf_score"]} for x in rrf[:top_n]]
        ce = r.reranker.predict([[query, x["text"]] for x in rrf])
        ranked = sorted(zip(rrf, ce), key=lambda t: float(t[1]), reverse=True)
        return [{"id": x["id"], "text": x["text"], "score": float(s)} for x, s in ranked[:top_n]]
