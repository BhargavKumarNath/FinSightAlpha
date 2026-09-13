"""
Retrieval coverage: index consistency (fast, no models) + an end-to-end
hybrid search integration test (slow, loads the embedding + cross-encoder
models — marked `integration` and opt-in).
"""
import pickle
from pathlib import Path

import pytest

from src.retrieval.collection_manager import CollectionManager
from src.retrieval.hybrid_retriever import HybridRetriever

QDRANT_PATH = "data/qdrant_db"
REGISTRY_DIR = "data"
COLLECTION = "sec_filings"


def test_index_consistency_registry_qdrant_bm25_agree():
    """
    Formalizes the Phase 1 reconciliation invariant: registry chunk_count,
    Qdrant points_count, and both BM25 pickle corpora must all agree.
    This is the exact check that would have caught the 284-vs-288 drift
    (data/processed/*.jsonl silently going stale relative to the live index).
    """
    mgr = CollectionManager(qdrant_path=QDRANT_PATH, registry_dir=REGISTRY_DIR)
    try:
        docs = mgr.registry.list_documents(COLLECTION)
        assert len(docs) > 0, f"no documents registered for collection '{COLLECTION}'"
        registry_chunk_count = sum(d.chunk_count for d in docs)

        info = mgr.client.get_collection(COLLECTION)
        qdrant_points_count = info.points_count

        with open(Path(REGISTRY_DIR) / "bm25_index.pkl", "rb") as f:
            _, default_corpus = pickle.load(f)
        with open(Path(REGISTRY_DIR) / f"bm25_{COLLECTION}.pkl", "rb") as f:
            _, collection_corpus = pickle.load(f)

        assert registry_chunk_count == qdrant_points_count
        assert len(default_corpus) == qdrant_points_count
        assert len(collection_corpus) == qdrant_points_count
    finally:
        mgr.close()


@pytest.mark.integration
def test_hybrid_retriever_scratch_index_returns_relevant_results(tmp_path):
    """
    Builds a throwaway index (own Qdrant path, own BM25 file) from the real
    data/processed/*.jsonl and asserts a real query returns relevant,
    on-topic results through the full pipeline: embed -> Qdrant dense
    search -> BM25 sparse search -> RRF fusion -> cross-encoder rerank.

    Slow (loads SentenceTransformer + CrossEncoder): run explicitly with
    `pytest -m integration`.
    """
    retriever = HybridRetriever(
        collection_name="test_sec_filings",
        qdrant_path=str(tmp_path / "qdrant_db"),
        bm25_path=str(tmp_path / "bm25_index.pkl"),
        bm25_dir=str(tmp_path),
    )
    try:
        retriever.build_index("data/processed")

        results = retriever.search(
            "What are NVIDIA's primary strategies for mitigating supply chain risk?",
            top_n=5,
        )

        assert len(results) > 0
        assert all("text" in r and "score" in r for r in results)
        assert any(
            keyword in r["text"].lower()
            for r in results
            for keyword in ("suppl", "manufactur", "tsmc")
        )
    finally:
        retriever.close()
