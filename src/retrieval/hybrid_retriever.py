import os
import json
import pickle
import glob
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
import torch
import warnings

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi
from transformers import logging

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore")
logging.set_verbosity_error()


# HYBRID RETRIEVER
class HybridRetriever:
    def __init__(
        self,
        collection_name: str = "sec_filings",
        qdrant_path: str = "data/qdrant_db",
        bm25_path: str = "data/bm25_index.pkl",
    ):
        self.collection_name = collection_name
        self.qdrant_path = qdrant_path
        self.bm25_path = bm25_path

        # GPU detection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Embedding model running on: {self.device.upper()}")

        # Model
        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2", device=self.device
        )
        self.vector_size = self.embedding_model.get_embedding_dimension()

        # Qdrant
        os.makedirs(self.qdrant_path, exist_ok=True)
        self.qdrant_client = QdrantClient(path=self.qdrant_path)

        # BM25
        self.bm25: BM25Okapi = None
        self.corpus_metadata: List[Dict[str, Any]] = []

    # TOKENIZER
    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    # BUILD INDEX
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
    def _rrf(self, dense, sparse, k=60, top_n=10):
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
                "score": score,
            }
            for doc_id, score in ranked[:top_n]
        ]

    # SEARCH
    def search(self, query: str, top_n: int = 5, fetch_k: int = 50):
        if self.bm25 is None:
            self.load_bm25()

        print(f"\nQuery: {query}")

        # Dense
        q_vec = self.embedding_model.encode(query).tolist()
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

        return self._rrf(dense, sparse, top_n=top_n)

    # CLEAN SHUTDOWN
    def close(self):
        self.qdrant_client.close()


# MAIN
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

    finally:
        retriever.close()