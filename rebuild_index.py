"""
Rebuild Script — Re-ingest documents and rebuild the vector + BM25 indices.

Run this after fixing the document processor to replace the poisoned index.
"""
import sys
import os
import shutil

sys.path.insert(0, ".")
from src.ingestion.document_processor import DocumentProcessor
from src.retrieval.hybrid_retriever import HybridRetriever

PROCESSED_DIR = "data/processed"
RAW_DIR = "data/raw"
QDRANT_PATH = "data/qdrant_db"
BM25_PATH = "data/bm25_index.pkl"


def main():
    print("=" * 60)
    print("FinSightAlpha Index Rebuild")
    print("=" * 60)

    # Step 1: Clear old processed data
    print("\n[1/3] Clearing old processed data...")
    if os.path.exists(PROCESSED_DIR):
        for f in os.listdir(PROCESSED_DIR):
            fp = os.path.join(PROCESSED_DIR, f)
            if os.path.isfile(fp):
                os.remove(fp)
                print(f"  Removed: {f}")

    # Step 2: Re-process raw documents
    print("\n[2/3] Re-processing raw documents with clean parser...")
    processor = DocumentProcessor(
        max_chunk_chars=1500,
        overlap_chars=200,
        min_chunk_chars=80,
    )
    processor.process_directory(input_dir=RAW_DIR, output_dir=PROCESSED_DIR)

    # Step 3: Rebuild vector + BM25 indices
    print("\n[3/3] Rebuilding Qdrant + BM25 indices...")
    retriever = HybridRetriever(
        qdrant_path=QDRANT_PATH,
        bm25_path=BM25_PATH,
    )

    try:
        retriever.build_index(PROCESSED_DIR, batch_size=100)

        # Quick validation
        print("\n" + "=" * 60)
        print("VALIDATION: Testing retrieval on rebuilt index")
        print("=" * 60)

        test_queries = [
            "What is NVIDIA's data center revenue?",
            "What risk factors does NVIDIA list related to China?",
            "What were NVIDIA's R&D expenses?",
            "Annual Report on Form 10-K filing date",
        ]

        for q in test_queries:
            results = retriever.search(q, top_n=3)
            print(f"\n Query: {q}")
            if results:
                for i, r in enumerate(results, 1):
                    text_preview = r["text"][:150].replace("\n", " ")
                    print(f"  [{i}] score={r['score']:.3f} | {text_preview}...")
            else:
                print("  [!] No results found")

        print("\n" + "=" * 60)
        print("Index rebuild complete!")
        print("=" * 60)

    finally:
        retriever.close()


if __name__ == "__main__":
    main()
