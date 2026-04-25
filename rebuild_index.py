"""
Rebuild Script — Re-ingest documents using the production pipeline.

Supports two modes:
  1. Full rebuild (default): Clears and rebuilds everything
  2. Incremental: Adds only new/changed documents

Usage:
    python rebuild_index.py                        # Full rebuild
    python rebuild_index.py --incremental           # Incremental only
    python rebuild_index.py --collection nvidia_10k # Custom collection
"""
import sys
import argparse

sys.path.insert(0, ".")
from src.ingestion.pipeline import IngestionPipeline

RAW_DIR = "data/raw"
DEFAULT_COLLECTION = "sec_filings"


def main():
    parser = argparse.ArgumentParser(description="FinSightAlpha Index Builder")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Target collection name")
    parser.add_argument("--incremental", action="store_true", help="Skip already-indexed documents")
    parser.add_argument("--dir", default=RAW_DIR, help="Input directory")
    args = parser.parse_args()

    print("=" * 60)
    print("FinSightAlpha Pipeline Index Build")
    print("=" * 60)
    print(f"  Collection: {args.collection}")
    print(f"  Mode: {'incremental' if args.incremental else 'full rebuild'}")
    print(f"  Input: {args.dir}")

    pipeline = IngestionPipeline()

    try:
        if not args.incremental:
            # Full rebuild: drop and recreate collection
            pipeline.collection_mgr.create_collection(args.collection, recreate=True)

        result = pipeline.ingest_directory_sync(
            dir_path=args.dir,
            collection=args.collection,
            force=not args.incremental,
        )

        # Print summary
        print(f"\n{'=' * 60}")
        print(f"BUILD COMPLETE")
        print(f"{'=' * 60}")
        print(f"  Files processed: {result.successful}")
        print(f"  Files skipped:   {result.skipped}")
        print(f"  Files failed:    {result.failed}")
        print(f"  Total chunks:    {result.total_chunks}")
        print(f"  Duration:        {result.total_duration:.1f}s")

        # Show collection stats
        stats = pipeline.collection_mgr.collection_stats(args.collection)
        print(f"\n  Collection '{args.collection}':")
        print(f"    Vectors: {stats.get('vectors_count', 0)}")
        print(f"    Documents: {stats.get('indexed_documents', 0)}")

    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
