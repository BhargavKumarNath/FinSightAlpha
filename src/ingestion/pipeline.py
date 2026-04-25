"""
Async Ingestion Pipeline — Production-Grade Document Processing.

Orchestrates the full ingestion flow:
  Parse → Chunk → Embed → Index

Features:
  - Async file processing with configurable concurrency
  - Incremental indexing (skips already-indexed documents)
  - Multi-collection routing
  - Progress tracking with callbacks
  - Error isolation (failed files don't block the batch)
  - BM25 index rebuild after ingestion
"""

import asyncio
import json
import os
import pickle
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from src.ingestion.parsers.registry import ParserRegistry
from src.ingestion.parsers.base import ParsedDocument
from src.ingestion.chunking import DocumentChunk, SemanticChunker
from src.retrieval.collection_manager import (
    CollectionManager,
    DocumentRegistry,
    IndexedDocument,
)

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Result of a single document ingestion."""
    source_path: str
    success: bool
    chunks_indexed: int = 0
    skipped: bool = False
    error: Optional[str] = None
    duration_seconds: float = 0.0


@dataclass
class BatchResult:
    """Aggregate result of a batch ingestion."""
    total_files: int = 0
    successful: int = 0
    skipped: int = 0
    failed: int = 0
    total_chunks: int = 0
    total_duration: float = 0.0
    results: List[IngestionResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        processed = self.successful + self.failed
        return self.successful / processed if processed > 0 else 0.0


class IngestionPipeline:
    """
    Production-grade async ingestion pipeline.

    Coordinates parsing, chunking, embedding, and indexing with
    incremental update support and error isolation.

    Usage:
        pipeline = IngestionPipeline()

        # Ingest a single file
        result = await pipeline.ingest_file("report.pdf", collection="sec_filings")

        # Ingest a directory
        result = await pipeline.ingest_directory("data/raw", collection="sec_filings")

        # Sync convenience methods
        result = pipeline.ingest_file_sync("report.pdf", collection="sec_filings")
    """

    def __init__(
        self,
        collection_manager: Optional[CollectionManager] = None,
        parser_registry: Optional[ParserRegistry] = None,
        chunker: Optional[SemanticChunker] = None,
        embedding_model=None,
        qdrant_path: str = "data/qdrant_db",
        bm25_dir: str = "data",
        max_workers: int = 4,
        batch_size: int = 100,
    ):
        self.collection_mgr = collection_manager or CollectionManager(qdrant_path=qdrant_path)
        self.parser_registry = parser_registry or ParserRegistry.default()
        self.chunker = chunker or SemanticChunker()
        self.bm25_dir = bm25_dir
        self.max_workers = max_workers
        self.batch_size = batch_size

        # Embedding model (lazy loaded)
        self._embedding_model = embedding_model
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
            print(f"  [Pipeline] Loaded embedding model on {device.upper()}")
        return self._embedding_model

    # Public API

    async def ingest_file(
        self,
        file_path: str,
        collection: str = "sec_filings",
        force: bool = False,
        progress_callback: Optional[Callable] = None,
    ) -> IngestionResult:
        """
        Ingest a single file into a collection.

        Args:
            file_path: Path to the document file.
            collection: Target Qdrant collection name.
            force: If True, re-index even if already indexed.
            progress_callback: Optional callback(stage, detail) for progress.
        """
        start = time.time()

        try:
            # Stage 1: Parse
            if progress_callback:
                progress_callback("parse", file_path)

            parsed = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                self.parser_registry.parse,
                file_path,
            )

            # Check incremental: skip if already indexed
            if not force and self.collection_mgr.registry.is_indexed(
                parsed.content_hash(), collection
            ):
                return IngestionResult(
                    source_path=file_path, success=True, skipped=True,
                    duration_seconds=time.time() - start,
                )

            # Stage 2: Chunk
            if progress_callback:
                progress_callback("chunk", f"{parsed.char_count} chars")

            chunks = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                self.chunker.chunk,
                parsed,
            )

            if not chunks:
                return IngestionResult(
                    source_path=file_path, success=True, chunks_indexed=0,
                    duration_seconds=time.time() - start,
                )

            # Stage 3: Embed + Index
            if progress_callback:
                progress_callback("index", f"{len(chunks)} chunks")

            # Ensure collection exists
            self.collection_mgr.create_collection(collection)

            point_ids = await self._embed_and_index(chunks, collection)

            # Stage 4: Register in document registry
            doc_id = CollectionManager.generate_doc_id(file_path)
            self.collection_mgr.registry.register(
                IndexedDocument(
                    doc_id=doc_id,
                    content_hash=parsed.content_hash(),
                    source_path=file_path,
                    source_name=parsed.metadata.source_name,
                    chunk_count=len(chunks),
                    indexed_at=time.time(),
                    collection=collection,
                    point_ids=point_ids,
                    metadata={
                        "title": parsed.metadata.title,
                        "file_type": parsed.metadata.file_type,
                        "page_count": parsed.metadata.page_count,
                    },
                )
            )

            return IngestionResult(
                source_path=file_path, success=True,
                chunks_indexed=len(chunks),
                duration_seconds=time.time() - start,
            )

        except Exception as e:
            logger.error(f"Failed to ingest {file_path}: {e}")
            return IngestionResult(
                source_path=file_path, success=False,
                error=str(e),
                duration_seconds=time.time() - start,
            )

    async def ingest_directory(
        self,
        dir_path: str,
        collection: str = "sec_filings",
        recursive: bool = True,
        force: bool = False,
    ) -> BatchResult:
        """
        Ingest all supported files from a directory.

        Args:
            dir_path: Path to directory containing documents.
            collection: Target Qdrant collection name.
            recursive: Whether to scan subdirectories.
            force: If True, re-index even if already indexed.
        """
        # Discover files
        files = self._discover_files(dir_path, recursive)

        if not files:
            print(f"  No supported files found in {dir_path}")
            return BatchResult()

        print(f"\n  [Pipeline] Found {len(files)} files to process")
        start = time.time()

        # Process files with controlled concurrency
        batch_result = BatchResult(total_files=len(files))

        for file_path in files:
            result = await self.ingest_file(file_path, collection, force)
            batch_result.results.append(result)

            if result.skipped:
                batch_result.skipped += 1
                print(f"  [SKIP] {Path(file_path).name} (already indexed)")
            elif result.success:
                batch_result.successful += 1
                batch_result.total_chunks += result.chunks_indexed
                print(f"  [OK]   {Path(file_path).name} -> {result.chunks_indexed} chunks")
            else:
                batch_result.failed += 1
                print(f"  [FAIL] {Path(file_path).name}: {result.error}")

        batch_result.total_duration = time.time() - start

        # Rebuild BM25 index for the collection
        await self._rebuild_bm25(collection)

        print(f"\n  [Pipeline] Complete: {batch_result.successful} indexed, "
              f"{batch_result.skipped} skipped, {batch_result.failed} failed "
              f"({batch_result.total_chunks} total chunks in {batch_result.total_duration:.1f}s)")

        return batch_result

    # Sync Convenience Methods

    def ingest_file_sync(self, file_path: str, collection: str = "sec_filings", force: bool = False) -> IngestionResult:
        """Synchronous wrapper for ingest_file."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.ingest_file(file_path, collection, force))
        finally:
            loop.close()

    def ingest_directory_sync(self, dir_path: str, collection: str = "sec_filings", **kwargs) -> BatchResult:
        """Synchronous wrapper for ingest_directory."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.ingest_directory(dir_path, collection, **kwargs))
        finally:
            loop.close()

    # Internal Methods

    async def _embed_and_index(
        self, chunks: List[DocumentChunk], collection: str
    ) -> List[int]:
        """Embed chunks and upsert into Qdrant."""
        from qdrant_client.models import PointStruct

        texts = [c.text for c in chunks]
        metadatas = [c.metadata.to_dict() for c in chunks]

        # Get current max point ID in collection
        try:
            info = self.collection_mgr.client.get_collection(collection)
            base_id = info.points_count or 0
        except Exception:
            base_id = 0

        point_ids = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            batch_metas = metadatas[i : i + self.batch_size]

            # Embed (CPU-bound, run in executor)
            embeddings = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                lambda t=batch_texts: self.embedding_model.encode(t, batch_size=self.batch_size, show_progress_bar=False),
            )

            points = []
            for j in range(len(batch_texts)):
                pid = base_id + i + j
                point_ids.append(pid)
                points.append(
                    PointStruct(
                        id=pid,
                        vector=embeddings[j].tolist(),
                        payload={
                            "text": batch_texts[j],
                            "metadata": batch_metas[j],
                        },
                    )
                )

            self.collection_mgr.client.upsert(
                collection_name=collection, points=points
            )

        return point_ids

    async def _rebuild_bm25(self, collection: str) -> None:
        """Rebuild BM25 index for a collection from Qdrant points."""
        print(f"  [Pipeline] Rebuilding BM25 index for '{collection}'...")

        from rank_bm25 import BM25Okapi

        # Scroll all points from Qdrant
        all_points = []
        offset = None
        while True:
            results, offset = self.collection_mgr.client.scroll(
                collection_name=collection, limit=500, offset=offset,
                with_payload=True, with_vectors=False,
            )
            all_points.extend(results)
            if offset is None:
                break

        if not all_points:
            return

        tokenized_corpus = []
        corpus_metadata = []

        for point in all_points:
            text = point.payload.get("text", "")
            meta = point.payload.get("metadata", {})
            tokenized_corpus.append(text.lower().split())
            corpus_metadata.append({"text": text, "metadata": meta})

        bm25 = BM25Okapi(tokenized_corpus)

        bm25_path = os.path.join(self.bm25_dir, f"bm25_{collection}.pkl")
        with open(bm25_path, "wb") as f:
            pickle.dump((bm25, corpus_metadata), f)

        # Also save as default if collection is sec_filings
        if collection == "sec_filings":
            default_path = os.path.join(self.bm25_dir, "bm25_index.pkl")
            with open(default_path, "wb") as f:
                pickle.dump((bm25, corpus_metadata), f)

        print(f"  [Pipeline] BM25 index saved ({len(all_points)} documents)")

    def _discover_files(self, dir_path: str, recursive: bool = True) -> List[str]:
        """Discover all parseable files in a directory."""
        files = []
        if recursive:
            for root, _, filenames in os.walk(dir_path):
                for fname in filenames:
                    fpath = os.path.join(root, fname)
                    if self.parser_registry.can_handle(fpath):
                        files.append(fpath)
        else:
            for fname in os.listdir(dir_path):
                fpath = os.path.join(dir_path, fname)
                if os.path.isfile(fpath) and self.parser_registry.can_handle(fpath):
                    files.append(fpath)
        return sorted(files)

    def close(self):
        """Clean shutdown."""
        self._executor.shutdown(wait=False)
        self.collection_mgr.close()
