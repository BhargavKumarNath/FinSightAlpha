"""
Document Processor — Production-Grade Multi-Format Ingestion.

Backward-compatible facade over the new parser framework.
Maintains the same public API (process_file, process_directory)
while delegating to the pluggable parser registry and semantic chunker.

For new code, prefer using IngestionPipeline directly.
"""

import os
import json
import logging
from typing import Iterator, Dict, Any, List
from pathlib import Path

from src.ingestion.parsers.registry import ParserRegistry
from src.ingestion.chunking import SemanticChunker

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Production-grade document processor with multi-format support.

    Designed for plug-and-play ingestion: pass any directory of documents
    and get clean, semantically chunked text ready for embedding.

    Now delegates to ParserRegistry + SemanticChunker for extensibility.
    """

    def __init__(
        self,
        max_chunk_chars: int = 1500,
        overlap_chars: int = 200,
        min_chunk_chars: int = 80,
        strip_xbrl: bool = True,
    ):
        self.parser_registry = ParserRegistry.default()
        self.chunker = SemanticChunker(
            max_chunk_chars=max_chunk_chars,
            overlap_chars=overlap_chars,
            min_chunk_chars=min_chunk_chars,
        )

    # PUBLIC API

    def process_file(self, file_path: str) -> Iterator[Dict[str, Any]]:
        """
        Process a single file into clean, chunked documents.

        Auto-detects file format via the parser registry.
        Yields dicts with 'page_content' and 'metadata' keys.
        """
        file_path = str(file_path)
        print(f"Processing: {file_path}")

        try:
            parsed = self.parser_registry.parse(file_path)
        except ValueError as e:
            print(f"  [WARN] {e}")
            return
        except Exception as e:
            print(f"  [ERROR] Failed to parse {file_path}: {e}")
            return

        chunks = self.chunker.chunk(parsed)
        print(f"  Extracted {len(chunks)} quality chunks from {Path(file_path).name}")

        for chunk in chunks:
            yield chunk.to_jsonl_dict()

    def process_directory(self, input_dir: str, output_dir: str) -> None:
        """
        Processes all supported files in a directory tree and saves
        chunks as JSONL files.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        files = self._discover_files(input_dir)

        if not files:
            print(f"No supported files found in {input_dir}")
            return

        print(f"Found {len(files)} files to process")

        for file_path in files:
            rel_path = Path(file_path).relative_to(input_dir)
            safe_name = str(rel_path).replace(os.sep, "_").replace("/", "_")
            output_filename = f"{Path(safe_name).stem}_chunks.jsonl"
            output_filepath = os.path.join(output_dir, output_filename)

            print(f"\nChunking: {rel_path} -> {output_filename}")

            chunk_count = 0
            with open(output_filepath, "w", encoding="utf-8") as out_f:
                for chunk_data in self.process_file(file_path):
                    out_f.write(json.dumps(chunk_data, ensure_ascii=False) + "\n")
                    chunk_count += 1

            if chunk_count > 0:
                print(f"  Saved {chunk_count} chunks to {output_filename}")
            else:
                os.remove(output_filepath)
                print(f"  [SKIP] No quality chunks produced")

    def _discover_files(self, input_dir: str) -> List[str]:
        """Discover all parseable files in a directory tree."""
        files = []
        for root, _, filenames in os.walk(input_dir):
            for fname in filenames:
                fpath = os.path.join(root, fname)
                if self.parser_registry.can_handle(fpath):
                    files.append(fpath)
        return sorted(files)


if __name__ == "__main__":
    processor = DocumentProcessor()
    processor.process_directory(input_dir="data/raw", output_dir="data/processed")
