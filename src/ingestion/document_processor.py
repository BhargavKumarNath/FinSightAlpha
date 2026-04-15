import os
import glob
from typing import Iterator, Dict, Any
from pathlib import Path
from unstructured.partition.html import partition_html
from unstructured.partition.auto import partition
from unstructured.partition.text import partition_text
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Table
import json

class DocumentProcessor:
    """
    Parses complex SEC HTML filings, converts tables to readable formats, and chunks documents by headers to preserve semantic context.
    """
    def __init__(self, max_characters: int = 2000, overlap: int = 200):
        self.max_characters = max_characters
        self.overlap = overlap
    
    def process_file(self, file_path: str):
        print(f"Parsing Document: {file_path}")

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        # Split into spaCy-safe chunks
        max_chars = 800_000
        text_blocks = [
            text[i:i + max_chars]
            for i in range(0, len(text), max_chars)
        ]

        for block in text_blocks:
            elements = partition_text(text=block)

            chunks = chunk_by_title(
                elements,
                max_characters=self.max_characters,
                new_after_n_chars=self.max_characters - self.overlap,
                combine_text_under_n_chars=500
            )

            for chunk in chunks:
                yield {
                    "page_content": chunk.text,
                    "metadata": {
                        "source": str(file_path),
                        "element_type": type(chunk).__name__,
                    }
                }
        
    def process_directory(self, input_dir: str, output_dir: str) -> None:
        """
        Processes all downloaded SEC filings and saves chunks to JSON lines
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Find all primary document files downlaoded by sec edger downloader
        search_pattern = os.path.join(input_dir, "sec-edgar-filings", "**", "*.*")
        files = glob.glob(search_pattern, recursive=True)

        # Keep only valid filing file types
        files = [
            f for f in files
            if f.lower().endswith((".html", ".htm", ".txt"))
            and os.path.isfile(f)
        ]

        if not files:
            print("No downloaded files found. Run the scraper first.")
            return 

        for file_path in files:
            path_parts = Path(file_path).parts
            ticker, form_type, accession_number = path_parts[-4], path_parts[-3], path_parts[-2]
            output_filename = f"{ticker}_{form_type}_{accession_number}_chunks.jsonl"
            output_filepath = os.path.join(output_dir, output_filename)
            
            print(f"Chunking {ticker} {form_type} into {output_filename}...")
            
            chunk_count = 0
            with open(output_filepath, 'w', encoding='utf-8') as out_f:
                for chunk_data in self.process_file(file_path):
                    out_f.write(json.dumps(chunk_data) + "\n")
                    chunk_count += 1
                    
            print(f"Successfully saved {chunk_count} chunks to {output_filepath}")

if __name__ == "__main__":
    processor = DocumentProcessor()
    processor.process_directory(input_dir="data/raw", output_dir="data/processed")
