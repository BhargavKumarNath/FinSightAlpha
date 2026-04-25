"""
Collection Manager — Multi-Collection Lifecycle Management for Qdrant.

Enables dataset isolation via separate Qdrant collections:
  - One collection per dataset/domain (e.g., 'nvidia_10k', 'apple_10q')
  - Document registry tracks what's indexed for incremental updates
  - Collection metadata (stats, doc count) for operational visibility
"""

import json
import os
import time
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


@dataclass
class IndexedDocument:
    """Record of a document that has been indexed."""
    doc_id: str
    content_hash: str
    source_path: str
    source_name: str
    chunk_count: int
    indexed_at: float
    collection: str
    point_ids: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentRegistry:
    """
    Persistent registry tracking which documents are indexed.

    Backed by a JSON file per collection, enabling incremental
    indexing (skip already-indexed docs) and selective removal.
    """

    def __init__(self, registry_dir: str = "data"):
        self._dir = Path(registry_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Dict[str, IndexedDocument]] = {}

    def _registry_path(self, collection: str) -> Path:
        return self._dir / f".registry_{collection}.json"

    def _load(self, collection: str) -> Dict[str, IndexedDocument]:
        if collection in self._cache:
            return self._cache[collection]

        path = self._registry_path(collection)
        if not path.exists():
            self._cache[collection] = {}
            return self._cache[collection]

        with open(path, "r") as f:
            raw = json.load(f)

        docs = {}
        for doc_id, data in raw.items():
            docs[doc_id] = IndexedDocument(**data)
        self._cache[collection] = docs
        return docs

    def _save(self, collection: str) -> None:
        docs = self._cache.get(collection, {})
        path = self._registry_path(collection)
        with open(path, "w") as f:
            json.dump({k: asdict(v) for k, v in docs.items()}, f, indent=2)

    def is_indexed(self, content_hash: str, collection: str) -> bool:
        """Check if a document with this content hash is already indexed."""
        docs = self._load(collection)
        return any(d.content_hash == content_hash for d in docs.values())

    def get_by_hash(self, content_hash: str, collection: str) -> Optional[IndexedDocument]:
        """Retrieve an indexed document record by content hash."""
        docs = self._load(collection)
        for d in docs.values():
            if d.content_hash == content_hash:
                return d
        return None

    def register(self, doc: IndexedDocument) -> None:
        """Register a newly indexed document."""
        docs = self._load(doc.collection)
        docs[doc.doc_id] = doc
        self._save(doc.collection)

    def unregister(self, doc_id: str, collection: str) -> Optional[IndexedDocument]:
        """Remove a document from the registry. Returns the removed record."""
        docs = self._load(collection)
        removed = docs.pop(doc_id, None)
        if removed:
            self._save(collection)
        return removed

    def list_documents(self, collection: str) -> List[IndexedDocument]:
        """List all indexed documents in a collection."""
        return list(self._load(collection).values())

    def document_count(self, collection: str) -> int:
        return len(self._load(collection))

    def clear(self, collection: str) -> None:
        """Clear registry for a collection."""
        self._cache[collection] = {}
        path = self._registry_path(collection)
        if path.exists():
            path.unlink()


class CollectionManager:
    """
    Manages Qdrant collections with operational metadata.

    Provides a clean API for:
      - Creating/deleting collections with proper vector configs
      - Listing collections with stats
      - Document registry integration for incremental indexing
    """

    def __init__(
        self,
        qdrant_path: str = "data/qdrant_db",
        vector_size: int = 384,
        registry_dir: str = "data",
    ):
        self.qdrant_path = qdrant_path
        self.vector_size = vector_size
        os.makedirs(qdrant_path, exist_ok=True)

        self.client = QdrantClient(path=qdrant_path)
        self.registry = DocumentRegistry(registry_dir=registry_dir)

    def create_collection(
        self,
        name: str,
        vector_size: Optional[int] = None,
        distance: Distance = Distance.COSINE,
        recreate: bool = False,
    ) -> bool:
        """
        Create a Qdrant collection.

        Args:
            name: Collection name (e.g., 'nvidia_10k_fy2026').
            vector_size: Embedding dimension (default: 384 for MiniLM).
            distance: Distance metric for similarity search.
            recreate: If True, drops existing collection first.

        Returns:
            True if created, False if already exists (and recreate=False).
        """
        size = vector_size or self.vector_size

        if self.client.collection_exists(name):
            if recreate:
                self.client.delete_collection(name)
                self.registry.clear(name)
                print(f"  Dropped existing collection: {name}")
            else:
                print(f"  Collection already exists: {name}")
                return False

        self.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=size, distance=distance),
        )
        print(f"  Created collection: {name} (dim={size}, dist={distance.value})")
        return True

    def delete_collection(self, name: str) -> bool:
        """Delete a collection and its registry."""
        if self.client.collection_exists(name):
            self.client.delete_collection(name)
            self.registry.clear(name)
            print(f"  Deleted collection: {name}")
            return True
        return False

    def list_collections(self) -> List[str]:
        """List all collection names."""
        collections = self.client.get_collections().collections
        return [c.name for c in collections]

    def collection_stats(self, name: str) -> Dict[str, Any]:
        """Get detailed stats for a collection."""
        if not self.client.collection_exists(name):
            return {"error": f"Collection '{name}' not found"}

        info = self.client.get_collection(name)
        docs = self.registry.list_documents(name)

        return {
            "name": name,
            "points_count": info.points_count,
            "indexed_documents": len(docs),
            "documents": [
                {
                    "doc_id": d.doc_id,
                    "source": d.source_name,
                    "chunks": d.chunk_count,
                    "indexed_at": time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(d.indexed_at)
                    ),
                }
                for d in docs
            ],
        }

    def all_stats(self) -> List[Dict[str, Any]]:
        """Get stats for all collections."""
        return [self.collection_stats(name) for name in self.list_collections()]

    @staticmethod
    def generate_doc_id(source_path: str) -> str:
        """Generate a deterministic document ID from source path."""
        return hashlib.md5(source_path.encode()).hexdigest()[:12]

    def close(self):
        """Clean shutdown."""
        self.client.close()
