"""ChromaVectorStore — chromadb.PersistentClient wrapper.

Implements VectorStore Protocol. Each instance is backed by one collection
in a local PersistentClient. The collection name is typically per-tenant
or per-corpus — pass distinct names to keep collections isolated at the
same db path.

chromadb operations are synchronous; wrapped via asyncio.to_thread.
"""
from __future__ import annotations

import asyncio
from typing import Any, List, Mapping, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from accord_ai.knowledge.vector_store import DimensionMismatchError, Hit

# Namespaced sentinel so a caller's legitimate single-char metadata keys
# (e.g. "_") don't collide with our placeholder for empty-metadata adds.
_EMPTY_META_SENTINEL = "__accord_ai_empty_metadata"


class ChromaVectorStore:
    """chromadb-backed VectorStore. Persistent; tenant-isolated by collection name."""

    def __init__(
        self,
        *,
        path: str,
        collection_name: str,
        dimension: int = 384,
        client: Optional[chromadb.api.ClientAPI] = None,  # injection for tests
    ) -> None:
        self.dimension = dimension
        self._collection_name = collection_name
        self._client = client or chromadb.PersistentClient(
            path=path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        # get_or_create_collection is idempotent — safe on repeat opens
        self._collection = self._client.get_or_create_collection(name=collection_name)

    async def add(
        self,
        *,
        doc_id: str,
        embedding: List[float],
        document: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if len(embedding) != self.dimension:
            raise DimensionMismatchError(
                f"add embedding dim {len(embedding)} != store dim {self.dimension}"
            )
        # chromadb >= 0.5 accepts empty metadata dict, but some ops dislike it;
        # use a minimal placeholder when the caller gave nothing, strip on read.
        metadatas = [dict(metadata)] if metadata else [{_EMPTY_META_SENTINEL: ""}]
        await asyncio.to_thread(
            self._collection.upsert,
            ids=[doc_id],
            embeddings=[embedding],
            documents=[document],
            metadatas=metadatas,
        )

    async def query(
        self,
        *,
        embedding: List[float],
        k: int = 5,
    ) -> List[Hit]:
        if len(embedding) != self.dimension:
            raise DimensionMismatchError(
                f"query embedding dim {len(embedding)} != store dim {self.dimension}"
            )
        result = await asyncio.to_thread(
            self._collection.query,
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0] or [{}] * len(ids)
        distances = result["distances"][0]

        return [
            Hit(
                doc_id=i,
                document=d,
                # Strip the "_" placeholder we wrote for metadata-less adds
                metadata={
                    k: v for k, v in (m or {}).items()
                    if k != _EMPTY_META_SENTINEL
                },
                distance=float(dist),
            )
            for i, d, m, dist in zip(ids, documents, metadatas, distances)
        ]
