"""Retriever — composes Embedder + VectorStore into a text-in → Hits-out API.

Dimension safety: the constructor asserts embedder.dimension ==
store.dimension, raising DimensionMismatchError at wire-up time instead
of letting the mismatch surface as the first retrieval call's garbage
output or a runtime exception at call time.

Phase 7 interface sketch:
    TODO — 8.d fills this in with how extraction will call retrieve() and
    thread the returned Hits into the prompt. If sketching the interface
    feels contrived at that point, the Protocol isn't ready to freeze.
"""
from __future__ import annotations

from typing import Any, List, Mapping, Optional

from accord_ai.knowledge.embedder import Embedder
from accord_ai.knowledge.vector_store import (
    DimensionMismatchError,
    Hit,
    VectorStore,
)


class Retriever:
    """Thin composition of an Embedder and a VectorStore.

    Callers never touch the embedder or store directly — go through
    retrieve() / add_document() so dimension safety and a single call site
    are preserved.
    """

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        if embedder.dimension != store.dimension:
            raise DimensionMismatchError(
                f"embedder dim {embedder.dimension} != store dim {store.dimension}"
            )
        self._embedder = embedder
        self._store = store

    async def retrieve(self, query: str, *, k: int = 5) -> List[Hit]:
        """Embed `query` and return up to k hits sorted by distance ASC."""
        vec = await self._embedder.embed(query)
        return await self._store.query(embedding=vec, k=k)

    async def add_document(
        self,
        *,
        doc_id: str,
        document: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Embed the document text and insert it. Convenience for loaders —
        shares the dim-safety guarantee of the Retriever constructor.
        """
        vec = await self._embedder.embed(document)
        await self._store.add(
            doc_id=doc_id,
            embedding=vec,
            document=document,
            metadata=metadata,
        )
