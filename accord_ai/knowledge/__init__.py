"""Knowledge retrieval — Embedder + VectorStore + Retriever composition.

Reconstructed after the directory's __init__.py was accidentally truncated
during the Tier 1 v3-asset copy (P10.S.4/.5 boundary). The shape matches
what the existing tests (test_retriever, test_build_retriever,
test_knowledge_integration) already import against.
"""
from __future__ import annotations

from typing import Optional

from accord_ai.knowledge.chroma_vector_store import ChromaVectorStore
from accord_ai.knowledge.embedder import Embedder
from accord_ai.knowledge.fake_embedder import FakeEmbedder
from accord_ai.knowledge.fake_vector_store import FakeVectorStore
from accord_ai.knowledge.minilm_embedder import MiniLMEmbedder
from accord_ai.knowledge.retriever import Retriever
from accord_ai.knowledge.vector_store import (
    DimensionMismatchError,
    Hit,
    VectorStore,
)


def build_retriever(
    settings,
    *,
    collection_name: Optional[str] = None,
) -> Retriever:
    """Factory: MiniLMEmbedder + ChromaVectorStore wired via Retriever.

    `collection_name` defaults to settings.knowledge_collection_default — pass
    a per-tenant/per-corpus name to keep indices isolated at the same db path.
    """
    name = collection_name or settings.knowledge_collection_default
    embedder = MiniLMEmbedder(
        model_name=settings.knowledge_embedding_model,
        dimension=settings.knowledge_embedding_dimension,
    )
    store = ChromaVectorStore(
        path=settings.knowledge_db_path,
        collection_name=name,
        dimension=settings.knowledge_embedding_dimension,
    )
    return Retriever(embedder=embedder, store=store)


__all__ = [
    "ChromaVectorStore",
    "DimensionMismatchError",
    "Embedder",
    "FakeEmbedder",
    "FakeVectorStore",
    "Hit",
    "MiniLMEmbedder",
    "Retriever",
    "VectorStore",
    "build_retriever",
]
