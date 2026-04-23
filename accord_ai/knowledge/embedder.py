"""Embedder — turn text into fixed-dimension vectors.

Async Protocol to match Engine. Real implementations (Phase 7+) wrap
sentence-transformers (or any CPU-bound embedder) in asyncio.to_thread.
Phase 6 ships only the Protocol + FakeEmbedder for test scaffolding;
the real MiniLM adapter lands when Phase 7 has a concrete retrieval
call site to validate against.
"""
from __future__ import annotations

from typing import List, Protocol


class Embedder(Protocol):
    """Non-streaming text -> vector embedder."""

    dimension: int
    """Fixed output dimensionality. Set per-model, constant across calls."""

    async def embed(self, text: str) -> List[float]:
        """Embed a single text. Returns a `dimension`-length vector."""
        ...

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts in one call. Real impls batch more
        efficiently than a loop over embed()."""
        ...
