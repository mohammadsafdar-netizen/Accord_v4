"""FakeEmbedder — deterministic stdlib-only embedder for tests.

Same text -> same vector across runs (hash-seeded). Different texts ->
different vectors (hash collisions are vanishingly rare in practice).

IMPORTANT: Vectors are NOT semantically meaningful — do not compute cosine
similarity and expect results to reflect actual relatedness. Tests that
care about ranking should control result order via FakeVectorStore (8.b),
which returns pre-sorted canned result batches.
"""
from __future__ import annotations

import hashlib
import random
from typing import List


class FakeEmbedder:
    """Deterministic hash-seeded embedder. No network, no model, no deps."""

    def __init__(self, dimension: int = 384) -> None:
        # 384 matches all-MiniLM-L6-v2, the likely Phase-7 real impl
        self.dimension = dimension
        # Inputs seen by embed / embed_batch — for assertion in tests
        self.calls: List[str] = []

    async def embed(self, text: str) -> List[float]:
        self.calls.append(text)
        return self._vectorize(text)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        self.calls.extend(texts)
        return [self._vectorize(t) for t in texts]

    def _vectorize(self, text: str) -> List[float]:
        # Hash -> 32-bit seed -> stdlib Random -> dimension U(0,1) floats
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2 ** 32)
        rng = random.Random(seed)
        return [rng.random() for _ in range(self.dimension)]
