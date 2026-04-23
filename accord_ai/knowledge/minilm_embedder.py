"""MiniLMEmbedder — local sentence-transformers wrapper.

Default: all-MiniLM-L6-v2 (384d, ~80MB, MIT, CPU). Loads lazily on first
embed call so `import` stays under 100ms. sentence-transformers' encode()
is CPU-bound; we wrap in asyncio.to_thread so concurrent extraction turns
don't block the event loop.

Thread-safe lazy load via double-checked locking — safe under asyncio +
threadpool executors.
"""
from __future__ import annotations

import asyncio
import threading
from typing import List, Optional

from sentence_transformers import SentenceTransformer


class MiniLMEmbedder:
    """sentence-transformers-backed Embedder. Lazy-loaded, thread-safe."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimension: int = 384,
    ) -> None:
        self._model_name = model_name
        self.dimension = dimension
        self._model: Optional[SentenceTransformer] = None
        self._load_lock = threading.Lock()

    def _ensure_loaded(self) -> SentenceTransformer:
        if self._model is None:
            with self._load_lock:
                if self._model is None:   # double-check inside lock
                    model = SentenceTransformer(self._model_name)
                    actual = model.get_sentence_embedding_dimension()
                    if actual != self.dimension:
                        raise ValueError(
                            f"model {self._model_name!r} produces {actual}d vectors, "
                            f"but embedder was constructed with dimension={self.dimension}"
                        )
                    self._model = model
        return self._model

    async def embed(self, text: str) -> List[float]:
        vecs = await asyncio.to_thread(self._encode_sync, [text])
        return vecs[0]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._encode_sync, texts)

    def _encode_sync(self, texts: List[str]) -> List[List[float]]:
        model = self._ensure_loaded()
        vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return vectors.tolist()
