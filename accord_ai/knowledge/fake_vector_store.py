"""FakeVectorStore — FIFO-queued canned result batches for tests.

Tests control retrieval results directly by pre-loading the queue. Each
query() call pops one batch. Add calls are tracked on .adds for assertion
but do NOT influence future query results — ranking is test-specified via
the queue, not computed from stored vectors.

Dimension is enforced: add/query with wrong-dim embedding raises
DimensionMismatchError. Catches wiring bugs at the Protocol boundary.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Mapping, Optional

from accord_ai.knowledge.vector_store import DimensionMismatchError, Hit


class FakeVectorStore:
    def __init__(
        self,
        results: Optional[List[List[Hit]]] = None,
        *,
        dimension: int = 384,
    ) -> None:
        self.dimension = dimension
        self._queue: Deque[List[Hit]] = deque(results or [])
        # Assertion hooks.
        # `adds` is a call LOG, not a current-state snapshot — duplicate
        # doc_ids appear multiple times even though real impls overwrite.
        self.adds: List[Dict[str, Any]] = []
        self.queries: List[List[float]] = []

    def enqueue(self, hits: List[Hit]) -> None:
        """Append another canned result batch for the next query() call."""
        self._queue.append(hits)

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
        self.adds.append({
            "doc_id": doc_id,
            "document": document,
            "metadata": dict(metadata or {}),
        })

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
        self.queries.append(list(embedding))
        if not self._queue:
            return []
        return self._queue.popleft()[:k]
