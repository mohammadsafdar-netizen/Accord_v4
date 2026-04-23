"""VectorStore — Protocol for a dense vector index with add + query.

Real implementations (ChromaVectorStore etc.) land when Phase 7 has a
concrete retrieval call site. Phase 6 ships Protocol + Hit result type +
DimensionMismatchError for wiring validation.

Dimension enforcement rationale: a mismatched embedder/store wiring (e.g.
384d MiniLM output queried against a 768d-trained index) yields silent
garbage at the provider layer — results are technically ranked but
semantically meaningless. Raising at the Protocol boundary turns the bug
into a call-site failure instead of a retrieval-quality regression.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Protocol


class DimensionMismatchError(ValueError):
    """Raised when an add/query embedding doesn't match the store's dimension."""


@dataclass(frozen=True)
class Hit:
    """One result from a VectorStore query.

    `distance`: lower = more similar. Metric (cosine, L2, dot, etc.) is
    provider-dependent; callers treat it as "smaller is better" but do not
    assume a fixed scale or upper bound.

    `metadata`: read-only contract. The dataclass is frozen (reassigning
    `hit.metadata = {...}` raises) but Python does not deeply freeze the
    underlying Mapping — `hit.metadata["k"] = v` would technically succeed.
    Callers MUST NOT mutate the metadata; copy first if you need to modify.
    """
    doc_id: str
    document: str
    metadata: Mapping[str, Any]
    distance: float


class VectorStore(Protocol):
    """Dense vector index."""

    dimension: int

    async def add(
        self,
        *,
        doc_id: str,
        embedding: List[float],
        document: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Insert or overwrite a document. Raises DimensionMismatchError on
        embedding-dimension mismatch."""
        ...

    async def query(
        self,
        *,
        embedding: List[float],
        k: int = 5,
    ) -> List[Hit]:
        """Return up to k hits sorted by distance ASC. Empty list if no
        matches. Raises DimensionMismatchError on embedding-dimension
        mismatch."""
        ...
