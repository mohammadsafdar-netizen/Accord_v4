"""Test doubles for library consumers.

These Fakes are production code (used by accord_ai's own test suite) but
curated here as a public API for downstream users who want to unit-test
code that depends on accord_ai without standing up real LLM / vector-store
infrastructure.

    from accord_ai.testing import FakeEngine, FakeEmbedder, FakeVectorStore
"""
from accord_ai.knowledge.fake_embedder import FakeEmbedder
from accord_ai.knowledge.fake_vector_store import FakeVectorStore
from accord_ai.llm.fake_engine import FakeEngine

__all__ = ["FakeEmbedder", "FakeEngine", "FakeVectorStore"]
