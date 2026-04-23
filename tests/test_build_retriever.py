"""8.d — build_retriever factory."""
import os

import pytest

from accord_ai.config import Settings
from accord_ai.knowledge import (
    ChromaVectorStore,
    MiniLMEmbedder,
    Retriever,
    build_retriever,
)


def test_build_retriever_returns_minilm_plus_chroma(tmp_path, monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_DB_PATH", str(tmp_path / "chroma"))
    retriever = build_retriever(Settings())
    assert isinstance(retriever, Retriever)
    assert isinstance(retriever._embedder, MiniLMEmbedder)
    assert isinstance(retriever._store, ChromaVectorStore)


def test_build_retriever_respects_collection_name_override(tmp_path, monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_DB_PATH", str(tmp_path / "chroma"))
    retriever = build_retriever(Settings(), collection_name="acme")
    assert retriever._store._collection_name == "acme"


def test_build_retriever_uses_settings_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_DB_PATH", str(tmp_path / "chroma"))
    r = build_retriever(Settings())
    assert r._embedder.dimension == 384
    assert r._store.dimension == 384


def test_build_retriever_custom_dimension(tmp_path, monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_DB_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("KNOWLEDGE_EMBEDDING_DIMENSION", "768")
    r = build_retriever(Settings())
    assert r._embedder.dimension == 768
    assert r._store.dimension == 768
