"""Tests for GET /explain/{field_path} (Phase 1.9)."""
import pytest
from fastapi.testclient import TestClient

from accord_ai.api import build_fastapi_app
from accord_ai.app import build_intake_app
from accord_ai.config import Settings
from accord_ai.knowledge import FakeEmbedder, FakeVectorStore, Retriever
from accord_ai.knowledge.vector_store import Hit
from accord_ai.llm.fake_engine import FakeEngine


def _make_app(tmp_path, monkeypatch, *, retriever=None):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "api.db"))
    monkeypatch.setenv("KNOWLEDGE_DB_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("ACCORD_AUTH_DISABLED", "true")
    settings = Settings()
    intake = build_intake_app(settings, engine=FakeEngine(), refiner_engine=FakeEngine())
    app = build_fastapi_app(settings, intake=intake)

    if retriever is not None:
        # Patch the module-level cached helper to use the fake retriever.
        import accord_ai.api as api_mod

        async def _fake_retrieve(s, tenant, field):
            return await retriever.retrieve(field, k=5)

        monkeypatch.setattr(api_mod, "_explain_retrieve", _fake_retrieve)

    return app


def _fake_retriever_with_hits():
    DIM = 4
    embedder = FakeEmbedder(dimension=DIM)
    store = FakeVectorStore(dimension=DIM)
    store.enqueue([
        Hit(
            doc_id="doc-1",
            document="A fleet vehicle is a vehicle owned by a company.",
            metadata={"title": "Fleet Vehicle Definition", "explanation": "Explains fleet vehicles."},
            distance=0.1,
        )
    ])
    return Retriever(embedder=embedder, store=store)


def _empty_retriever():
    DIM = 4
    embedder = FakeEmbedder(dimension=DIM)
    store = FakeVectorStore(dimension=DIM)
    # No enqueued results — query() returns []
    return Retriever(embedder=embedder, store=store)


def test_explain_happy_path(tmp_path, monkeypatch):
    retriever = _fake_retriever_with_hits()
    app = _make_app(tmp_path, monkeypatch, retriever=retriever)
    with TestClient(app) as client:
        r = client.get("/explain/business_name")
    assert r.status_code == 200
    body = r.json()
    assert body["field"] == "business_name"
    assert isinstance(body["explanation"], str)
    assert len(body["explanation"]) > 0
    assert isinstance(body["sources"], list)


def test_explain_unknown_field_returns_404(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        r = client.get("/explain/totally_unknown_field_xyz")
    assert r.status_code == 404
    assert "unknown field" in r.json()["detail"]


def test_explain_no_hits_returns_empty_sources(tmp_path, monkeypatch):
    retriever = _empty_retriever()
    app = _make_app(tmp_path, monkeypatch, retriever=retriever)
    with TestClient(app) as client:
        r = client.get("/explain/business_name")
    assert r.status_code == 200
    body = r.json()
    assert body["sources"] == []
    assert "No knowledge-base entries" in body["explanation"]


def test_explain_tenant_scoped_via_header(tmp_path, monkeypatch):
    retriever = _fake_retriever_with_hits()
    app = _make_app(tmp_path, monkeypatch, retriever=retriever)
    with TestClient(app) as client:
        r = client.get(
            "/explain/business_name",
            headers={"x-tenant-slug": "acme-corp"},
        )
    assert r.status_code == 200
    assert r.json()["field"] == "business_name"
