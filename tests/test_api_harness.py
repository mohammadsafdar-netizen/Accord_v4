"""Tests for the /harness family of endpoints (Phase 1.10)."""
import pytest
from fastapi.testclient import TestClient

from accord_ai.api import build_fastapi_app
from accord_ai.app import build_intake_app
from accord_ai.config import Settings
from accord_ai.llm.fake_engine import FakeEngine


def _make_app(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "api.db"))
    monkeypatch.setenv("KNOWLEDGE_DB_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("ACCORD_AUTH_DISABLED", "true")
    settings = Settings()
    intake = build_intake_app(settings, engine=FakeEngine(), refiner_engine=FakeEngine())
    return build_fastapi_app(settings, intake=intake)


def test_harness_returns_declarative_version(tmp_path, monkeypatch):
    """GET /harness returns declarative ruleset shape with critical fields."""
    app = _make_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        r = client.get("/harness")
    assert r.status_code == 200
    body = r.json()
    assert body["version"] == "1.0-declarative"
    assert "commercial_auto" in body["critical_fields_per_lob"]
    assert isinstance(body["critical_fields_per_lob"]["commercial_auto"], list)
    assert len(body["critical_fields_per_lob"]["commercial_auto"]) > 0
    assert "negation" in body["active_rules"]
    assert isinstance(body["refiner_harness_enabled"], bool)


def test_harness_audit_returns_stats_shape(tmp_path, monkeypatch):
    """GET /harness/audit returns the expected shape (zeroed stub for now)."""
    app = _make_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        r = client.get("/harness/audit")
    assert r.status_code == 200
    body = r.json()
    assert "refinement_count" in body
    assert "most_failed_paths" in body
    assert isinstance(body["most_failed_paths"], list)


def test_harness_history_stub(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        r = client.get("/harness/history")
    assert r.status_code == 200
    body = r.json()
    assert "versions" in body
    assert "Phase D" in body["note"]


def test_harness_rollback_returns_501(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        r = client.post("/harness/rollback")
    assert r.status_code == 501
    assert "Phase D" in r.json()["detail"]


def test_harness_provenance_stub(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        r = client.get("/harness/provenance")
    assert r.status_code == 200
    assert r.json()["entries"] == []


def test_harness_review_queue_stub(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        r = client.get("/harness/review-queue")
    assert r.status_code == 200
    assert r.json()["queue"] == []
