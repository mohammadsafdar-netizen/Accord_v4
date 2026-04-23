"""Tests for POST /feedback (Phase 1.9)."""
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


def test_feedback_persists_to_sqlite(tmp_path, monkeypatch):
    """POST /feedback writes to SQLite feedback table (Phase 2.2 — JSONL replaced)."""
    import sqlite3

    app = _make_app(tmp_path, monkeypatch)
    db_path = str(tmp_path / "api.db")
    with TestClient(app) as client:
        r = client.post(
            "/feedback",
            json={"session_id": "sess-1", "turn": 2, "rating": 4, "comment": "Good question"},
            headers={"x-tenant-slug": "acme"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["captured"] is True
    assert len(body["id"]) == 36  # UUID

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM feedback WHERE id = ?", (body["id"],)).fetchone()
    conn.close()
    assert row is not None
    assert row["rating"] == 4
    assert row["session_id"] == "sess-1"


def test_feedback_rating_out_of_range_returns_422(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        r = client.post(
            "/feedback",
            json={"session_id": "sess-1", "turn": 1, "rating": 6},
        )
    assert r.status_code == 422

    with TestClient(app) as client:
        r = client.post(
            "/feedback",
            json={"session_id": "sess-1", "turn": 1, "rating": 0},
        )
    assert r.status_code == 422
