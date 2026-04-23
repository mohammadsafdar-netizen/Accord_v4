"""Tests for POST /correction (Phase 2.2 — SQLite-backed, JSONL replaced)."""
from __future__ import annotations

import sqlite3

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


def _correction_payload(session_id="sess-1"):
    return {
        "session_id": session_id,
        "turn": 3,
        "field_path": "business_name",
        "wrong_value": "Acme Inc",
        "correct_value": "Acme LLC",
        "explanation": "Legal entity type changed",
    }


def test_correction_persists_to_sqlite(tmp_path, monkeypatch):
    """POST /correction returns {captured: true, id: <uuid>} and writes to SQLite."""
    app = _make_app(tmp_path, monkeypatch)
    db_path = str(tmp_path / "api.db")
    with TestClient(app) as client:
        r = client.post(
            "/correction",
            json=_correction_payload(),
            headers={"x-tenant-slug": "acme"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["captured"] is True
    assert len(body["id"]) == 36  # UUID

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM corrections WHERE id = ?", (body["id"],)).fetchone()
    conn.close()
    assert row is not None
    assert row["field_path"] == "business_name"
    assert row["status"] == "pending"


def test_correction_pii_redacted_at_collector_boundary(tmp_path, monkeypatch):
    """PII in values is redacted before reaching the DB row."""
    app = _make_app(tmp_path, monkeypatch)
    db_path = str(tmp_path / "api.db")
    ssn_payload = {
        "session_id": "sess-1",
        "turn": 1,
        "field_path": "notes",
        "wrong_value": "SSN 123-45-6789",
        "correct_value": "SSN 987-65-4321",
    }
    with TestClient(app) as client:
        r = client.post("/correction", json=ssn_payload)
    assert r.status_code == 200

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT wrong_value_json, correct_value_json FROM corrections").fetchone()
    conn.close()
    assert "123-45-6789" not in (row["wrong_value_json"] or "")
    assert "987-65-4321" not in (row["correct_value_json"] or "")


def test_correction_rows_tagged_with_tenant(tmp_path, monkeypatch):
    """Each /correction call persists a row tagged with the effective tenant."""
    app = _make_app(tmp_path, monkeypatch)
    db_path = str(tmp_path / "api.db")
    with TestClient(app) as client:
        r1 = client.post("/correction", json=_correction_payload("s1"), headers={"x-tenant-slug": "t1"})
        r2 = client.post("/correction", json=_correction_payload("s2"), headers={"x-tenant-slug": "t2"})
    assert r1.status_code == 200
    assert r2.status_code == 200

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT tenant, session_id FROM corrections ORDER BY created_at").fetchall()
    conn.close()
    assert len(rows) == 2
    # Both rows are in the DB; auth disabled → tenant is "default" for both
    assert all(r["tenant"] is not None for r in rows)
