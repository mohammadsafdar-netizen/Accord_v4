"""Tests for DPO admin endpoints (Phase 2.3) — 4 tests."""
from __future__ import annotations

import json
import sqlite3
import uuid

import pytest
from fastapi.testclient import TestClient

from accord_ai.api import build_fastapi_app
from accord_ai.app import build_intake_app
from accord_ai.config import Settings
from accord_ai.llm.fake_engine import FakeEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app(tmp_path, monkeypatch, *, auth_disabled=True):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "api.db"))
    monkeypatch.setenv("KNOWLEDGE_DB_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("TRAINING_DATA_DIR", str(tmp_path / "training"))
    monkeypatch.setenv("DPO_THRESHOLD", "2")
    if auth_disabled:
        monkeypatch.setenv("ACCORD_AUTH_DISABLED", "true")
    settings = Settings()
    intake = build_intake_app(settings, engine=FakeEngine(), refiner_engine=FakeEngine())
    return build_fastapi_app(settings, intake=intake)


def _seed_corrections(db_path: str, tenant: str, n: int) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = OFF")
    for i in range(n):
        sid = f"seed-{i}"
        # Insert message so build_pairs can find user_text
        conn.execute(
            "INSERT INTO messages (message_id, session_id, created_at, role, content)"
            " VALUES (?, ?, datetime('now'), 'user', ?)",
            (uuid.uuid4().hex, sid, f"User message {i}"),
        )
        conn.execute(
            """
            INSERT INTO corrections
                (id, tenant, session_id, turn, field_path,
                 wrong_value_json, correct_value_json, correction_type, status, created_at)
            VALUES (?, ?, ?, 1, 'business_name', ?, ?, 'value_correction', 'pending', datetime('now'))
            """,
            (uuid.uuid4().hex, tenant, sid,
             json.dumps(f"Wrong {i}"), json.dumps(f"Correct {i}")),
        )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_admin_dpo_export_below_threshold_returns_400(tmp_path, monkeypatch):
    """POST /admin/dpo/export/{tenant} without force returns 400 when below threshold."""
    app = _make_app(tmp_path, monkeypatch)
    db_path = str(tmp_path / "api.db")
    _seed_corrections(db_path, "acme", 1)  # 1 correction, threshold=2

    with TestClient(app) as client:
        r = client.post("/admin/dpo/export/acme", json={})
    assert r.status_code == 400
    body = r.json()
    assert body["detail"]["error"] == "below_threshold"
    assert body["detail"]["pending"] == 1
    assert body["detail"]["threshold"] == 2


def test_admin_dpo_export_force_below_threshold_succeeds(tmp_path, monkeypatch):
    """POST /admin/dpo/export/{tenant} with force=true exports even below threshold."""
    app = _make_app(tmp_path, monkeypatch)
    db_path = str(tmp_path / "api.db")
    _seed_corrections(db_path, "acme", 1)  # 1 correction, threshold=2

    with TestClient(app) as client:
        r = client.post("/admin/dpo/export/acme", json={"force": True})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert body["tenant"] == "acme"
    assert body["path"] is not None
    assert body["eligible_for_training"] is True


def test_admin_dpo_status_returns_counts(tmp_path, monkeypatch):
    """GET /admin/dpo/status/{tenant} returns pending/graduated counts."""
    app = _make_app(tmp_path, monkeypatch)
    db_path = str(tmp_path / "api.db")
    _seed_corrections(db_path, "acme", 3)

    with TestClient(app) as client:
        # Export one batch first
        client.post("/admin/dpo/export/acme", json={"force": True})
        r = client.get("/admin/dpo/status/acme")
    assert r.status_code == 200
    body = r.json()
    assert body["tenant"] == "acme"
    assert body["graduated"] == 3
    assert body["pending"] == 0
    assert body["eligible_for_training"] is False


def test_admin_dpo_export_requires_admin_key(tmp_path, monkeypatch):
    """Tenant-bound key (non-admin) returns 403 on /admin endpoints."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "api.db"))
    monkeypatch.setenv("KNOWLEDGE_DB_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("TRAINING_DATA_DIR", str(tmp_path / "training"))
    monkeypatch.setenv("DPO_THRESHOLD", "2")
    monkeypatch.setenv("INTAKE_API_KEYS", json.dumps({"tenant-only-key": "acme"}))
    # No ACCORD_AUTH_DISABLED, no INTAKE_API_KEY → tenant key is non-admin.
    settings = Settings()
    intake = build_intake_app(settings, engine=FakeEngine(), refiner_engine=FakeEngine())
    app = build_fastapi_app(settings, intake=intake)

    with TestClient(app) as client:
        r = client.post(
            "/admin/dpo/export/acme",
            json={"force": True},
            headers={"Authorization": "Bearer tenant-only-key"},
        )
    assert r.status_code == 403
