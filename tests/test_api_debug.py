"""Tests for GET /debug/session/{id} (Phase 1.10, admin-only, PII-redacted)."""
import json
import pytest
from fastapi.testclient import TestClient

from accord_ai.api import build_fastapi_app
from accord_ai.app import build_intake_app
from accord_ai.config import Settings
from accord_ai.llm.fake_engine import FakeEngine


def _make_app(tmp_path, monkeypatch, *, api_key=None, auth_disabled=True):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "api.db"))
    monkeypatch.setenv("KNOWLEDGE_DB_PATH", str(tmp_path / "chroma"))
    if auth_disabled:
        monkeypatch.setenv("ACCORD_AUTH_DISABLED", "true")
        monkeypatch.delenv("INTAKE_API_KEY", raising=False)
        monkeypatch.delenv("INTAKE_API_KEYS", raising=False)
    elif api_key:
        monkeypatch.setenv("INTAKE_API_KEY", api_key)
        monkeypatch.delenv("ACCORD_AUTH_DISABLED", raising=False)
        monkeypatch.delenv("INTAKE_API_KEYS", raising=False)
    settings = Settings()
    engine = FakeEngine(["Hello, let's start."])
    intake = build_intake_app(settings, engine=engine, refiner_engine=FakeEngine())
    return build_fastapi_app(settings, intake=intake)


def test_debug_session_full_dump_shape(tmp_path, monkeypatch):
    """Admin can fetch a full session dump with required fields."""
    app = _make_app(tmp_path, monkeypatch, auth_disabled=True)
    with TestClient(app) as client:
        sid = client.post("/start-session").json()["submission_id"]
        r = client.get(f"/debug/session/{sid}")
    assert r.status_code == 200
    body = r.json()
    assert "session" in body
    assert "submission" in body
    assert "turns" in body
    assert body["harness_version"] == "1.0-declarative"
    assert "last_refiner_stage" in body
    assert body["session"]["session_id"] == sid


def test_debug_session_tenant_key_returns_403(tmp_path, monkeypatch):
    """A per-tenant bound key must not access /debug — admin only."""
    admin_key = "admin-secret-key"
    tenant_key = "tenant-only-key"

    monkeypatch.setenv("DB_PATH", str(tmp_path / "api.db"))
    monkeypatch.setenv("KNOWLEDGE_DB_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("INTAKE_API_KEY", admin_key)
    monkeypatch.setenv("INTAKE_API_KEYS", json.dumps({tenant_key: "acme"}))
    monkeypatch.delenv("ACCORD_AUTH_DISABLED", raising=False)
    settings = Settings()
    engine = FakeEngine(["Hello."])
    intake = build_intake_app(settings, engine=engine, refiner_engine=FakeEngine())
    app = build_fastapi_app(settings, intake=intake)

    with TestClient(app) as client:
        # Create session with admin key
        sid = client.post(
            "/start-session",
            headers={"x-api-key": admin_key},
        ).json()["submission_id"]

        # Try to access debug with tenant key → 403
        r = client.get(
            f"/debug/session/{sid}",
            headers={"x-api-key": tenant_key, "x-tenant-slug": "acme"},
        )
    assert r.status_code == 403
    assert "admin" in r.json()["detail"]


def test_debug_session_unknown_session_returns_404(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch, auth_disabled=True)
    with TestClient(app) as client:
        r = client.get("/debug/session/no-such-session")
    assert r.status_code == 404
