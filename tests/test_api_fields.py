"""Tests for GET /fields/{session_id} (Phase 1.10)."""
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
    engine = FakeEngine(["Hello, let's start."])
    intake = build_intake_app(settings, engine=engine, refiner_engine=FakeEngine())
    return build_fastapi_app(settings, intake=intake)


def test_fields_returns_submission_shape(tmp_path, monkeypatch):
    """GET /fields/{id} returns a dict that contains known CustomerSubmission fields."""
    app = _make_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        sid = client.post("/start-session").json()["submission_id"]
        r = client.get(f"/fields/{sid}")
    assert r.status_code == 200
    body = r.json()
    # CustomerSubmission has at least these keys
    for key in ("business_name", "contacts", "policy_dates"):
        assert key in body, f"expected key {key!r} in /fields response"


def test_fields_unknown_session_returns_404(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        r = client.get("/fields/no-such-session")
    assert r.status_code == 404
