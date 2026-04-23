"""Tests for POST /enrich (Phase 1.9 stub)."""
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


def test_enrich_shape(tmp_path, monkeypatch):
    from accord_ai.llm.fake_engine import FakeEngine as _FE
    monkeypatch.setenv("DB_PATH", str(tmp_path / "api.db"))
    monkeypatch.setenv("KNOWLEDGE_DB_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("ACCORD_AUTH_DISABLED", "true")
    from accord_ai.config import Settings
    from accord_ai.app import build_intake_app
    from accord_ai.api import build_fastapi_app
    engine = _FE(["Hello, let's get started."])
    settings = Settings()
    intake = build_intake_app(settings, engine=engine, refiner_engine=_FE())
    app = build_fastapi_app(settings, intake=intake)
    with TestClient(app) as client:
        sid = client.post("/start-session").json()["submission_id"]
        r = client.post("/enrich", json={"session_id": sid})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["validators_run"] == 0
    assert body["results"] == []
    assert body["cached"] is False


def test_enrich_unknown_session_returns_404(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        r = client.post("/enrich", json={"session_id": "no-such-session"})
    assert r.status_code == 404
