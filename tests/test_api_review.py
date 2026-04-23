"""API tests for GET /review/{session_id} (Phase 1.6.E) — 3 tests."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from accord_ai.api import build_fastapi_app
from accord_ai.app import build_intake_app
from accord_ai.config import Settings
from accord_ai.llm.fake_engine import FakeEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings(tmp_path) -> Settings:
    return Settings(
        db_path=str(tmp_path / "accord.db"),
        filled_pdf_dir=str(tmp_path / "filled"),
        accord_auth_disabled=True,
        harness_max_refines=0,
    )


def _make_app(tmp_path):
    settings = _settings(tmp_path)
    engine = FakeEngine(["Hello!"])
    intake = build_intake_app(settings, engine=engine, refiner_engine=FakeEngine())
    app = build_fastapi_app(settings, intake=intake)
    return app, intake


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_get_review_returns_payload(tmp_path):
    """GET /review/{session_id} returns a well-formed ReviewPayload."""
    app, _ = _make_app(tmp_path)
    with TestClient(app) as client:
        sid = client.post("/start-session").json()["submission_id"]
        r = client.get(f"/review/{sid}")

    assert r.status_code == 200
    data = r.json()
    # Required ReviewPayload fields
    assert "session_id" in data
    assert "ready_to_finalize" in data
    assert "summary" in data
    assert "conflicts" in data
    assert "prefills" in data
    assert "compliance" in data
    assert "warnings" in data
    assert "info" in data
    assert data["session_id"] == sid
    assert isinstance(data["ready_to_finalize"], bool)
    assert data["prefills"] == []


def test_review_cached_on_repeat_call_same_submission(tmp_path):
    """Second GET /review call with unchanged submission returns same data."""
    app, _ = _make_app(tmp_path)
    with TestClient(app) as client:
        sid = client.post("/start-session").json()["submission_id"]
        r1 = client.get(f"/review/{sid}")
        r2 = client.get(f"/review/{sid}")

    assert r1.status_code == 200
    assert r2.status_code == 200
    # Both responses should have identical ready_to_finalize (cache hit or same data)
    assert r1.json()["ready_to_finalize"] == r2.json()["ready_to_finalize"]
    assert r1.json()["session_id"] == r2.json()["session_id"]


def test_review_route_exists_in_openapi(tmp_path):
    """GET /review/{session_id} is registered in the OpenAPI spec."""
    app, _ = _make_app(tmp_path)
    with TestClient(app) as client:
        schema = client.get("/openapi.json").json()

    paths = schema.get("paths", {})
    review_path = "/review/{session_id}"
    assert review_path in paths, f"Route {review_path!r} not found in OpenAPI spec"
    assert "get" in paths[review_path]
