"""Tests for AnswerRequest payload limits (P10.0.f.1).

Caps /answer's message field at max_length=50_000 to match CompleteRequest
and prevent unbounded-payload DoS. Pydantic validation returns 422 on
request-body failure.

Context: before this cap, a compromised API key could call /answer in a
tight loop with 10MB messages, stalling the local LLM or exploding the
hosted LLM's bill. CompleteRequest already capped its message field
(v3 wire convention); /answer was the outlier.

These tests focus on validation (body-shape rejection), which happens
BEFORE the handler runs — so they don't need a real LLM or session.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from accord_ai.api import AnswerRequest, CompleteRequest, build_fastapi_app
from accord_ai.app import build_intake_app
from accord_ai.config import Settings
from accord_ai.testing import FakeEngine


@pytest.fixture
def client(tmp_path):
    settings = Settings(
        db_path=str(tmp_path / "accord.db"),
        filled_pdf_dir=str(tmp_path / "filled"),
        accord_auth_disabled=True,
        harness_max_refines=0,
    )
    intake = build_intake_app(
        settings, engine=FakeEngine(), refiner_engine=FakeEngine(),
    )
    app = build_fastapi_app(settings, intake=intake)
    with TestClient(app) as c:
        yield c, intake


# ---------------------------------------------------------------------------
# Rejection tests — validation fails before the handler runs, no LLM needed
# ---------------------------------------------------------------------------

def test_answer_rejects_message_over_cap(client):
    """50_001 chars — must fail validation with 422."""
    c, intake = client
    # Real session via the store so the session_id format passes any upstream
    # validation; message-length validation fires first anyway.
    sid = intake.store.create_session(tenant="acme")
    resp = c.post("/answer", json={
        "session_id": sid,
        "message": "x" * 50_001,
    })
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert any(
        "max_length" in str(err).lower()
        or "too long"   in str(err).lower()
        or "string_too_long" in str(err.get("type", "")).lower()
        for err in detail
    ), f"expected max_length validation error, got: {detail}"


def test_answer_rejects_wildly_oversized_payload(client):
    """10MB message — must not reach the LLM."""
    c, intake = client
    sid = intake.store.create_session(tenant="acme")
    huge = "x" * 10_000_000
    resp = c.post("/answer", json={
        "session_id": sid,
        "message": huge,
    })
    assert resp.status_code == 422


def test_answer_rejects_empty_message(client):
    """min_length=1 — empty strings shouldn't trigger the extraction loop
    with no content to extract from."""
    c, intake = client
    sid = intake.store.create_session(tenant="acme")
    resp = c.post("/answer", json={
        "session_id": sid,
        "message": "",
    })
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Schema assertions — regression canaries on the model definition itself
# ---------------------------------------------------------------------------

def test_answer_request_model_field_has_cap():
    """Regression canary against future refactors that delete the limit."""
    field = AnswerRequest.model_fields["message"]
    max_len = None
    min_len = None
    for m in (field.metadata or []):
        if hasattr(m, "max_length"):
            max_len = m.max_length
        if hasattr(m, "min_length"):
            min_len = m.min_length
    assert max_len == 50_000, f"expected max_length=50000, got {max_len}"
    assert min_len == 1, f"expected min_length=1, got {min_len}"


def test_complete_request_cap_unchanged_for_parity():
    """Both endpoints cap message at the same ceiling. If they diverge,
    it's almost certainly a mistake."""
    field = CompleteRequest.model_fields["message"]
    max_len = None
    for m in (field.metadata or []):
        if hasattr(m, "max_length"):
            max_len = m.max_length
    assert max_len == 50_000


def test_pydantic_rejects_oversized_directly():
    """AnswerRequest constructor rejects oversized messages without going
    through FastAPI — proves the constraint is on the model, not the route."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        AnswerRequest(session_id="abc", message="x" * 50_001)
    with pytest.raises(ValidationError):
        AnswerRequest(session_id="abc", message="")
    # At-cap passes
    AnswerRequest(session_id="abc", message="x" * 50_000)


def test_pydantic_accepts_normal_length():
    """Sanity check — the rejections above shouldn't be rejecting
    reasonable payloads too."""
    r = AnswerRequest(session_id="abc", message="Normal question about my fleet.")
    assert r.message.startswith("Normal")
