"""Tests for /complete + /pdf endpoints (P10.A.5)."""
from __future__ import annotations

from datetime import date

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("fitz")

from accord_ai.api import build_fastapi_app
from accord_ai.app import build_intake_app
from accord_ai.config import Settings
from accord_ai.schema import (
    Address,
    CommercialAutoDetails,
    CustomerSubmission,
    Driver,
    GeneralLiabilityCoverage,
    GeneralLiabilityDetails,
    PolicyDates,
    Vehicle,
)
from accord_ai.testing import FakeEngine


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Isolate DB + filled-PDF storage per test.
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


def _seed_ca_session(intake, tenant="acme") -> str:
    sid = intake.store.create_session(tenant=tenant)
    submission = CustomerSubmission(
        business_name="Acme Trucking",
        mailing_address=Address(
            line_one="123 Main", city="Austin", state="TX", zip_code="78701",
        ),
        policy_dates=PolicyDates(effective_date=date(2026, 5, 1)),
        lob_details=CommercialAutoDetails(
            drivers=[Driver(first_name="Alice", last_name="Jones")],
            vehicles=[Vehicle(year=2024, make="Freightliner", model="Cascadia")],
        ),
    )
    intake.store.update_submission(sid, submission, tenant=tenant)
    return sid


# --- /complete happy path ----------------------------------------------------

def test_complete_returns_expected_forms(client):
    c, intake = client
    sid = _seed_ca_session(intake)

    resp = c.post(
        "/complete", json={"submission_id": sid, "tenant_slug": "acme"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == sid
    form_numbers = {f["form_number"] for f in body["forms"]}
    assert form_numbers == {"125", "127", "129", "137", "163"}
    assert body["total_bytes"] > 0
    assert body["total_written"] == 5
    for f in body["forms"]:
        assert len(f["content_hash"]) == 64
        assert f["byte_length"] > 0
        assert f["dedup_skipped"] is False


def test_complete_second_call_dedups(client):
    c, intake = client
    sid = _seed_ca_session(intake)

    first  = c.post(
        "/complete", json={"submission_id": sid, "tenant_slug": "acme"},
    ).json()
    second = c.post(
        "/complete", json={"submission_id": sid, "tenant_slug": "acme"},
    ).json()

    for f in second["forms"]:
        assert f["dedup_skipped"] is True
    assert second["total_written"] == 0
    # Hashes stable across calls.
    first_hashes  = {f["form_number"]: f["content_hash"] for f in first["forms"]}
    second_hashes = {f["form_number"]: f["content_hash"] for f in second["forms"]}
    assert first_hashes == second_hashes


def test_complete_no_lob_returns_empty_forms(client):
    c, intake = client
    sid = intake.store.create_session(tenant="acme")
    intake.store.update_submission(
        sid,
        CustomerSubmission(business_name="No LOB Yet"),
        tenant="acme",
    )
    resp = c.post(
        "/complete", json={"submission_id": sid, "tenant_slug": "acme"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["forms"] == []
    assert body["total_written"] == 0


def test_complete_gl_produces_two_forms(client):
    c, intake = client
    sid = intake.store.create_session(tenant="acme")
    intake.store.update_submission(
        sid,
        CustomerSubmission(
            business_name="GlobeX",
            lob_details=GeneralLiabilityDetails(
                coverage=GeneralLiabilityCoverage(
                    each_occurrence_limit=1_000_000,
                ),
            ),
        ),
        tenant="acme",
    )
    body = c.post(
        "/complete", json={"submission_id": sid, "tenant_slug": "acme"},
    ).json()
    assert {f["form_number"] for f in body["forms"]} == {"125", "126"}


# --- /complete error surface -------------------------------------------------

def test_complete_unknown_session_404(client):
    c, _ = client
    resp = c.post(
        "/complete",
        json={"submission_id": "x" * 32, "tenant_slug": "acme"},
    )
    assert resp.status_code == 404


def test_complete_wrong_tenant_404(client):
    c, intake = client
    sid = _seed_ca_session(intake, tenant="acme")
    resp = c.post(
        "/complete", json={"submission_id": sid, "tenant_slug": "globex"},
    )
    assert resp.status_code == 404


# --- /pdf happy path ---------------------------------------------------------

def test_get_filled_pdf(client):
    c, intake = client
    sid = _seed_ca_session(intake)
    c.post("/complete", json={"submission_id": sid, "tenant_slug": "acme"})

    resp = c.get(f"/pdf/{sid}/125", headers={"x-tenant-slug": "acme"})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert resp.content.startswith(b"%PDF-")
    assert 'acord_125_filled.pdf' in resp.headers["content-disposition"]


def test_get_filled_pdf_unknown_form_404(client):
    c, intake = client
    sid = _seed_ca_session(intake)
    c.post("/complete", json={"submission_id": sid, "tenant_slug": "acme"})
    resp = c.get(f"/pdf/{sid}/999", headers={"x-tenant-slug": "acme"})
    assert resp.status_code == 404


def test_get_filled_pdf_before_complete_404(client):
    c, intake = client
    sid = _seed_ca_session(intake)
    resp = c.get(f"/pdf/{sid}/125", headers={"x-tenant-slug": "acme"})
    assert resp.status_code == 404


def test_get_filled_pdf_unknown_session_404(client):
    c, _ = client
    resp = c.get(f"/pdf/{'x'*32}/125", headers={"x-tenant-slug": "acme"})
    assert resp.status_code == 404


def test_get_filled_pdf_wrong_tenant_404(client):
    c, intake = client
    sid = _seed_ca_session(intake, tenant="acme")
    c.post("/complete", json={"submission_id": sid, "tenant_slug": "acme"})
    # Caller claims globex — should look identical to "no such session".
    resp = c.get(f"/pdf/{sid}/125", headers={"x-tenant-slug": "globex"})
    assert resp.status_code == 404


def test_get_filled_pdf_malformed_path_segments_404(client):
    c, intake = client
    sid = _seed_ca_session(intake)
    c.post("/complete", json={"submission_id": sid, "tenant_slug": "acme"})
    # Malformed form number → ValueError in the store → 404 via the handler.
    resp = c.get(f"/pdf/{sid}/abc", headers={"x-tenant-slug": "acme"})
    assert resp.status_code == 404


# --- Dedup interaction with data changes -------------------------------------

def test_complete_rewrites_after_data_change(client):
    c, intake = client
    sid = _seed_ca_session(intake)
    first = c.post(
        "/complete", json={"submission_id": sid, "tenant_slug": "acme"},
    ).json()

    # Update submission — 125 holds business_name so its hash must change.
    submission = intake.store.get_session(sid, tenant="acme").submission
    updated = submission.model_copy(update={"business_name": "Different Corp"})
    intake.store.update_submission(sid, updated, tenant="acme")

    second = c.post(
        "/complete", json={"submission_id": sid, "tenant_slug": "acme"},
    ).json()
    first_125  = next(f for f in first["forms"]  if f["form_number"] == "125")
    second_125 = next(f for f in second["forms"] if f["form_number"] == "125")
    assert first_125["content_hash"] != second_125["content_hash"]
    assert second_125["dedup_skipped"] is False


# --- Audit events ------------------------------------------------------------

def test_complete_records_audit_event(client):
    c, intake = client
    sid = _seed_ca_session(intake)
    c.post("/complete", json={"submission_id": sid, "tenant_slug": "acme"})

    events = intake.store.list_audit_events(
        event_type="submission.completed",
    )
    assert len(events) == 1
    e = events[0]
    assert e.tenant == "acme"
    assert set(e.payload["forms"]) == {"125", "127", "129", "137", "163"}
    assert e.payload["total_bytes"] > 0
    assert e.payload["total_written"] == 5


# --- Chat-open gate ----------------------------------------------------------

def test_complete_respects_chat_open_gate(tmp_path):
    """ACCORD_CHAT_OPEN=true must open /complete (matches v3 convention)."""
    settings = Settings(
        db_path=str(tmp_path / "accord.db"),
        filled_pdf_dir=str(tmp_path / "filled"),
        accord_auth_disabled=False,
        accord_chat_open=True,
        intake_api_key=None,   # would normally be required
        harness_max_refines=0,
    )
    intake = build_intake_app(
        settings, engine=FakeEngine(), refiner_engine=FakeEngine(),
    )
    app = build_fastapi_app(settings, intake=intake)
    sid = _seed_ca_session(intake)

    with TestClient(app) as c:
        resp = c.post(
            "/complete",
            json={"submission_id": sid, "tenant_slug": "acme"},
            headers={"x-tenant-slug": "acme"},
        )
        assert resp.status_code == 200

        # /pdf is NOT in chat-open — still gated.
        resp = c.get(f"/pdf/{sid}/125", headers={"x-tenant-slug": "acme"})
        assert resp.status_code in (401, 500)  # no auth configured at all
