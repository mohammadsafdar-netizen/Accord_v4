"""Tests for POST /upload-image (Phase 1.5 — real OCR endpoint)."""
from __future__ import annotations

import io
import json
from datetime import date
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from accord_ai.api import build_fastapi_app
from accord_ai.app import build_intake_app
from accord_ai.config import Settings
from accord_ai.llm.fake_engine import FakeEngine
from accord_ai.schema import (
    CommercialAutoDetails,
    CustomerSubmission,
    Driver,
    GeneralLiabilityDetails,
    Vehicle,
    WorkersCompDetails,
)


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


def _make_app(tmp_path, engine: FakeEngine | None = None):
    settings = _settings(tmp_path)
    eng = engine or FakeEngine()
    intake = build_intake_app(settings, engine=eng, refiner_engine=FakeEngine())
    app = build_fastapi_app(settings, intake=intake)
    return app, intake


def _make_png(width=300, height=200) -> bytes:
    img = Image.new("RGB", (width, height), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _upload(client, session_id: str, kind="drivers_license", data=None):
    return client.post(
        "/upload-image",
        data={"kind": kind, "session_id": session_id},
        files={"file": ("id.png", io.BytesIO(data or _make_png()), "image/png")},
    )


def _seed_ca_session(intake, drivers=None, vehicles=None) -> str:
    sid = intake.store.create_session()
    sub = CustomerSubmission(
        business_name="Acme",
        lob_details=CommercialAutoDetails(
            drivers=drivers or [],
            vehicles=vehicles or [],
        ),
    )
    intake.store.update_submission(sid, sub)
    return sid


def _seed_wc_session(intake) -> str:
    sid = intake.store.create_session()
    sub = CustomerSubmission(
        business_name="Acme",
        lob_details=WorkersCompDetails(),
    )
    intake.store.update_submission(sid, sub)
    return sid


def _seed_gl_session(intake) -> str:
    sid = intake.store.create_session()
    sub = CustomerSubmission(
        business_name="Acme",
        lob_details=GeneralLiabilityDetails(),
    )
    intake.store.update_submission(sid, sub)
    return sid


def _dl_payload(**overrides) -> dict:
    base = {
        "first_name": "John",
        "last_name": "Doe",
        "license_number": "TX12345678",
        "license_state": "TX",
        "license_class": "C",
        "date_of_birth": "1985-01-15",
        "license_expiration": "2029-01-15",
    }
    base.update(overrides)
    return base


def _reg_payload(**overrides) -> dict:
    base = {
        "vin": "1HGCM82633A004352",
        "year": 2020,
        "make": "Honda",
        "model": "Civic",
        "registration_state": "TX",
    }
    base.update(overrides)
    return base


def _ic_payload(**overrides) -> dict:
    base = {
        "carrier": "Acme Insurance",
        "policy_number": "POL-999",
        "effective_date": "2025-01-01",
        "expiration_date": "2026-01-01",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 9. upload DL → new driver added
# ---------------------------------------------------------------------------

def test_upload_dl_merges_new_driver(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "accord_ai.extraction.ocr.ocr_image",
        lambda _: "JOHN DOE TX12345678",
    )
    engine = FakeEngine([_dl_payload()])
    app, intake = _make_app(tmp_path, engine)
    sid = _seed_ca_session(intake)

    with TestClient(app) as client:
        r = _upload(client, sid, kind="drivers_license")

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["kind"] == "drivers_license"
    assert "license_number" in body["extracted"]

    session = intake.store.get_session(sid)
    drivers = session.submission.lob_details.drivers
    assert len(drivers) == 1
    assert drivers[0].last_name == "Doe"
    assert drivers[0].license_number == "TX12345678"


# ---------------------------------------------------------------------------
# 10. upload DL → existing driver merged (not duplicated)
# ---------------------------------------------------------------------------

def test_upload_dl_merges_existing_driver_by_license_number(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "accord_ai.extraction.ocr.ocr_image",
        lambda _: "JOHN DOE TX12345678",
    )
    # Return DL with class C — existing driver has no class, should get merged
    engine = FakeEngine([_dl_payload(license_class="C")])
    app, intake = _make_app(tmp_path, engine)
    # Pre-seed a driver with same license_number but missing license_state
    existing = Driver(first_name="John", last_name="Doe", license_number="TX12345678")
    sid = _seed_ca_session(intake, drivers=[existing])

    with TestClient(app) as client:
        r = _upload(client, sid, kind="drivers_license")

    assert r.status_code == 200
    session = intake.store.get_session(sid)
    drivers = session.submission.lob_details.drivers
    # Should still be 1 driver — merge, not append
    assert len(drivers) == 1
    # license_state merged in from the DL upload
    assert drivers[0].license_state == "TX"


# ---------------------------------------------------------------------------
# 11. upload registration → vehicle merged by VIN
# ---------------------------------------------------------------------------

def test_upload_registration_merges_vehicle_by_vin(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "accord_ai.extraction.ocr.ocr_image",
        lambda _: "VIN 1HGCM82633A004352",
    )
    engine = FakeEngine([_reg_payload()])
    app, intake = _make_app(tmp_path, engine)
    sid = _seed_ca_session(intake)

    with TestClient(app) as client:
        r = _upload(client, sid, kind="vehicle_registration")

    assert r.status_code == 200
    session = intake.store.get_session(sid)
    vehicles = session.submission.lob_details.vehicles
    assert len(vehicles) == 1
    assert vehicles[0].vin == "1HGCM82633A004352"
    assert vehicles[0].make == "Honda"


# ---------------------------------------------------------------------------
# 12. upload insurance card → prior_insurance on WC lob
# ---------------------------------------------------------------------------

def test_upload_insurance_card_fills_prior_insurance(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "accord_ai.extraction.ocr.ocr_image",
        lambda _: "POLICY POL-999",
    )
    engine = FakeEngine([_ic_payload()])
    app, intake = _make_app(tmp_path, engine)
    sid = _seed_wc_session(intake)

    with TestClient(app) as client:
        r = _upload(client, sid, kind="insurance_card")

    assert r.status_code == 200
    session = intake.store.get_session(sid)
    prior = session.submission.lob_details.prior_insurance
    assert len(prior) == 1
    assert prior[0].policy_number == "POL-999"
    assert prior[0].carrier_name == "Acme Insurance"


# ---------------------------------------------------------------------------
# 13. unreadable image → 422
# ---------------------------------------------------------------------------

def test_upload_unreadable_returns_422(tmp_path, monkeypatch):
    from accord_ai.extraction.ocr.errors import OCRReadError
    monkeypatch.setattr(
        "accord_ai.extraction.ocr.ocr_image",
        Mock(side_effect=OCRReadError("blank image")),
    )
    engine = FakeEngine()
    app, intake = _make_app(tmp_path, engine)
    sid = _seed_ca_session(intake)

    with TestClient(app) as client:
        r = _upload(client, sid, kind="drivers_license")

    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "ocr_failed"


# ---------------------------------------------------------------------------
# 14. bad kind → 422
# ---------------------------------------------------------------------------

def test_upload_bad_kind_returns_422(tmp_path):
    app, intake = _make_app(tmp_path)
    sid = "any-fake-session-id"
    with TestClient(app) as client:
        r = _upload(client, sid, kind="unknown_kind")
    assert r.status_code == 422
    assert "kind" in r.json()["detail"]


# ---------------------------------------------------------------------------
# 15. oversize → 413
# ---------------------------------------------------------------------------

def test_upload_oversize_returns_413(tmp_path):
    app, intake = _make_app(tmp_path)
    big = b"\xff" * (10 * 1024 * 1024 + 1)
    with TestClient(app) as client:
        r = client.post(
            "/upload-image",
            data={"kind": "drivers_license", "session_id": "fake"},
            files={"file": ("id.png", io.BytesIO(big), "image/png")},
        )
    assert r.status_code == 413


# ---------------------------------------------------------------------------
# 16. OCR text PII is redacted before logging
# ---------------------------------------------------------------------------

def test_upload_ocr_output_pii_redacted_in_logs(tmp_path, monkeypatch, caplog):
    import logging
    monkeypatch.setattr(
        "accord_ai.extraction.ocr.ocr_image",
        lambda _: "JOHN DOE\nEIN 12-3456789\nDL TX12345678",
    )
    engine = FakeEngine([_dl_payload()])
    app, intake = _make_app(tmp_path, engine)
    sid = _seed_ca_session(intake)

    with caplog.at_level(logging.DEBUG, logger="accord_ai.api"):
        with TestClient(app) as client:
            _upload(client, sid, kind="drivers_license")

    log_text = " ".join(caplog.messages)
    assert "12-3456789" not in log_text


# ---------------------------------------------------------------------------
# 17. wrong LOB for vehicle_registration → 422
# ---------------------------------------------------------------------------

def test_upload_wrong_lob_for_registration_rejects(tmp_path):
    app, intake = _make_app(tmp_path)
    sid = _seed_gl_session(intake)
    with TestClient(app) as client:
        r = _upload(client, sid, kind="vehicle_registration")
    assert r.status_code == 422
    assert "commercial_auto" in r.json()["detail"]


# ---------------------------------------------------------------------------
# 18. Tesseract not installed → 503
# ---------------------------------------------------------------------------

def test_upload_without_tesseract_returns_503(tmp_path, monkeypatch):
    from accord_ai.extraction.ocr.errors import OCRConfigError
    monkeypatch.setattr(
        "accord_ai.extraction.ocr.ocr_image",
        Mock(side_effect=OCRConfigError("tesseract not found")),
    )
    engine = FakeEngine()
    app, intake = _make_app(tmp_path, engine)
    sid = _seed_ca_session(intake)

    with TestClient(app) as client:
        r = _upload(client, sid, kind="drivers_license")

    assert r.status_code == 503
    assert r.json()["detail"]["error"] == "ocr_unavailable"
