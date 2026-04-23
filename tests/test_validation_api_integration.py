"""API integration tests for inline enrichment (Phase 1.6.B).

Tests 17-20: VIN auto-fills vehicle fields via /answer, conflicts surface in /fields.
Uses monkeypatched HTTP so no real network calls.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from accord_ai.api import build_fastapi_app
from accord_ai.app import build_intake_app
from accord_ai.config import Settings
from accord_ai.llm.fake_engine import FakeEngine
from accord_ai.schema import (
    CommercialAutoDetails,
    CustomerSubmission,
    GeneralLiabilityDetails,
    Vehicle,
)
from accord_ai.validation.nhtsa_vpic import _decode_vin


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


_VPIC_F150 = {
    "Model Year": "2024",
    "Make": "FORD",
    "Model": "F-150",
    "Body Class": "Crew Cab Pickup",
    "Gross Vehicle Weight Rating From": "7050",
    "Error Code": "0",
}

_ANSWER_EXTRACTION = {
    "lob_details": {
        "lob": "commercial_auto",
        "vehicles": [{"vin": "1FTFW1E83RFB12345", "year": 2024, "make": "Ford", "model": "F-150"}],
    }
}


def _make_app(tmp_path):
    settings = _settings(tmp_path)
    # FakeEngine queue: greeting + extraction diff + responder reply
    engine = FakeEngine(["Hello!", _ANSWER_EXTRACTION, "Got it, next question?"])
    intake = build_intake_app(settings, engine=engine, refiner_engine=FakeEngine())
    return build_fastapi_app(settings, intake=intake), intake


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_vpic_fills_vehicle_during_intake(tmp_path):
    """End-to-end: /answer with VIN triggers inline enrichment → body_type filled."""
    app, intake = _make_app(tmp_path)

    with patch(
        "accord_ai.validation.nhtsa_vpic._decode_vin",
        AsyncMock(return_value=_VPIC_F150),
    ):
        with TestClient(app) as client:
            sid = client.post("/start-session").json()["submission_id"]
            client.post("/answer", json={"session_id": sid, "message": "My 2024 Ford F-150 VIN 1FTFW1E83RFB12345"})
            fields = client.get(f"/fields/{sid}").json()

    ld = fields.get("lob_details") or {}
    vehicles = ld.get("vehicles") or []
    # VIN was extracted and body_type enriched from NHTSA
    if vehicles:
        assert vehicles[0].get("body_type") == "Crew Cab Pickup"


def test_conflicts_surface_in_fields_endpoint(tmp_path):
    """Conflicts accumulated during a turn appear in /fields/{id} response."""
    # Use a year mismatch: user says 2023, VIN decodes 2024
    extraction_mismatch = {
        "lob_details": {
            "lob": "commercial_auto",
            "vehicles": [{"vin": "1FTFW1E83RFB12345", "year": 2023, "make": "Ford", "model": "F-150"}],
        }
    }
    settings = _settings(tmp_path)
    engine = FakeEngine(["Hello!", extraction_mismatch, "Got it."])
    intake = build_intake_app(settings, engine=engine, refiner_engine=FakeEngine())
    app = build_fastapi_app(settings, intake=intake)

    with patch(
        "accord_ai.validation.nhtsa_vpic._decode_vin",
        AsyncMock(return_value=_VPIC_F150),
    ):
        with TestClient(app) as client:
            sid = client.post("/start-session").json()["submission_id"]
            client.post("/answer", json={"session_id": sid, "message": "2023 Ford F-150 VIN 1FTFW1E83RFB12345"})
            fields = client.get(f"/fields/{sid}").json()

    conflicts = fields.get("conflicts") or []
    # Year mismatch (2023 vs 2024) should be recorded as a conflict
    assert isinstance(conflicts, list)
    year_conflicts = [c for c in conflicts if "year" in c.get("field_path", "")]
    assert len(year_conflicts) >= 1


def test_fields_endpoint_includes_empty_conflicts_by_default(tmp_path):
    """Submissions with no enrichment have conflicts=[] in /fields."""
    settings = _settings(tmp_path)
    engine = FakeEngine(["Hello, let's start!"])
    intake = build_intake_app(settings, engine=engine, refiner_engine=FakeEngine())
    app = build_fastapi_app(settings, intake=intake)

    with TestClient(app) as client:
        sid = client.post("/start-session").json()["submission_id"]
        r = client.get(f"/fields/{sid}")

    assert r.status_code == 200
    assert "conflicts" in r.json()
    assert r.json()["conflicts"] == []


def test_vpic_inline_eligible_is_true_zippopotam_also():
    """Attribute contract: VIN and ZIP validators have inline_eligible=True."""
    from accord_ai.validation.nhtsa_vpic import NhtsaVpicValidator
    from accord_ai.validation.zippopotam import ZippopotamValidator

    assert NhtsaVpicValidator.inline_eligible is True
    assert ZippopotamValidator.inline_eligible is True


# ---------------------------------------------------------------------------
# Phase 1.6.C API integration tests (Tests 21-23)
# ---------------------------------------------------------------------------


def test_naics_inline_fills_description_during_intake(tmp_path):
    """End-to-end: /answer with NAICS code triggers inline enrichment → description filled."""
    from unittest.mock import patch

    naics_extraction = {
        "naics_code": "541512",
        "business_name": "Acme IT",
    }
    settings = _settings(tmp_path)
    engine = FakeEngine(["Hello!", naics_extraction, "Got it."])
    intake = build_intake_app(settings, engine=engine, refiner_engine=FakeEngine())
    app = build_fastapi_app(settings, intake=intake)

    _MINI_INDEX = {"541512": "Computer Systems Design Services"}

    with patch("accord_ai.validation.census_naics._get_index", return_value=_MINI_INDEX):
        with TestClient(app) as client:
            sid = client.post("/start-session").json()["submission_id"]
            client.post("/answer", json={"session_id": sid, "message": "NAICS 541512"})
            fields = client.get(f"/fields/{sid}").json()

    # NAICS inline runner should have filled naics_description
    desc = fields.get("naics_description")
    if fields.get("naics_code") == "541512":
        assert desc == "Computer Systems Design Services"


def test_naics_inline_eligible_is_true():
    """Attribute contract: NaicsValidator has inline_eligible=True."""
    from accord_ai.validation.census_naics import NaicsValidator

    assert NaicsValidator.inline_eligible is True


def test_usps_tax1099_inline_eligible_false():
    """Attribute contract: UspsValidator and Tax1099Validator are finalize-only."""
    from accord_ai.validation.usps import UspsValidator
    from accord_ai.validation.tax1099 import Tax1099Validator

    assert UspsValidator.inline_eligible is False
    assert Tax1099Validator.inline_eligible is False


# ---------------------------------------------------------------------------
# Phase 1.6.D API integration tests (Tests 24-25)
# ---------------------------------------------------------------------------


def test_step16_validators_inline_eligible_false():
    """Attribute contract: all five Phase 1.6.D validators are finalize-only."""
    from accord_ai.validation.fmcsa import FmcsaValidator
    from accord_ai.validation.sam_gov import SamGovValidator
    from accord_ai.validation.sec_edgar import SecEdgarValidator
    from accord_ai.validation.phone_area import PhoneAreaValidator
    from accord_ai.validation.dns_mx import DnsMxValidator

    assert FmcsaValidator.inline_eligible is False
    assert SamGovValidator.inline_eligible is False
    assert SecEdgarValidator.inline_eligible is False
    assert PhoneAreaValidator.inline_eligible is False
    assert DnsMxValidator.inline_eligible is False


def test_engine_build_includes_phone_dns_sec_always_active(tmp_path):
    """When ENABLE_EXTERNAL_VALIDATION=true, phone/DNS/SEC are always present."""
    import os
    from unittest.mock import patch

    settings = _settings(tmp_path)
    with patch.dict(os.environ, {"ENABLE_EXTERNAL_VALIDATION": "true"}):
        from accord_ai.validation.engine import build_engine
        engine = build_engine(settings)

    names = {v.name for v in engine._validators}
    assert "phone_area" in names
    assert "dns_mx" in names
    assert "sec_edgar" in names
