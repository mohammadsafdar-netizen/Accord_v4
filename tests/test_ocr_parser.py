"""Unit tests for accord_ai.extraction.ocr.parser (Phase 1.5).

Uses FakeEngine — no LLM required.
"""
from __future__ import annotations

from datetime import date

import pytest

from accord_ai.extraction.ocr.parser import (
    DriverLicenseFields,
    InsuranceCardFields,
    RegistrationFields,
    parse_driver_license,
    parse_insurance_card,
    parse_registration,
)
from accord_ai.llm.fake_engine import FakeEngine


# ---------------------------------------------------------------------------
# 5. parse_driver_license → populated fields
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_parse_driver_license_populates_fields():
    payload = {
        "first_name": "John",
        "last_name": "Doe",
        "license_number": "TX12345678",
        "license_state": "TX",
        "license_class": "C",
        "date_of_birth": "1985-01-15",
        "license_expiration": "2029-01-15",
        "address": "123 Main St",
    }
    engine = FakeEngine([payload])
    result = await parse_driver_license("fake OCR text", engine)
    assert isinstance(result, DriverLicenseFields)
    assert result.first_name == "John"
    assert result.last_name == "Doe"
    assert result.license_number == "TX12345678"
    assert result.license_state == "TX"
    assert result.date_of_birth == date(1985, 1, 15)
    assert result.license_expiration == date(2029, 1, 15)


# ---------------------------------------------------------------------------
# 6. parse_insurance_card → populated fields
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_parse_insurance_card_populates_fields():
    payload = {
        "carrier": "Acme Insurance Co",
        "policy_number": "POL-9876543",
        "named_insured": "John Doe",
        "effective_date": "2025-01-01",
        "expiration_date": "2026-01-01",
    }
    engine = FakeEngine([payload])
    result = await parse_insurance_card("fake OCR text", engine)
    assert isinstance(result, InsuranceCardFields)
    assert result.carrier == "Acme Insurance Co"
    assert result.policy_number == "POL-9876543"
    assert result.effective_date == date(2025, 1, 1)


# ---------------------------------------------------------------------------
# 7. parse_registration → populated fields
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_parse_registration_populates_fields():
    payload = {
        "vin": "1HGCM82633A004352",
        "year": 2020,
        "make": "Honda",
        "model": "Civic",
        "registration_state": "TX",
        "registration_expiration": "2025-12-31",
        "owner_name": "John Doe",
    }
    engine = FakeEngine([payload])
    result = await parse_registration("fake OCR text", engine)
    assert isinstance(result, RegistrationFields)
    assert result.vin == "1HGCM82633A004352"
    assert result.year == 2020
    assert result.make == "Honda"
    assert result.registration_state == "TX"


# ---------------------------------------------------------------------------
# 8. missing fields → None (not raised)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_parser_returns_none_for_missing_fields():
    # Only first_name provided — everything else should be None
    payload = {"first_name": "Jane"}
    engine = FakeEngine([payload])
    result = await parse_driver_license("partial OCR text", engine)
    assert result.first_name == "Jane"
    assert result.last_name is None
    assert result.date_of_birth is None
    assert result.license_number is None
