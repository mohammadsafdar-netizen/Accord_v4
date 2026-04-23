"""Tests for PhoneAreaValidator (Phase 1.6.D)."""

from __future__ import annotations

import pathlib
import textwrap

import pytest

from accord_ai.schema import Contact, CustomerSubmission
from accord_ai.validation.phone_area import PhoneAreaValidator, _extract_area_code


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_validator():
    return PhoneAreaValidator()  # uses real data/area_codes.csv


def _sub_with_contact_phone(phone: str) -> CustomerSubmission:
    return CustomerSubmission(contacts=[Contact(phone=phone, full_name="Jane Doe")])


def _sub_with_top_phone(phone: str) -> CustomerSubmission:
    return CustomerSubmission(phone=phone)


# ---------------------------------------------------------------------------
# Tests — _extract_area_code
# ---------------------------------------------------------------------------


def test_extract_area_code_standard_format():
    assert _extract_area_code("(214) 555-1234") == "214"


def test_extract_area_code_dashes():
    assert _extract_area_code("214-555-1234") == "214"


def test_extract_area_code_with_country_code():
    assert _extract_area_code("+1 (617) 555-9999") == "617"


def test_extract_area_code_no_area_code():
    assert _extract_area_code("555-1234") is None  # 7 digits only


# ---------------------------------------------------------------------------
# Tests — PhoneAreaValidator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_phone_area_valid_returns_no_finding():
    """Valid US area code (214 = TX) → no findings."""
    v = _make_validator()
    result = await v.run(_sub_with_contact_phone("(214) 555-1234"))

    assert result.success is True
    assert result.findings == []


@pytest.mark.asyncio
async def test_phone_area_invalid_area_code_returns_error():
    """Area code 555 is not a valid NANPA area code → error finding."""
    v = _make_validator()
    result = await v.run(_sub_with_contact_phone("(555) 123-4567"))

    assert result.success is True
    assert len(result.findings) == 1
    f = result.findings[0]
    assert f.severity == "error"
    assert "555" in f.message
    assert f.field_path == "contacts[0].phone"


@pytest.mark.asyncio
async def test_phone_area_unparseable_returns_warning():
    """Phone with too few digits → warning (not error — could be extension/typo)."""
    v = _make_validator()
    result = await v.run(_sub_with_contact_phone("555-1234"))  # 7 digits, no area code

    assert result.success is True
    assert len(result.findings) == 1
    f = result.findings[0]
    assert f.severity == "warning"
    assert "unparseable" in f.message.lower()


@pytest.mark.asyncio
async def test_phone_area_no_contacts_returns_no_findings():
    """Submission with no contacts → no findings."""
    v = _make_validator()
    result = await v.run(CustomerSubmission(business_name="Acme"))

    assert result.success is True
    assert result.findings == []
