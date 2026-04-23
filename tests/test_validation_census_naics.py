"""Tests for NaicsValidator (Phase 1.6.C)."""

from __future__ import annotations

import pathlib
import textwrap
from unittest.mock import patch

import pytest

from accord_ai.schema import CustomerSubmission
from accord_ai.validation.census_naics import NaicsValidator, _load_index, lookup_naics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sub(naics_code=None, naics_description=None) -> CustomerSubmission:
    return CustomerSubmission(
        naics_code=naics_code,
        naics_description=naics_description,
    )


_MINI_INDEX = {
    "541512": "Computer Systems Design Services",
    "722511": "Full-Service Restaurants",
    "236115": "New Single-Family Housing Construction (Except For-Sale Builders)",
    "11": "Agriculture, Forestry, Fishing and Hunting",
    "541": "Professional, Scientific, and Technical Services",
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prefill_fills_description_when_missing():
    """prefill() fills naics_description when naics_code is just extracted."""
    with patch("accord_ai.validation.census_naics._get_index", return_value=_MINI_INDEX):
        sub = _sub(naics_code="541512")
        just = {"naics_code": "541512"}
        result = await NaicsValidator().prefill(sub, just)

    assert result is not None
    assert result.patch["naics_description"] == "Computer Systems Design Services"
    assert result.source == "census_naics"


@pytest.mark.asyncio
async def test_prefill_noop_when_description_already_present():
    """prefill() returns None when naics_description is already set."""
    with patch("accord_ai.validation.census_naics._get_index", return_value=_MINI_INDEX):
        sub = _sub(naics_code="541512", naics_description="IT Consulting")
        just = {"naics_code": "541512"}
        result = await NaicsValidator().prefill(sub, just)

    assert result is None


@pytest.mark.asyncio
async def test_run_invalid_code_emits_warning():
    """run() emits a warning finding for an unknown NAICS code."""
    with patch("accord_ai.validation.census_naics._get_index", return_value=_MINI_INDEX):
        sub = _sub(naics_code="999999")
        result = await NaicsValidator().run(sub)

    assert result.success is True
    assert len(result.findings) == 1
    f = result.findings[0]
    assert f.severity == "warning"
    assert "999999" in f.message
    assert f.field_path == "naics_code"


@pytest.mark.asyncio
async def test_run_description_mismatch_emits_warning():
    """run() warns when stated description differs from official NAICS title."""
    with patch("accord_ai.validation.census_naics._get_index", return_value=_MINI_INDEX):
        sub = _sub(naics_code="541512", naics_description="Software Development")
        result = await NaicsValidator().run(sub)

    assert result.success is True
    assert any("does not match" in f.message for f in result.findings)
    assert result.findings[0].severity == "warning"


@pytest.mark.asyncio
async def test_six_digit_code_supported():
    """6-digit NAICS codes (national level) are looked up correctly."""
    with patch("accord_ai.validation.census_naics._get_index", return_value=_MINI_INDEX):
        sub = _sub(naics_code="722511")
        just = {"naics_code": "722511"}
        result = await NaicsValidator().prefill(sub, just)

    assert result is not None
    assert result.patch["naics_description"] == "Full-Service Restaurants"


def test_eager_index_is_loaded():
    """_NAICS_INDEX is non-empty at import time — eager load succeeded."""
    from accord_ai.validation.census_naics import _NAICS_INDEX
    assert len(_NAICS_INDEX) > 0
    # Spot-check a well-known code present in the committed CSV
    assert any(code.startswith("48") or code.startswith("54") for code in _NAICS_INDEX)


def test_load_index_from_csv(tmp_path):
    """_load_index() parses a CSV with code,title columns."""
    csv_content = textwrap.dedent("""\
        code,title
        541512,Computer Systems Design Services
        722511,Full-Service Restaurants
    """)
    csv_file = tmp_path / "naics.csv"
    csv_file.write_text(csv_content)
    idx = _load_index(csv_file)
    assert idx["541512"] == "Computer Systems Design Services"
    assert idx["722511"] == "Full-Service Restaurants"
