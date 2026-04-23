"""Tests for UspsValidator (Phase 1.6.C)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from accord_ai.schema import Address, CustomerSubmission
from accord_ai.validation.usps import UspsValidator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sub_with_mailing(
    line_one="100 Main St",
    city="Dallas",
    state="TX",
    zip_code="75201",
) -> CustomerSubmission:
    return CustomerSubmission(
        mailing_address=Address(line_one=line_one, city=city, state=state, zip_code=zip_code)
    )


_STD_SAME = {
    "streetAddress": "100 MAIN ST",
    "city": "DALLAS",
    "state": "TX",
    "ZIPCode": "75201",
    "ZIPPlus4": "1234",
}

_STD_DIFFERS = {
    "streetAddress": "100 MAIN STREET",
    "city": "DALLAS",
    "state": "TX",
    "ZIPCode": "75201",
    "ZIPPlus4": "9999",
}


def _make_validator():
    return UspsValidator(consumer_key="test_key", consumer_secret="test_secret")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_valid_address_confirmed_deliverable():
    """When standardized address matches entered, emits info finding."""
    from accord_ai.validation.usps import _ADDR_OK
    v = _make_validator()
    with (
        patch.object(v, "_get_token", AsyncMock(return_value="tok")),
        patch("accord_ai.validation.usps._lookup_address", AsyncMock(return_value=(_ADDR_OK, _STD_SAME))),
    ):
        result = await v.run(_sub_with_mailing())

    assert result.success is True
    assert len(result.findings) == 1
    f = result.findings[0]
    assert f.severity == "info"
    assert "deliverable" in f.message.lower()


@pytest.mark.asyncio
async def test_standardized_differs_emits_warning():
    """When USPS standardizes to a different address, emits warning."""
    from accord_ai.validation.usps import _ADDR_OK
    v = _make_validator()
    with (
        patch.object(v, "_get_token", AsyncMock(return_value="tok")),
        patch("accord_ai.validation.usps._lookup_address", AsyncMock(return_value=(_ADDR_OK, _STD_DIFFERS))),
    ):
        result = await v.run(_sub_with_mailing())

    assert result.success is True
    assert len(result.findings) >= 1
    assert any(f.severity == "warning" for f in result.findings)


@pytest.mark.asyncio
async def test_no_street_address_skipped():
    """Address without line_one is skipped — no findings."""
    v = _make_validator()
    sub = CustomerSubmission(
        mailing_address=Address(city="Dallas", state="TX", zip_code="75201")
    )
    with patch.object(v, "_get_token", AsyncMock(return_value="tok")):
        result = await v.run(sub)

    assert result.success is True
    assert result.findings == []


@pytest.mark.asyncio
async def test_token_fetch_failure_returns_error():
    """When OAuth token cannot be obtained, returns success=False."""
    v = _make_validator()
    with patch.object(v, "_get_token", AsyncMock(return_value=None)):
        result = await v.run(_sub_with_mailing())

    assert result.success is False
    assert result.error is not None


@pytest.mark.asyncio
async def test_prefill_returns_none():
    """UspsValidator.prefill() always returns None (finalize-only)."""
    v = _make_validator()
    result = await v.prefill(CustomerSubmission(), {})
    assert result is None
