"""Tests for ZippopotamValidator (Phase 1.6.B)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from accord_ai.schema import Address, CustomerSubmission
from accord_ai.validation.zippopotam import ZippopotamValidator


_ZIP_75201 = {"city": "Dallas", "state": "TX", "state_full": "Texas"}
_ZIP_NOT_FOUND = None


def _sub_with_zip(zip_code: str, city: str = "", state: str = "") -> CustomerSubmission:
    return CustomerSubmission(
        mailing_address=Address(zip_code=zip_code, city=city or None, state=state or None)
    )


@pytest.mark.asyncio
async def test_zip_fills_city_state():
    """When city/state are empty, prefill fills them from Zippopotam."""
    with patch(
        "accord_ai.validation.zippopotam._lookup_zip",
        AsyncMock(return_value=_ZIP_75201),
    ):
        sub = _sub_with_zip("75201")
        just = {"mailing_address": {"zip_code": "75201"}}
        result = await ZippopotamValidator().prefill(sub, just)

    assert result is not None
    addr_patch = result.patch.get("mailing_address", {})
    assert addr_patch.get("city") == "Dallas"
    assert addr_patch.get("state") == "TX"
    assert result.conflicts == []


@pytest.mark.asyncio
async def test_zip_mismatch_city_state_creates_conflict():
    """When user supplied wrong state for the ZIP, a conflict is recorded."""
    with patch(
        "accord_ai.validation.zippopotam._lookup_zip",
        AsyncMock(return_value=_ZIP_75201),
    ):
        sub = _sub_with_zip("75201", city="Dallas", state="CA")
        just = {"mailing_address": {"zip_code": "75201"}}
        result = await ZippopotamValidator().prefill(sub, just)

    assert result is not None
    # City matches → no city conflict. State "CA" ≠ "TX" → conflict.
    assert len(result.conflicts) >= 1
    c = result.conflicts[0]
    assert c.source == "zippopotam"
    assert c.user_value == "CA"
    assert c.enriched_value == "TX"


@pytest.mark.asyncio
async def test_zip_not_found_returns_no_patch():
    """When Zippopotam returns 404 (bad ZIP), prefill returns None."""
    with patch(
        "accord_ai.validation.zippopotam._lookup_zip",
        AsyncMock(return_value=None),
    ):
        sub = _sub_with_zip("00000")
        just = {"mailing_address": {"zip_code": "00000"}}
        result = await ZippopotamValidator().prefill(sub, just)

    assert result is None
