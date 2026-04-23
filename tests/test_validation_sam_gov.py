"""Tests for SamGovValidator (Phase 1.6.D)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from accord_ai.schema import CustomerSubmission
from accord_ai.validation.sam_gov import SamGovValidator, _SamEntity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_validator():
    return SamGovValidator(api_key="test_sam_key")


def _sub(ein=None, business_name=None) -> CustomerSubmission:
    return CustomerSubmission(ein=ein, business_name=business_name)


_ACTIVE_ENTITY = _SamEntity(
    uei="ABC123XYZ456",
    cage_code="1ABC2",
    registration_status="Active",
    exp_date="2025-12-31",
    business_types=["MN", "LJ"],
)

_EXPIRED_ENTITY = _SamEntity(
    uei="DEF456XYZ789",
    cage_code="2DEF3",
    registration_status="Expired",
    exp_date="2023-06-30",
    business_types=[],
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sam_gov_active_registration_returns_info():
    """Active SAM.gov registration → info finding with UEI and CAGE code."""
    v = _make_validator()
    with patch.object(v, "_lookup", AsyncMock(return_value=_ACTIVE_ENTITY)):
        result = await v.run(_sub(ein="12-3456789"))

    assert result.success is True
    assert len(result.findings) == 1
    f = result.findings[0]
    assert f.severity == "info"
    assert "Active" in f.message
    assert f.details["uei"] == "ABC123XYZ456"
    assert f.details["cage_code"] == "1ABC2"


@pytest.mark.asyncio
async def test_sam_gov_expired_returns_warning():
    """Expired SAM.gov registration → warning finding."""
    v = _make_validator()
    with patch.object(v, "_lookup", AsyncMock(return_value=_EXPIRED_ENTITY)):
        result = await v.run(_sub(ein="12-3456789"))

    assert result.success is True
    assert len(result.findings) == 1
    f = result.findings[0]
    assert f.severity == "warning"
    assert "Expired" in f.message


@pytest.mark.asyncio
async def test_sam_gov_no_match_returns_info_finding():
    """No SAM.gov match → info finding (not an error — most businesses aren't federal)."""
    v = _make_validator()
    with patch.object(v, "_lookup", AsyncMock(return_value=None)):
        result = await v.run(_sub(ein="99-9999999"))

    assert result.success is True
    assert len(result.findings) == 1
    f = result.findings[0]
    assert f.severity == "info"
    assert "No SAM.gov" in f.message


@pytest.mark.asyncio
async def test_sam_gov_missing_ein_returns_no_findings():
    """Submission without EIN → no findings (nothing to look up)."""
    v = _make_validator()
    result = await v.run(_sub(business_name="Acme Corp"))

    assert result.success is True
    assert result.findings == []
