"""Tests for FmcsaValidator (Phase 1.6.D)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from accord_ai.schema import CommercialAutoDetails, CustomerSubmission
from accord_ai.validation.fmcsa import FmcsaValidator, _CarrierData, _parse_carrier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_validator():
    return FmcsaValidator(web_key="test_key")


def _carrier(
    operating_status="AUTHORIZED",
    safety_rating="SATISFACTORY",
    out_of_service=False,
    rated_date="2024-01-01",
) -> _CarrierData:
    return _CarrierData(
        legal_name="Acme Trucking LLC",
        operating_status=operating_status,
        safety_rating=safety_rating,
        rated_date=rated_date,
        out_of_service=out_of_service,
        power_units=50,
        drivers=45,
        crash_count_2yr=2,
        inspection_count_2yr=10,
    )


def _sub_with_dot(dot="1234567") -> CustomerSubmission:
    return CustomerSubmission(
        lob_details=CommercialAutoDetails(fmcsa_dot_number=dot),
    )


def _sub_trucking_naics_no_dot() -> CustomerSubmission:
    return CustomerSubmission(
        naics_code="484110",
        lob_details=CommercialAutoDetails(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fmcsa_active_authorized_returns_info():
    """Authorized + satisfactory carrier → single info finding with carrier details."""
    v = _make_validator()
    with patch.object(v, "_fetch", AsyncMock(return_value=_carrier())):
        result = await v.run(_sub_with_dot())

    assert result.success is True
    info_findings = [f for f in result.findings if f.severity == "info"]
    assert len(info_findings) == 1
    assert "Acme Trucking" in info_findings[0].details["legal_name"]
    assert info_findings[0].details["power_units"] == 50


@pytest.mark.asyncio
async def test_fmcsa_out_of_service_returns_error():
    """NOT AUTHORIZED operating status → error finding."""
    v = _make_validator()
    with patch.object(v, "_fetch", AsyncMock(return_value=_carrier(
        operating_status="NOT AUTHORIZED", out_of_service=True
    ))):
        result = await v.run(_sub_with_dot())

    assert result.success is True
    assert any(f.severity == "error" and "NOT AUTHORIZED" in f.message for f in result.findings)


@pytest.mark.asyncio
async def test_fmcsa_conditional_safety_rating_returns_warning():
    """CONDITIONAL safety rating → warning finding in addition to info."""
    v = _make_validator()
    with patch.object(v, "_fetch", AsyncMock(return_value=_carrier(safety_rating="CONDITIONAL"))):
        result = await v.run(_sub_with_dot())

    assert result.success is True
    warning_findings = [f for f in result.findings if f.severity == "warning"]
    assert len(warning_findings) == 1
    assert "CONDITIONAL" in warning_findings[0].message


@pytest.mark.asyncio
async def test_fmcsa_trucking_naics_without_dot_number_warns():
    """Trucking NAICS (484xxx) but no DOT number → warning finding."""
    v = _make_validator()
    result = await v.run(_sub_trucking_naics_no_dot())

    assert result.success is True
    assert any(f.severity == "warning" and "dot number" in f.message.lower() for f in result.findings)


@pytest.mark.asyncio
async def test_fmcsa_non_trucking_submission_skipped():
    """Non-CA submission with no DOT → no findings (not applicable)."""
    v = _make_validator()
    sub = CustomerSubmission(business_name="Acme Bakery", naics_code="722511")
    result = await v.run(sub)

    assert result.success is True
    assert result.findings == []
