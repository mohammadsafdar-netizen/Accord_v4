"""Tests for CrossFieldValidator (Phase 1.6.E) — 15 tests."""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from unittest.mock import patch

import pytest

from accord_ai.schema import (
    Address,
    CommercialAutoDetails,
    Contact,
    CustomerSubmission,
    Driver,
    GeneralLiabilityDetails,
    PolicyDates,
    Vehicle,
)
from accord_ai.validation.cross_field import CrossFieldValidator, _ALL_CHECKS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_TODAY = date.today()


def _run(sub: CustomerSubmission) -> list:
    import asyncio
    return asyncio.run(CrossFieldValidator().run(sub))


def _ca_sub(**ca_kwargs) -> CustomerSubmission:
    return CustomerSubmission(
        lob_details=CommercialAutoDetails(**ca_kwargs),
    )


# ---------------------------------------------------------------------------
# 1. driver_age_vs_experience
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_driver_age_vs_experience_impossible():
    """22yo driver with 10 years experience → warning (max = 22-16 = 6)."""
    dob = _TODAY - timedelta(days=365 * 22)
    sub = _ca_sub(drivers=[Driver(date_of_birth=dob, years_experience=10)])
    result = await CrossFieldValidator().run(sub)
    assert any("experience" in f.message.lower() for f in result.findings)
    assert any(f.severity == "warning" for f in result.findings)


@pytest.mark.asyncio
async def test_driver_age_vs_experience_ok():
    """40yo driver with 10 years experience → no finding."""
    dob = _TODAY - timedelta(days=365 * 40)
    sub = _ca_sub(drivers=[Driver(date_of_birth=dob, years_experience=10)])
    result = await CrossFieldValidator().run(sub)
    exp_findings = [f for f in result.findings if "experience" in f.message.lower()]
    assert exp_findings == []


# ---------------------------------------------------------------------------
# 2. revenue_per_employee
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_revenue_per_employee_extreme_high():
    """$10M revenue, 1 employee → info (RPE = $10M, > $2M threshold)."""
    sub = CustomerSubmission(annual_revenue=Decimal("10000000"), full_time_employees=1)
    result = await CrossFieldValidator().run(sub)
    assert any(f.severity == "info" and "revenue" in f.message.lower() for f in result.findings)


@pytest.mark.asyncio
async def test_revenue_per_employee_extreme_low():
    """$10K revenue, 10 employees → info (RPE = $1K, < $15K threshold)."""
    sub = CustomerSubmission(annual_revenue=Decimal("10000"), full_time_employees=10)
    result = await CrossFieldValidator().run(sub)
    assert any(f.severity == "info" and "revenue" in f.message.lower() for f in result.findings)


# ---------------------------------------------------------------------------
# 3 & 4. vehicle/driver count mismatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vehicle_count_mismatch():
    """vehicle_count=5 but 2 vehicles listed → warning."""
    sub = _ca_sub(vehicle_count=5, vehicles=[Vehicle(vin="A"), Vehicle(vin="B")])
    result = await CrossFieldValidator().run(sub)
    assert any("vehicle_count" in f.field_path for f in result.findings)
    assert any(f.severity == "warning" for f in result.findings)


@pytest.mark.asyncio
async def test_vehicle_count_matches_no_finding():
    """vehicle_count=2, 2 vehicles listed → no vehicle_count finding."""
    sub = _ca_sub(vehicle_count=2, vehicles=[Vehicle(vin="A"), Vehicle(vin="B")])
    result = await CrossFieldValidator().run(sub)
    assert not any("vehicle_count" in f.field_path for f in result.findings)


@pytest.mark.asyncio
async def test_driver_count_mismatch():
    """driver_count=3, 1 driver listed → warning."""
    sub = _ca_sub(driver_count=3, drivers=[Driver(first_name="Jane")])
    result = await CrossFieldValidator().run(sub)
    assert any("driver_count" in f.field_path for f in result.findings)


# ---------------------------------------------------------------------------
# 5. policy_window
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_policy_window_past_effective():
    """Effective date 6 months ago → warning."""
    eff = _TODAY - timedelta(days=180)
    sub = CustomerSubmission(policy_dates=PolicyDates(effective_date=eff))
    result = await CrossFieldValidator().run(sub)
    assert any("past" in f.message.lower() for f in result.findings)


@pytest.mark.asyncio
async def test_policy_window_far_future_effective():
    """Effective date 2 years from now → warning."""
    eff = _TODAY + timedelta(days=800)
    sub = CustomerSubmission(policy_dates=PolicyDates(effective_date=eff))
    result = await CrossFieldValidator().run(sub)
    assert any("future" in f.message.lower() for f in result.findings)


# ---------------------------------------------------------------------------
# 6. license_expires_before_policy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_license_expired_before_policy():
    """Driver's license expired last year, policy starts this year → warning."""
    eff = _TODAY + timedelta(days=30)
    exp = _TODAY - timedelta(days=60)
    sub = _ca_sub(
        drivers=[Driver(first_name="Bob", license_expiration=exp)],
    )
    sub = CustomerSubmission(
        policy_dates=PolicyDates(effective_date=eff),
        lob_details=CommercialAutoDetails(
            drivers=[Driver(first_name="Bob", license_expiration=exp)]
        ),
    )
    result = await CrossFieldValidator().run(sub)
    assert any("license" in f.message.lower() and "expired" in f.message.lower()
               for f in result.findings)


# ---------------------------------------------------------------------------
# 7. trucking_carrier_has_vehicles
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trucking_dot_without_vehicles():
    """DOT number present but no vehicles → error."""
    sub = _ca_sub(fmcsa_dot_number="1234567", vehicles=[])
    result = await CrossFieldValidator().run(sub)
    assert any(f.severity == "error" and "vehicle" in f.message.lower() for f in result.findings)


# ---------------------------------------------------------------------------
# 8. contact_completeness
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_contact_missing_reach():
    """Contact has name but no phone or email → error."""
    sub = CustomerSubmission(contacts=[Contact(full_name="Jane Doe")])
    result = await CrossFieldValidator().run(sub)
    assert any(f.severity == "error" and "unreachable" in f.message.lower() for f in result.findings)


# ---------------------------------------------------------------------------
# 9. hazmat_requires_ca_lob / trucking naics with non-CA LOB
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hazmat_trucking_naics_non_ca_lob():
    """Trucking NAICS (484xxx) paired with GL LOB → warning."""
    sub = CustomerSubmission(
        naics_code="484110",
        lob_details=GeneralLiabilityDetails(),
    )
    result = await CrossFieldValidator().run(sub)
    assert any("trucking" in f.message.lower() for f in result.findings)
    assert any(f.severity == "warning" for f in result.findings)


# ---------------------------------------------------------------------------
# 10. garage_vs_mailing_state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_garage_vs_mailing_state_differ():
    """Vehicle garaged in CA, mailing address TX → info."""
    sub = CustomerSubmission(
        mailing_address=Address(state="TX"),
        lob_details=CommercialAutoDetails(
            vehicles=[Vehicle(garage_address=Address(state="CA"))],
        ),
    )
    result = await CrossFieldValidator().run(sub)
    assert any(f.severity == "info" and "garaged" in f.message.lower() for f in result.findings)


# ---------------------------------------------------------------------------
# 15. isolation: one check crashing doesn't break others
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_check_crash_doesnt_break_others():
    """If one check raises, the remaining 9 checks still execute."""
    original_checks = list(_ALL_CHECKS)

    def _crashing_check(sub):
        raise RuntimeError("simulated crash")

    # Inject a crashing check at position 0
    patched = [("crashing_check", _crashing_check)] + original_checks[1:]

    with patch("accord_ai.validation.cross_field._ALL_CHECKS", patched):
        # Use a submission that would trigger multiple other checks
        sub = CustomerSubmission(annual_revenue=Decimal("10000000"), full_time_employees=1)
        result = await CrossFieldValidator().run(sub)

    # Revenue check (position 1 in original, position 1 in patched) should still run
    assert result.success is True
    assert any("revenue" in f.message.lower() for f in result.findings)
