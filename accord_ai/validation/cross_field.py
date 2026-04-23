"""Cross-field invariant validator — finalize-only.

One validator, 10 named check functions. Each check is pure, isolated, and
≤30 LOC. Exceptions in one check are caught and logged — the other 9 still run.
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from decimal import Decimal
from typing import Callable, List, Optional

from accord_ai.validation.types import (
    PrefillPatch,
    ValidationFinding,
    ValidationResult,
)

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ca(submission):
    """Return CommercialAutoDetails if the submission is a CA LOB, else None."""
    ld = getattr(submission, "lob_details", None)
    if ld is not None and getattr(ld, "lob", None) == "commercial_auto":
        return ld
    return None


# ---------------------------------------------------------------------------
# Individual checks (pure sync functions)
# ---------------------------------------------------------------------------


def _check_driver_age_experience(sub) -> List[ValidationFinding]:
    ca = _ca(sub)
    if ca is None:
        return []
    today = date.today()
    findings = []
    for i, driver in enumerate(ca.drivers or []):
        if not (driver.date_of_birth and driver.years_experience):
            continue
        age = (today - driver.date_of_birth).days // 365
        max_possible = max(0, age - 16)
        if driver.years_experience > max_possible:
            findings.append(ValidationFinding(
                validator="cross_field",
                field_path=f"lob_details.drivers[{i}].years_experience",
                severity="warning",
                message=(
                    f"Driver {i}: claimed {driver.years_experience} years experience "
                    f"but only {max_possible} are possible given age {age}"
                ),
                details={
                    "claimed_experience": driver.years_experience,
                    "age": age,
                    "max_possible": max_possible,
                },
            ))
    return findings


def _check_revenue_per_employee(sub) -> List[ValidationFinding]:
    revenue = getattr(sub, "annual_revenue", None)
    ft = getattr(sub, "full_time_employees", None) or 0
    pt = getattr(sub, "part_time_employees", None) or 0
    count = ft + pt
    if not revenue or not count:
        return []
    rpe = float(revenue) / count
    if rpe > 2_000_000 or rpe < 15_000:
        direction = "high" if rpe > 2_000_000 else "low"
        return [ValidationFinding(
            validator="cross_field",
            field_path="annual_revenue",
            severity="info",
            message=(
                f"Revenue per employee is unusually {direction}: "
                f"${rpe:,.0f}/employee ({count} employees, ${float(revenue):,.0f} revenue)"
            ),
            details={"revenue_per_employee": round(rpe, 2), "employee_count": count},
        )]
    return []


def _check_vehicle_count(sub) -> List[ValidationFinding]:
    ca = _ca(sub)
    if ca is None or ca.vehicle_count is None:
        return []
    actual = len(ca.vehicles or [])
    if ca.vehicle_count != actual:
        return [ValidationFinding(
            validator="cross_field",
            field_path="lob_details.vehicle_count",
            severity="warning",
            message=(
                f"Declared vehicle_count={ca.vehicle_count} but "
                f"{actual} vehicle(s) listed in vehicles[]"
            ),
            details={"declared": ca.vehicle_count, "actual": actual},
        )]
    return []


def _check_driver_count(sub) -> List[ValidationFinding]:
    ca = _ca(sub)
    if ca is None or ca.driver_count is None:
        return []
    actual = len(ca.drivers or [])
    if ca.driver_count != actual:
        return [ValidationFinding(
            validator="cross_field",
            field_path="lob_details.driver_count",
            severity="warning",
            message=(
                f"Declared driver_count={ca.driver_count} but "
                f"{actual} driver(s) listed in drivers[]"
            ),
            details={"declared": ca.driver_count, "actual": actual},
        )]
    return []


def _check_policy_window(sub) -> List[ValidationFinding]:
    pd = getattr(sub, "policy_dates", None)
    if pd is None:
        return []
    today = date.today()
    findings = []

    eff = getattr(pd, "effective_date", None)
    exp = getattr(pd, "expiration_date", None)

    if eff is not None:
        if eff < today - timedelta(days=30):
            findings.append(ValidationFinding(
                validator="cross_field",
                field_path="policy_dates.effective_date",
                severity="warning",
                message=f"Policy effective date {eff} is more than 30 days in the past",
                details={"effective_date": str(eff), "today": str(today)},
            ))
        elif eff > today + timedelta(days=365):
            findings.append(ValidationFinding(
                validator="cross_field",
                field_path="policy_dates.effective_date",
                severity="warning",
                message=f"Policy effective date {eff} is more than 1 year in the future",
                details={"effective_date": str(eff), "today": str(today)},
            ))

    if eff is not None and exp is not None:
        if exp < eff:
            findings.append(ValidationFinding(
                validator="cross_field",
                field_path="policy_dates.expiration_date",
                severity="error",
                message=f"Policy expiration {exp} is before effective date {eff}",
                details={"effective_date": str(eff), "expiration_date": str(exp)},
            ))
        elif exp > eff + timedelta(days=730):
            findings.append(ValidationFinding(
                validator="cross_field",
                field_path="policy_dates.expiration_date",
                severity="warning",
                message=f"Policy term exceeds 2 years ({eff} → {exp})",
                details={"effective_date": str(eff), "expiration_date": str(exp)},
            ))

    return findings


def _check_license_policy_overlap(sub) -> List[ValidationFinding]:
    pd = getattr(sub, "policy_dates", None)
    if pd is None:
        return []
    eff = getattr(pd, "effective_date", None)
    if eff is None:
        return []

    ca = _ca(sub)
    if ca is None:
        return []

    findings = []
    for i, driver in enumerate(ca.drivers or []):
        exp = getattr(driver, "license_expiration", None)
        if exp is not None and exp < eff:
            name = " ".join(filter(None, [driver.first_name, driver.last_name])) or f"driver[{i}]"
            findings.append(ValidationFinding(
                validator="cross_field",
                field_path=f"lob_details.drivers[{i}].license_expiration",
                severity="warning",
                message=(
                    f"{name}'s license expired {exp} before policy start {eff}"
                ),
                details={"license_expiration": str(exp), "effective_date": str(eff)},
            ))
    return findings


def _check_trucking_fleet(sub) -> List[ValidationFinding]:
    ca = _ca(sub)
    if ca is None:
        return []
    if ca.fmcsa_dot_number and len(ca.vehicles or []) == 0:
        return [ValidationFinding(
            validator="cross_field",
            field_path="lob_details.vehicles",
            severity="error",
            message=(
                f"DOT carrier (#{ca.fmcsa_dot_number}) must declare at least one vehicle"
            ),
            details={"dot_number": ca.fmcsa_dot_number},
        )]
    return []


def _check_contact_has_name_and_reach(sub) -> List[ValidationFinding]:
    contacts = getattr(sub, "contacts", None) or []
    if not contacts:
        return []
    reachable = any(
        c.full_name and (c.phone or c.email)
        for c in contacts
    )
    if not reachable:
        return [ValidationFinding(
            validator="cross_field",
            field_path="contacts",
            severity="error",
            message="No contact has both a name and a phone or email — insured is unreachable",
            details={"contact_count": len(contacts)},
        )]
    return []


def _check_hazmat_lob(sub) -> List[ValidationFinding]:
    """Warn if hazmat=True is set on a non-CA LOB, or trucking NAICS with non-CA LOB."""
    ld = getattr(sub, "lob_details", None)
    lob = getattr(ld, "lob", None) if ld is not None else None
    findings = []

    # Schema guard: hazmat field only exists on CommercialAutoDetails
    if getattr(ld, "hazmat", None) is True and lob != "commercial_auto":
        findings.append(ValidationFinding(
            validator="cross_field",
            field_path="lob_details.hazmat",
            severity="warning",
            message="hazmat=True is set but LOB is not commercial_auto (hazmat is a CA concept)",
            details={"lob": lob},
        ))

    # Cross-field: trucking NAICS paired with non-CA LOB
    naics = (getattr(sub, "naics_code", None) or "").strip()
    if naics.startswith("484") and lob is not None and lob != "commercial_auto":
        findings.append(ValidationFinding(
            validator="cross_field",
            field_path="naics_code",
            severity="warning",
            message=(
                f"Trucking NAICS {naics!r} but LOB is {lob!r} — "
                "trucking operations typically require commercial_auto coverage"
            ),
            details={"naics_code": naics, "lob": lob},
        ))

    return findings


def _check_address_state_sanity(sub) -> List[ValidationFinding]:
    """Info if any vehicle's garage state differs from the mailing address state."""
    mailing = getattr(sub, "mailing_address", None)
    if not mailing or not getattr(mailing, "state", None):
        return []

    ca = _ca(sub)
    if ca is None or not ca.vehicles:
        return []

    garage_states = {
        v.garage_address.state
        for v in ca.vehicles
        if getattr(v, "garage_address", None) and v.garage_address.state
    }
    out_of_state = garage_states - {mailing.state}
    if out_of_state:
        return [ValidationFinding(
            validator="cross_field",
            field_path="lob_details.vehicles[].garage_address.state",
            severity="info",
            message=(
                f"Vehicle(s) garaged in {out_of_state} "
                f"but mailing address is {mailing.state}"
            ),
            details={"garage_states": sorted(out_of_state), "mailing_state": mailing.state},
        )]
    return []


# ---------------------------------------------------------------------------
# Registry + Validator class
# ---------------------------------------------------------------------------

CheckFn = Callable[[object], List[ValidationFinding]]

_ALL_CHECKS: List[tuple] = [
    ("driver_age_vs_experience",      _check_driver_age_experience),
    ("revenue_per_employee",          _check_revenue_per_employee),
    ("declared_vs_actual_vehicles",   _check_vehicle_count),
    ("declared_vs_actual_drivers",    _check_driver_count),
    ("policy_dates_reasonable",       _check_policy_window),
    ("license_expires_before_policy", _check_license_policy_overlap),
    ("trucking_carrier_has_vehicles", _check_trucking_fleet),
    ("contact_completeness",          _check_contact_has_name_and_reach),
    ("hazmat_requires_ca_lob",        _check_hazmat_lob),
    ("garage_vs_mailing_state",       _check_address_state_sanity),
]


class CrossFieldValidator:
    """10 cross-field sanity checks — always-active, finalize-only."""

    name: str = "cross_field"
    applicable_fields: tuple = ("*",)
    inline_eligible: bool = False

    async def run(self, submission: object) -> ValidationResult:
        from datetime import datetime, timezone
        started = datetime.now(tz=timezone.utc)
        t0 = time.monotonic()
        findings: List[ValidationFinding] = []

        for check_name, fn in _ALL_CHECKS:
            try:
                findings.extend(fn(submission) or [])
            except Exception as exc:
                _logger.warning("cross_field_check_crashed check=%s error=%s", check_name, exc)

        duration_ms = (time.monotonic() - t0) * 1000
        return ValidationResult(
            validator=self.name,
            ran_at=started,
            duration_ms=duration_ms,
            success=True,
            findings=findings,
        )

    async def prefill(self, submission: object, just_extracted: dict) -> Optional[PrefillPatch]:
        return None
