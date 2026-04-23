"""Shared test fixtures — fully-valid submissions per LOB.

Every factory returns a ``CustomerSubmission`` that satisfies every
critical field in ``accord_ai.harness.critical_fields`` for that LOB,
so ``SchemaJudge().evaluate(fixture).passed`` is ``True``. Tests that
need a submission to perturb should start from one of these factories
rather than constructing a minimal one — otherwise the expanded judge
will refuse to pass and FakeEngine-queued refinement loops exhaust.

Usage:
    from tests._fixtures import valid_ca, valid_gl, valid_wc

    sub = valid_ca().model_copy(update={"business_name": "New Name"})
    assert SchemaJudge().evaluate(sub).passed
"""
from __future__ import annotations

from datetime import date
from typing import Any, Dict

from accord_ai.schema import (
    Address,
    CommercialAutoDetails,
    Contact,
    CustomerSubmission,
    GeneralLiabilityDetails,
    PayrollByClass,
    PolicyDates,
    WorkersCompCoverage,
    WorkersCompDetails,
)


def valid_ca() -> CustomerSubmission:
    """Commercial-auto submission satisfying all 23 v3-critical fields.

    v3's CA plugin requires ``named_insured.contact.phone`` and
    ``named_insured.contact.email`` — v4's path is ``contacts[0].phone``
    and ``contacts[0].email``. Root ``phone``/``email`` stay unset here
    because v3 never made them critical; they're optional v4-only fields.
    """
    return CustomerSubmission(
        business_name="Acme Trucking",
        mailing_address=Address(
            line_one="123 Main", city="Austin",
            state="TX", zip_code="78701",
        ),
        entity_type="llc",
        ein="12-3456789",
        business_start_date=date(2010, 1, 1),
        policy_status="new",
        policy_dates=PolicyDates(effective_date=date(2026, 5, 1)),
        contacts=[Contact(
            full_name="Jane Doe",
            phone="512-555-0100",
            email="ops@acme.test",
        )],
        full_time_employees=10,
        nature_of_business="fleet trucking",
        lob_details=CommercialAutoDetails(
            vehicle_count=3,
            radius_of_operations="50 miles",
            hazmat=False,
            fleet_use_type="service",
            fleet_for_hire=True,
            states_of_operation=["TX", "OK"],
            trailer_interchange=False,
            driver_training=True,
        ),
    )


def valid_gl() -> CustomerSubmission:
    """General-liability submission satisfying all v3-critical fields."""
    return CustomerSubmission(
        business_name="GlobeX",
        mailing_address=Address(
            line_one="500 Oak", city="Dallas",
            state="TX", zip_code="75201",
        ),
        entity_type="corporation",
        ein="98-7654321",
        business_start_date=date(2015, 1, 1),
        policy_status="new",
        policy_dates=PolicyDates(effective_date=date(2026, 3, 1)),
        nature_of_business="office services",
        operations_description="professional consulting",
        annual_revenue=1_000_000,
        lob_details=GeneralLiabilityDetails(),
    )


def valid_wc() -> CustomerSubmission:
    """Workers-comp submission satisfying all ACORD-130-minimum fields."""
    return CustomerSubmission(
        business_name="Initech",
        mailing_address=Address(
            line_one="1 Park", city="Austin",
            state="TX", zip_code="78701",
        ),
        entity_type="corporation",
        ein="11-2233445",
        business_start_date=date(2012, 1, 1),
        policy_status="new",
        policy_dates=PolicyDates(effective_date=date(2026, 4, 1)),
        lob_details=WorkersCompDetails(
            payroll_by_class=[PayrollByClass(class_code="8810")],
            coverage=WorkersCompCoverage(
                employers_liability_per_accident=500_000,
            ),
        ),
    )


def valid_ca_dict() -> Dict[str, Any]:
    """``valid_ca()`` serialized for FakeEngine queues.

    FakeEngine accepts either a dict (it json.dumps on delivery) or a
    JSON string. Returning exclude_none=True cuts the 750 B empty-field
    noise so the prompt-diff in the extractor and refiner tests stays
    readable.
    """
    return valid_ca().model_dump(mode="json", exclude_none=True)


def valid_gl_dict() -> Dict[str, Any]:
    return valid_gl().model_dump(mode="json", exclude_none=True)


def valid_wc_dict() -> Dict[str, Any]:
    return valid_wc().model_dump(mode="json", exclude_none=True)
