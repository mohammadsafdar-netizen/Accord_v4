"""Schema port verification (P10.S.1).

These tests lock down the surface so a future refactor can't silently
drop a field. They don't test behavior — they test shape.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from accord_ai.schema import (
    Address,
    Classification,
    CommercialAutoDetails,
    Contact,
    CustomerSubmission,
    Driver,
    GeneralLiabilityDetails,
    Location,
    LossHistory,
    PayrollByClass,
    PolicyDates,
    PriorInsurance,
    Producer,
    Vehicle,
    Violation,
    WorkersCompDetails,
)


# --- Empty construction still works (migration safety) -----------------------

def test_empty_submission_builds():
    s = CustomerSubmission()
    assert s.business_name is None
    assert s.contacts == []
    assert s.loss_history == []
    assert s.locations == []
    assert s.additional_interests == []
    assert s.lob_details is None


def test_existing_minimal_fields_still_accept_same_args():
    """Smoke test: pre-10.S.1 construction patterns keep working."""
    s = CustomerSubmission(
        business_name="Acme",
        mailing_address=Address(city="Austin"),
        lob_details=CommercialAutoDetails(
            drivers=[Driver(first_name="Alice")],
            vehicles=[Vehicle(year=2024, make="Ford")],
        ),
    )
    assert s.business_name == "Acme"
    assert s.lob_details.drivers[0].first_name == "Alice"


# --- New classes --------------------------------------------------------------

def test_contact_shape():
    c = Contact(
        full_name="Jane Doe", phone="512-555-0100",
        email="j@x.test", role="CFO",
    )
    assert c.full_name == "Jane Doe"


def test_producer_carries_address():
    p = Producer(
        agency_name="Broker Co",
        mailing_address=Address(city="Dallas"),
    )
    assert p.mailing_address.city == "Dallas"


def test_violation_shape():
    v = Violation(
        occurred_on=date(2024, 1, 15),
        type="accident",
        description="at-fault",
    )
    assert v.type == "accident"
    assert v.occurred_on == date(2024, 1, 15)


def test_prior_insurance_shape():
    p = PriorInsurance(
        carrier_name="Old Co", premium_amount=Decimal("12500.00"),
    )
    assert p.carrier_name == "Old Co"


def test_location_shape():
    loc = Location(
        address=Address(state="TX"),
        annual_payroll=Decimal("250000"),
    )
    assert loc.address.state == "TX"


# --- Driver expansion --------------------------------------------------------

def test_driver_full_expansion():
    d = Driver(
        first_name="Alice", last_name="Jones",
        date_of_birth=date(1988, 6, 1),
        sex="F", marital_status="M",
        license_state="TX", licensed_year=2008,
        hire_date=date(2015, 3, 1),
        relationship="employee",
        mvr_status="clean",
        pct_use=Decimal("100"),
        vehicle_assigned=0,
        violations=[Violation(type="accident", description="minor")],
    )
    assert d.pct_use == Decimal("100")
    assert len(d.violations) == 1


def test_driver_enum_validation():
    with pytest.raises(Exception):
        Driver(sex="X")    # not a valid Sex literal


# --- Vehicle expansion -------------------------------------------------------

def test_vehicle_coverage_shape():
    from accord_ai.schema import VehicleCoverage
    c = VehicleCoverage(
        liability=True, collision=True, comprehensive=True,
        collision_deductible_amount=Decimal("1000"),
        comprehensive_deductible_amount=Decimal("500"),
        uninsured_motorists=False,
    )
    assert c.liability is True
    assert c.collision_deductible_amount == Decimal("1000")
    assert c.uninsured_motorists is False


def test_vehicle_carries_coverage():
    from accord_ai.schema import Vehicle, VehicleCoverage
    v = Vehicle(
        vin="1FUJGLDR0LLAB1234",
        coverage=VehicleCoverage(liability=True),
    )
    assert v.coverage.liability is True


def test_vehicle_full_expansion():
    v = Vehicle(
        year=2024, make="Freightliner", model="Cascadia",
        vin="1FUJGLDR0LLAB1234",
        gvw=80000, cost_new=Decimal("185000"),
        use_type="commercial", vehicle_type="commercial",
        radius_of_travel=500, seating_capacity=2,
        registration_state="TX",
    )
    assert v.gvw == 80000
    assert v.cost_new == Decimal("185000")


# --- CustomerSubmission expansion -------------------------------------------

def test_submission_business_classification_fields():
    s = CustomerSubmission(
        business_name="Acme",
        naics_code="484121",
        sic_code="4213",
        entity_type="llc",
        operations_description="long-haul trucking",
        business_start_date=date(2010, 1, 1),
        years_in_business=15,
        website="https://acme.example",
        full_time_employees=25,
        part_time_employees=3,
        annual_revenue=Decimal("5000000"),
        annual_payroll=Decimal("1250000"),
    )
    assert s.entity_type == "llc"
    assert s.years_in_business == 15


def test_submission_policy_framing():
    s = CustomerSubmission(
        policy_number="POL-123",
        policy_status="new",
        billing_plan="agency",
        payment_plan="monthly",
    )
    assert s.policy_status == "new"


def test_submission_entity_type_enum_validation():
    with pytest.raises(Exception):
        CustomerSubmission(entity_type="unknown_shape")


def test_producer_on_submission():
    s = CustomerSubmission(producer=Producer(agency_name="Broker Co"))
    assert s.producer.agency_name == "Broker Co"


def test_locations_list():
    s = CustomerSubmission(locations=[
        Location(address=Address(state="TX")),
        Location(address=Address(state="CA")),
    ])
    assert [loc.address.state for loc in s.locations] == ["TX", "CA"]


# --- WC prior_insurance ------------------------------------------------------

def test_wc_prior_insurance_list():
    wc = WorkersCompDetails(
        prior_insurance=[
            PriorInsurance(carrier_name=f"Carrier {i}")
            for i in range(5)
        ],
    )
    assert len(wc.prior_insurance) == 5


# --- JSON round-trip ---------------------------------------------------------

def test_full_submission_json_roundtrip():
    """A fully-populated submission serializes + re-parses without loss."""
    s = CustomerSubmission(
        business_name="Acme",
        entity_type="corporation",
        annual_revenue=Decimal("1000000"),
        business_start_date=date(2010, 1, 1),
        contacts=[Contact(full_name="CFO", role="CFO")],
        producer=Producer(agency_name="Broker Co"),
        lob_details=CommercialAutoDetails(
            drivers=[Driver(
                first_name="Alice", date_of_birth=date(1988, 1, 1),
                violations=[Violation(type="accident")],
            )],
        ),
    )
    dumped = s.model_dump_json()
    restored = CustomerSubmission.model_validate_json(dumped)
    assert restored.business_name == "Acme"
    assert restored.producer.agency_name == "Broker Co"
    assert restored.lob_details.drivers[0].violations[0].type == "accident"


# --- Schema surface snapshot -------------------------------------------------

def test_json_schema_surface_count():
    """Lock the schema surface so a silent drop surfaces in CI."""
    sch = CustomerSubmission.model_json_schema()
    assert len(sch["properties"]) >= 25    # root-level fields
    # $defs floor bumped 14 → 15 after VehicleCoverage added in P10.S.10a.
    assert len(sch["$defs"])      >= 15    # nested classes


# --- Dead-field cull verification --------------------------------------------
#
# Documents which v3 fields were intentionally dropped — keyed to the specific
# class the cull applies to (so legitimate uses like Contact.full_name don't
# false-positive). Remove a row here if you intentionally re-add a field.

@pytest.mark.parametrize("model_cls,field_name,reason", [
    (CustomerSubmission,     "fax",                "v3 BusinessInfo.fax — dead input"),
    (Producer,               "fax",                "v3 ProducerInfo.fax — dead input"),
    (CustomerSubmission,     "deposit_amount",     "carrier-set, not user-provided"),
    (CustomerSubmission,     "estimated_premium",  "carrier-set, not user-provided"),
    (Driver,                 "ssn_or_tax_id",      "PII, never mapped in v3 either"),
    (Driver,                 "full_name",          "redundant with first+last"),
    (Driver,                 "license_type",       "ACORD inconsistency — fold into license_number"),
    (Vehicle,                "modified_equipment", "never mapped to an ACORD widget"),
    (Vehicle,                "deductible_collision",
        "deduplicated with VehicleCoverage.collision_deductible_amount in "
        "P10.S.10a / P10.0.f.2 to eliminate LLM extractor ambiguity"),
    (Vehicle,                "deductible_comprehensive",
        "deduplicated with VehicleCoverage.comprehensive_deductible_amount in "
        "P10.S.10a / P10.0.f.2 to eliminate LLM extractor ambiguity"),
    (CustomerSubmission,     "experience_mod",     "lives on WorkersCompDetails instead"),
    (CustomerSubmission,     "cyber_info",         "deprecated LOB path"),
])
def test_culled_fields_absent(model_cls, field_name, reason):
    assert field_name not in model_cls.model_fields, (
        f"{field_name!r} re-appeared on {model_cls.__name__} ({reason})"
    )
