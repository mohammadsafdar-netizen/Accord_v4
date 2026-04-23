from datetime import date

import pytest
from pydantic import ValidationError

from accord_ai.schema import (
    AdditionalInterest,
    Address,
    Classification,
    CommercialAutoCoverage,
    CommercialAutoDetails,
    CustomerSubmission,
    Driver,
    GeneralLiabilityCoverage,
    GeneralLiabilityDetails,
    LossHistory,
    PayrollByClass,
    PolicyDates,
    Vehicle,
    WorkersCompCoverage,
    WorkersCompDetails,
)


# --- Address ---

def test_address_empty_is_valid():
    assert Address().city is None


def test_address_builds_from_dict():
    a = Address(line_one="123 Main St", city="Detroit", state="MI", zip_code="48201")
    assert a.city == "Detroit"


def test_address_whitespace_stripped():
    assert Address(city="  Detroit  ").city == "Detroit"


def test_address_drops_unknown_field_silently():
    """Schema is extra='ignore' so LLM placement errors don't reject the
    whole submission. Unknown attributes are dropped; known ones stay."""
    a = Address(city="Detroit", cty="Typo")
    assert a.city == "Detroit"
    assert not hasattr(a, "cty")


# --- Driver ---

def test_driver_empty_is_valid():
    assert Driver().first_name is None


def test_driver_parses_iso_date_string():
    d = Driver(first_name="Alice", last_name="Nguyen", date_of_birth="1985-03-21")
    assert d.date_of_birth == date(1985, 3, 21)


def test_driver_rejects_bad_date():
    with pytest.raises(ValidationError):
        Driver(date_of_birth="not-a-date")


def test_driver_drops_unknown_field_silently():
    d = Driver(first_name="Alice", favorite_color="red")
    assert d.first_name == "Alice"
    assert not hasattr(d, "favorite_color")


def test_driver_whitespace_stripped_on_nested_model():
    d = Driver(first_name="  Alice  ", last_name="\tNguyen\n")
    assert d.first_name == "Alice"
    assert d.last_name == "Nguyen"


# --- Vehicle ---

def test_vehicle_empty_is_valid():
    assert Vehicle().year is None


def test_vehicle_with_nested_garage_address():
    v = Vehicle(year=2022, make="Ford", vin="1FTFW1E50NFA12345",
                garage_address={"city": "Warren", "state": "MI"})
    assert v.garage_address.state == "MI"


def test_vehicle_drops_unknown_field_silently():
    v = Vehicle(year=2022, colour="blue")
    assert v.year == 2022
    assert not hasattr(v, "colour")


# --- AdditionalInterest / LossHistory ---

def test_additional_interest_with_address():
    ai = AdditionalInterest(name="Ally Bank", role="lienholder",
                            address={"city": "Detroit", "state": "MI"})
    assert ai.role == "lienholder"
    assert ai.address.city == "Detroit"


def test_loss_history_parses_date():
    lh = LossHistory(date_of_loss="2024-03-15", type_of_loss="collision", amount_paid=8500)
    assert lh.date_of_loss == date(2024, 3, 15)


# --- Classification / PayrollByClass / PolicyDates ---

def test_classification_builds():
    c = Classification(class_code="91580", naics_code="484110", annual_gross_receipts=2_500_000)
    assert c.naics_code == "484110"


def test_payroll_by_class_builds():
    p = PayrollByClass(class_code="8810", description="Clerical", payroll=125_000,
                       employee_count=3, state="MI")
    assert p.payroll == 125_000


def test_policy_dates_parses():
    pd = PolicyDates(effective_date="2026-05-01", expiration_date="2027-05-01")
    assert pd.effective_date == date(2026, 5, 1)


# --- Per-LOB coverages ---

def test_commercial_auto_coverage_has_comp_coll():
    c = CommercialAutoCoverage(liability_limit_csl=1_000_000, comp_deductible=500, coll_deductible=1000)
    assert c.comp_deductible == 500


def test_general_liability_coverage_has_aggregate():
    c = GeneralLiabilityCoverage(each_occurrence_limit=1_000_000,
                                 general_aggregate_limit=2_000_000, claims_made_basis=False)
    assert c.general_aggregate_limit == 2_000_000


def test_workers_comp_coverage_has_employers_liability():
    c = WorkersCompCoverage(employers_liability_per_accident=1_000_000,
                            employers_liability_per_employee=1_000_000,
                            employers_liability_per_policy=1_000_000)
    assert c.employers_liability_per_accident == 1_000_000


# --- Per-LOB details ---

def test_commercial_auto_details_has_drivers_and_vehicles():
    d = CommercialAutoDetails(drivers=[{"first_name": "Alice"}],
                              vehicles=[{"year": 2020, "make": "Ford"}],
                              driver_count=1, vehicle_count=1)
    assert d.lob == "commercial_auto"
    assert d.vehicles[0].make == "Ford"


def test_general_liability_details_has_classifications():
    d = GeneralLiabilityDetails(employee_count=12,
                                classifications=[{"class_code": "91580", "annual_gross_receipts": 500_000}])
    assert d.lob == "general_liability"


def test_workers_comp_details_has_payroll():
    d = WorkersCompDetails(experience_mod=0.95, owner_exclusion=True,
                           payroll_by_class=[{"class_code": "8810", "payroll": 100_000}])
    assert d.lob == "workers_comp"
    assert d.experience_mod == 0.95


# --- CustomerSubmission + discriminated union ---

def test_submission_empty_has_no_lob_details():
    sub = CustomerSubmission()
    assert sub.lob_details is None
    assert sub.additional_interests == []


def test_submission_commercial_auto_from_dict():
    sub = CustomerSubmission(
        business_name="Acme Trucking",
        lob_details={"lob": "commercial_auto",
                     "drivers": [{"first_name": "Alice"}],
                     "vehicles": [{"year": 2022, "vin": "1FTFW1E50NFA12345"}]},
    )
    assert isinstance(sub.lob_details, CommercialAutoDetails)


def test_submission_general_liability_from_dict():
    sub = CustomerSubmission(
        lob_details={"lob": "general_liability", "employee_count": 25,
                     "classifications": [{"class_code": "91580"}]},
    )
    assert isinstance(sub.lob_details, GeneralLiabilityDetails)


def test_submission_workers_comp_from_dict():
    sub = CustomerSubmission(
        lob_details={"lob": "workers_comp", "experience_mod": 1.05,
                     "payroll_by_class": [{"class_code": "3632", "payroll": 500_000}]},
    )
    assert isinstance(sub.lob_details, WorkersCompDetails)


def test_submission_rejects_lob_details_without_discriminator():
    with pytest.raises(ValidationError):
        CustomerSubmission(lob_details={"drivers": []})


def test_submission_drops_mismatched_lob_fields():
    """experience_mod lives on WorkersCompDetails. Placing it under
    commercial_auto is an LLM field-placement error — drop silently
    instead of rejecting the whole submission."""
    sub = CustomerSubmission(
        lob_details={
            "lob": "commercial_auto",
            "experience_mod": 0.9,          # WC-only field, misplaced
            "vehicles": [{"year": 2022}],   # real CA field, should survive
        },
    )
    assert sub.lob_details.lob == "commercial_auto"
    assert not hasattr(sub.lob_details, "experience_mod")
    assert sub.lob_details.vehicles[0].year == 2022


def test_submission_rejects_unknown_lob_value():
    """lob='aviation' — not in the discriminated union."""
    with pytest.raises(ValidationError):
        CustomerSubmission(lob_details={"lob": "aviation"})


def test_submission_universal_fields_work_with_any_lob():
    sub = CustomerSubmission(
        business_name="Acme",
        policy_dates={"effective_date": "2026-05-01", "expiration_date": "2027-05-01"},
        additional_interests=[{"name": "Ally Bank", "role": "lienholder"}],
        loss_history=[{"date_of_loss": "2024-03-15", "type_of_loss": "collision", "amount_paid": 8500}],
        lob_details={"lob": "workers_comp", "experience_mod": 0.95},
    )
    assert sub.policy_dates.effective_date == date(2026, 5, 1)
    assert sub.additional_interests[0].role == "lienholder"


def test_submission_whitespace_stripped_on_business_name():
    assert CustomerSubmission(business_name="  Acme Trucking  ").business_name == "Acme Trucking"


def test_submission_drops_unknown_top_level_field():
    """Typo'd field name (business_naem) is dropped; the rest of the
    submission still constructs cleanly."""
    sub = CustomerSubmission(business_name="Acme", business_naem="typo")
    assert sub.business_name == "Acme"
    assert not hasattr(sub, "business_naem")


def test_submission_exclude_none_drops_empties():
    sub = CustomerSubmission(business_name="Acme")
    data = sub.model_dump(exclude_none=True)
    assert data["business_name"] == "Acme"
    assert "ein" not in data
    assert "lob_details" not in data
    assert data["additional_interests"] == []


def test_submission_roundtrip_preserves_lob_details():
    original = CustomerSubmission(
        business_name="Acme",
        ein="12-3456789",
        policy_dates={"effective_date": "2026-05-01"},
        lob_details={"lob": "commercial_auto",
                     "drivers": [{"first_name": "A", "date_of_birth": "1990-01-01"}],
                     "vehicles": [{"year": 2020, "vin": "1FTFW1E50NFA12345"}],
                     "coverage": {"liability_limit_csl": 1_000_000, "comp_deductible": 500}},
    )
    data = original.model_dump(mode="json")
    restored = CustomerSubmission.model_validate(data)
    assert isinstance(restored.lob_details, CommercialAutoDetails)
    assert restored.lob_details.drivers[0].date_of_birth == date(1990, 1, 1)
    assert restored.lob_details.coverage.comp_deductible == 500


# --- validate_assignment ---

def test_mutation_strips_whitespace():
    """str_strip_whitespace applies on assignment when validate_assignment=True."""
    sub = CustomerSubmission(business_name="Acme")
    sub.business_name = "  Pivoted Acme  "
    assert sub.business_name == "Pivoted Acme"


def test_mutation_to_incompatible_type_raises():
    """validate_assignment catches bad writes — can't assign a string to List[Driver]."""
    sub = CustomerSubmission()
    with pytest.raises(ValidationError):
        sub.drivers = "not-a-list"
