"""Tests for array expansion in the mapper (P10.A.3b)."""
from __future__ import annotations

from datetime import date

import pytest

from accord_ai.forms import (
    fill_form,
    load_form_spec,
    map_submission_to_form,
)
from accord_ai.forms.mapper import (
    _FORM_ALIASES,
    _SCHEMA_RESOLVERS,
    _resolve,
    array_aliases,
)
from accord_ai.schema import (
    CommercialAutoDetails,
    CustomerSubmission,
    Driver,
    PayrollByClass,
    Vehicle,
    WorkersCompDetails,
)


# ---------------------------------------------------------------------------
# Path resolver: list indexing
# ---------------------------------------------------------------------------

def test_resolve_list_index():
    s = CustomerSubmission(
        lob_details=CommercialAutoDetails(
            drivers=[Driver(first_name="Alice"), Driver(first_name="Bob")],
        ),
    )
    assert _resolve(s, "lob_details.drivers[0].first_name") == "Alice"
    assert _resolve(s, "lob_details.drivers[1].first_name") == "Bob"


def test_resolve_list_index_out_of_range():
    s = CustomerSubmission(
        lob_details=CommercialAutoDetails(drivers=[Driver(first_name="Alice")]),
    )
    assert _resolve(s, "lob_details.drivers[5].first_name") is None


def test_resolve_list_index_on_empty_list():
    s = CustomerSubmission(lob_details=CommercialAutoDetails(drivers=[]))
    assert _resolve(s, "lob_details.drivers[0].first_name") is None


def test_resolve_list_index_none_list():
    s = CustomerSubmission()   # no lob_details → drivers missing
    assert _resolve(s, "lob_details.drivers[0].first_name") is None


def test_resolve_invalid_index_segment_raises():
    with pytest.raises(ValueError):
        _resolve(CustomerSubmission(), "drivers[abc].first_name")


# ---------------------------------------------------------------------------
# array_aliases helper
# ---------------------------------------------------------------------------

def test_array_fields_generates_a_through_letter():
    out = array_aliases(
        "Driver_GivenName", "lob_details.drivers", "first_name", max_count=3,
    )
    assert set(out.keys()) == {
        "Driver_GivenName_A", "Driver_GivenName_B", "Driver_GivenName_C",
    }


def test_array_fields_rejects_out_of_range_max_count():
    with pytest.raises(ValueError):
        array_aliases("X", "foo", "bar", max_count=0)
    with pytest.raises(ValueError):
        array_aliases("X", "foo", "bar", max_count=27)


def test_array_fields_resolver_returns_none_for_missing_slot():
    out = array_aliases(
        "Driver_GivenName", "lob_details.drivers", "first_name", max_count=3,
    )
    # array_aliases returns schema keys; resolvers live in _SCHEMA_RESOLVERS.
    s = CustomerSubmission(
        lob_details=CommercialAutoDetails(
            drivers=[Driver(first_name="Alice")],
        ),
    )
    assert out["Driver_GivenName_A"] == "lob_details.drivers[0].first_name"
    assert _SCHEMA_RESOLVERS[out["Driver_GivenName_A"]](s) == "Alice"
    assert _SCHEMA_RESOLVERS[out["Driver_GivenName_B"]](s) is None
    assert _SCHEMA_RESOLVERS[out["Driver_GivenName_C"]](s) is None


# ---------------------------------------------------------------------------
# Form 127: driver roster
# ---------------------------------------------------------------------------

def test_form_127_maps_drivers_in_order():
    s = CustomerSubmission(
        lob_details=CommercialAutoDetails(drivers=[
            Driver(
                first_name="Alice", last_name="Jones",
                date_of_birth=date(1988, 6, 1),
                license_number="D12345", license_state="TX",
                years_experience=10,
            ),
            Driver(
                first_name="Bob", last_name="Smith",
                date_of_birth=date(1990, 2, 14),
                license_number="D67890", license_state="TX",
            ),
        ]),
    )
    m = map_submission_to_form(s, "127")
    assert m["Driver_GivenName_A"] == "Alice"
    assert m["Driver_Surname_A"] == "Jones"
    assert m["Driver_BirthDate_A"] == "06/01/1988"
    assert m["Driver_LicenseNumberIdentifier_A"] == "D12345"
    assert m["Driver_LicensedStateOrProvinceCode_A"] == "TX"
    assert m["Driver_ExperienceYearCount_A"] == "10"

    assert m["Driver_GivenName_B"] == "Bob"
    assert "Driver_ExperienceYearCount_B" not in m   # years_experience unset


def test_form_127_empty_driver_list_emits_nothing():
    s = CustomerSubmission(lob_details=CommercialAutoDetails(drivers=[]))
    m = map_submission_to_form(s, "127")
    assert m == {}


def test_form_127_truncates_beyond_max_count():
    # 127 expanded from 8 → 13 slots (A-M) in P10.S.7 to match the form's
    # physical row count. Feed 20 drivers, assert A-M emit and N+ truncate.
    drivers = [Driver(first_name=f"D{i}") for i in range(20)]
    s = CustomerSubmission(lob_details=CommercialAutoDetails(drivers=drivers))
    m = map_submission_to_form(s, "127")
    emitted = {k: v for k, v in m.items() if k.startswith("Driver_GivenName_")}
    assert set(emitted.keys()) == {
        f"Driver_GivenName_{c}" for c in "ABCDEFGHIJKLM"
    }
    assert emitted["Driver_GivenName_A"] == "D0"
    assert emitted["Driver_GivenName_M"] == "D12"


# ---------------------------------------------------------------------------
# Form 129: vehicle schedule
# ---------------------------------------------------------------------------

def test_form_129_maps_vehicles():
    s = CustomerSubmission(
        lob_details=CommercialAutoDetails(vehicles=[
            Vehicle(
                year=2024, make="Freightliner", model="Cascadia",
                vin="1FUJGLDR0LLAB1234", body_type="Tractor",
            ),
            Vehicle(
                year=2022, make="Peterbilt", model="579",
                vin="2FUJGLDR1LLCD5678",
            ),
        ]),
    )
    m = map_submission_to_form(s, "129")
    assert m["Vehicle_ModelYear_A"] == "2024"
    assert m["Vehicle_ManufacturersName_A"] == "Freightliner"
    assert m["Vehicle_ModelName_A"] == "Cascadia"
    assert m["Vehicle_VINIdentifier_A"] == "1FUJGLDR0LLAB1234"
    assert m["Vehicle_BodyCode_A"] == "Tractor"
    assert m["Vehicle_ModelYear_B"] == "2022"
    assert "Vehicle_BodyCode_B" not in m


# ---------------------------------------------------------------------------
# Form 130: WC rate classes
# ---------------------------------------------------------------------------

def test_form_130_maps_payroll_classes():
    s = CustomerSubmission(
        lob_details=WorkersCompDetails(payroll_by_class=[
            PayrollByClass(
                class_code="8810", description="Clerical",
                payroll=250_000, employee_count=5,
            ),
            PayrollByClass(
                class_code="5403", description="Carpentry",
                payroll=480_000, employee_count=8,
            ),
        ]),
    )
    m = map_submission_to_form(s, "130")
    assert m["WorkersCompensation_RateClass_ClassificationCode_A"] == "8810"
    assert m["WorkersCompensation_RateClass_DutiesDescription_A"] == "Clerical"
    assert m["WorkersCompensation_RateClass_RemunerationAmount_A"] == "250,000"
    assert m["WorkersCompensation_RateClass_FullTimeEmployeeCount_A"] == "5"
    assert m["WorkersCompensation_RateClass_RemunerationAmount_B"] == "480,000"


def test_form_130_14_classes_all_slots_used():
    classes = [
        PayrollByClass(class_code=f"C{i:04d}", payroll=1000 * (i + 1))
        for i in range(14)
    ]
    s = CustomerSubmission(
        lob_details=WorkersCompDetails(payroll_by_class=classes),
    )
    m = map_submission_to_form(s, "130")
    stem = "WorkersCompensation_RateClass_ClassificationCode"
    emitted = {k: v for k, v in m.items() if k.startswith(stem + "_")}
    assert set(emitted.keys()) == {f"{stem}_{c}" for c in "ABCDEFGHIJKLMN"}


# ---------------------------------------------------------------------------
# Registry invariant still holds with the new expanded keys
# ---------------------------------------------------------------------------

def test_every_mapped_array_field_exists_in_registry_spec():
    """Letter-suffixed keys must be real widgets on the corresponding blank.
    Repeats the scalar invariant but is the most load-bearing test for 3b
    because a stem typo expands to 14 bogus fields, not just one."""
    for form_number, mapping in _FORM_ALIASES.items():
        if not mapping:
            continue
        spec = load_form_spec(form_number)
        for acord_field in mapping:
            assert acord_field in spec.fields, (
                f"form {form_number}: mapped field {acord_field!r} not in spec"
            )


# ---------------------------------------------------------------------------
# End-to-end: array mapping → filler
# ---------------------------------------------------------------------------

def test_form_127_drivers_fill_end_to_end():
    pytest.importorskip("fitz")
    s = CustomerSubmission(
        lob_details=CommercialAutoDetails(drivers=[
            Driver(first_name="Alice", last_name="Jones", license_state="TX"),
            Driver(first_name="Bob",   last_name="Smith", license_state="CA"),
        ]),
    )
    mapped = map_submission_to_form(s, "127")
    pdf_bytes, res = fill_form("127", mapped)
    assert res.unknown_fields == ()
    assert res.filled_count == len(mapped)
    assert pdf_bytes.startswith(b"%PDF-")


def test_form_129_vehicles_fill_end_to_end():
    pytest.importorskip("fitz")
    s = CustomerSubmission(
        policy_dates=None,
        lob_details=CommercialAutoDetails(vehicles=[
            Vehicle(
                year=2024, make="Freightliner", model="Cascadia",
                vin="1FUJGLDR0LLAB1234",
            ),
        ]),
    )
    mapped = map_submission_to_form(s, "129")
    pdf_bytes, res = fill_form("129", mapped)
    assert res.unknown_fields == ()
    assert res.filled_count == len(mapped)
