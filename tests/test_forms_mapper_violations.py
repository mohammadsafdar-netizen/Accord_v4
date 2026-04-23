"""Tests for ACORD 127 violation mapping + expanded driver roster (P10.S.7)."""
from __future__ import annotations

from datetime import date

import pytest

from accord_ai.forms import fill_form, load_form_spec
from accord_ai.forms.mapper import (
    _COMPUTED_RESOLVERS,
    _FORM_ALIASES,
    map_submission_to_form,
)
from accord_ai.schema import (
    CommercialAutoDetails,
    CustomerSubmission,
    Driver,
    Violation,
)


# ---------------------------------------------------------------------------
# Driver roster expansion: 8 → 13 slots on 127
# ---------------------------------------------------------------------------

def test_driver_first_name_expands_to_13_slots_on_127():
    aliases = _FORM_ALIASES["127"]
    expected = {f"Driver_GivenName_{c}" for c in "ABCDEFGHIJKLM"}
    assert expected <= set(aliases.keys())


def test_driver_all_stems_go_to_m_on_127():
    """Every driver scalar stem should have A-M alias entries."""
    aliases = _FORM_ALIASES["127"]
    stems = [
        "Driver_GivenName", "Driver_Surname", "Driver_OtherGivenNameInitial",
        "Driver_BirthDate", "Driver_LicenseNumberIdentifier",
        "Driver_LicensedStateOrProvinceCode", "Driver_ExperienceYearCount",
    ]
    for stem in stems:
        for letter in "ABCDEFGHIJKLM":
            key = f"{stem}_{letter}"
            assert key in aliases, f"{key} missing from _FORM_127"


def test_driver_slot_m_fills_from_index_12():
    """Driver slot M (index 12) must resolve to drivers[12]."""
    drivers = [Driver(first_name=f"D{i}") for i in range(13)]
    s = CustomerSubmission(
        lob_details=CommercialAutoDetails(drivers=drivers),
    )
    m = map_submission_to_form(s, "127")
    assert m["Driver_GivenName_A"] == "D0"
    assert m["Driver_GivenName_M"] == "D12"


def test_driver_slot_13_onwards_truncates():
    """Beyond 13 drivers the form has no physical row — truncate silently."""
    drivers = [Driver(first_name=f"D{i}") for i in range(20)]
    s = CustomerSubmission(
        lob_details=CommercialAutoDetails(drivers=drivers),
    )
    m = map_submission_to_form(s, "127")
    # N (index 13) would have been the next letter; assert it's not emitted.
    assert "Driver_GivenName_N" not in m


# ---------------------------------------------------------------------------
# Violation computed resolvers — registered
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", [
    "@violation.first.occurred_on",
    "@violation.first.description",
    "@violation.first.place_combined",
])
def test_violation_computed_resolvers_registered(key):
    assert key in _COMPUTED_RESOLVERS


# ---------------------------------------------------------------------------
# Violation aliases on 127
# ---------------------------------------------------------------------------

def test_violation_aliases_present_on_127():
    aliases = _FORM_ALIASES["127"]
    assert (
        aliases["AccidentConviction_TrafficViolationDate_A"]
        == "@violation.first.occurred_on"
    )
    assert (
        aliases["AccidentConviction_TrafficViolationDescription_A"]
        == "@violation.first.description"
    )
    assert (
        aliases["AccidentConviction_PlaceOfIncident_A"]
        == "@violation.first.place_combined"
    )


# ---------------------------------------------------------------------------
# First-violation resolution semantics
# ---------------------------------------------------------------------------

def test_violation_picks_first_driver_with_violation():
    """drivers[0] has none, drivers[1] has one → pick drivers[1]'s."""
    s = CustomerSubmission(
        lob_details=CommercialAutoDetails(drivers=[
            Driver(first_name="Alice"),                            # no violations
            Driver(first_name="Bob", violations=[
                Violation(
                    occurred_on=date(2024, 3, 15),
                    description="speeding",
                    location_city="Austin",
                    location_state="TX",
                ),
            ]),
            Driver(first_name="Carol", violations=[
                Violation(description="shadowed"),                 # never picked
            ]),
        ]),
    )
    m = map_submission_to_form(s, "127")
    assert m["AccidentConviction_TrafficViolationDate_A"]        == "03/15/2024"
    assert m["AccidentConviction_TrafficViolationDescription_A"] == "speeding"
    assert m["AccidentConviction_PlaceOfIncident_A"]             == "Austin, TX"


def test_violation_fills_from_first_violation_of_same_driver():
    """drivers[0] has 3 violations → take violations[0]."""
    s = CustomerSubmission(
        lob_details=CommercialAutoDetails(drivers=[
            Driver(first_name="Alice", violations=[
                Violation(description="first"),
                Violation(description="second"),
                Violation(description="third"),
            ]),
        ]),
    )
    m = map_submission_to_form(s, "127")
    assert m["AccidentConviction_TrafficViolationDescription_A"] == "first"


def test_violation_no_drivers_no_emit():
    s = CustomerSubmission(lob_details=CommercialAutoDetails())
    m = map_submission_to_form(s, "127")
    for key in (
        "AccidentConviction_TrafficViolationDate_A",
        "AccidentConviction_TrafficViolationDescription_A",
        "AccidentConviction_PlaceOfIncident_A",
    ):
        assert key not in m


def test_violation_no_violations_no_emit():
    s = CustomerSubmission(
        lob_details=CommercialAutoDetails(drivers=[
            Driver(first_name="Alice"),   # empty violations list
            Driver(first_name="Bob"),
        ]),
    )
    m = map_submission_to_form(s, "127")
    for key in (
        "AccidentConviction_TrafficViolationDate_A",
        "AccidentConviction_TrafficViolationDescription_A",
        "AccidentConviction_PlaceOfIncident_A",
    ):
        assert key not in m


def test_violation_place_combined_city_only():
    """Place renders even when state is empty — use whichever is present."""
    s = CustomerSubmission(
        lob_details=CommercialAutoDetails(drivers=[
            Driver(first_name="Alice", violations=[
                Violation(location_city="Austin"),   # no state
            ]),
        ]),
    )
    m = map_submission_to_form(s, "127")
    assert m["AccidentConviction_PlaceOfIncident_A"] == "Austin"


def test_violation_place_combined_state_only():
    s = CustomerSubmission(
        lob_details=CommercialAutoDetails(drivers=[
            Driver(first_name="Alice", violations=[
                Violation(location_state="TX"),
            ]),
        ]),
    )
    m = map_submission_to_form(s, "127")
    assert m["AccidentConviction_PlaceOfIncident_A"] == "TX"


def test_violation_place_combined_both_empty_omits():
    s = CustomerSubmission(
        lob_details=CommercialAutoDetails(drivers=[
            Driver(first_name="Alice", violations=[
                Violation(description="fine print"),  # only desc, no place
            ]),
        ]),
    )
    m = map_submission_to_form(s, "127")
    assert "AccidentConviction_PlaceOfIncident_A" not in m
    # Description still fills, proving we didn't short-circuit too hard
    assert (
        m["AccidentConviction_TrafficViolationDescription_A"]
        == "fine print"
    )


def test_violation_on_gl_lob_no_emit():
    """Non-CA LOB → no drivers attribute → no violation."""
    from accord_ai.schema import GeneralLiabilityDetails
    s = CustomerSubmission(lob_details=GeneralLiabilityDetails())
    m = map_submission_to_form(s, "127")
    assert "AccidentConviction_TrafficViolationDate_A" not in m


# ---------------------------------------------------------------------------
# Deferred widgets stay unmapped (documented decisions)
# ---------------------------------------------------------------------------

def test_violation_year_count_deferred():
    """AccidentConviction_ViolationYearCount_A is intentionally unmapped —
    if this test fails it means someone added a mapping. Double-check it
    matches a real schema field before removing this test."""
    assert (
        "AccidentConviction_ViolationYearCount_A"
        not in _FORM_ALIASES["127"]
    )


def test_driver_producer_identifier_deferred():
    assert (
        "AccidentConviction_DriverProducerIdentifier_A"
        not in _FORM_ALIASES["127"]
    )


# ---------------------------------------------------------------------------
# Registry integrity
# ---------------------------------------------------------------------------

def test_127_aliases_all_resolve_after_expansion():
    from accord_ai.forms.mapper import _lookup_resolver
    for acord_field, schema_key in _FORM_ALIASES["127"].items():
        try:
            _lookup_resolver(schema_key)
        except KeyError:
            pytest.fail(f"127/{acord_field} → unresolved {schema_key!r}")


def test_127_all_aliased_widgets_exist_in_spec():
    """Every new alias key must be a real widget on the blank PDF."""
    spec = load_form_spec("127")
    for acord_field in _FORM_ALIASES["127"]:
        assert acord_field in spec.fields, (
            f"{acord_field!r} not in 127 spec"
        )


# ---------------------------------------------------------------------------
# End-to-end PDF fill
# ---------------------------------------------------------------------------

def test_127_fills_end_to_end_with_violation():
    pytest.importorskip("fitz")
    s = CustomerSubmission(
        business_name="Acme Trucking",
        lob_details=CommercialAutoDetails(drivers=[
            Driver(first_name="Alice", last_name="Jones", violations=[
                Violation(
                    occurred_on=date(2024, 6, 1),
                    description="minor speeding",
                    location_city="Dallas",
                    location_state="TX",
                ),
            ]),
        ]),
    )
    mapped = map_submission_to_form(s, "127")
    pdf_bytes, res = fill_form("127", mapped)
    assert res.unknown_fields == ()
    assert res.error_count == 0
    assert pdf_bytes.startswith(b"%PDF-")
