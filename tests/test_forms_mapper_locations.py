"""Tests for location fan-out on ACORD 125 (P10.S.8)."""
from __future__ import annotations

from decimal import Decimal

import pytest

from accord_ai.forms import fill_form, load_form_spec
from accord_ai.forms.mapper import (
    _FORM_ALIASES,
    _lookup_resolver,
    map_submission_to_form,
)
from accord_ai.schema import (
    Address,
    CommercialAutoDetails,
    CustomerSubmission,
    GeneralLiabilityDetails,
    Location,
    WorkersCompDetails,
)


def _sub_with_locations(n: int, *, lob_details=None) -> CustomerSubmission:
    locs = [
        Location(
            address=Address(
                line_one=f"{100 + i} Main St",
                line_two=f"Suite {i}" if i % 2 else None,
                city=f"City{i}",
                state="TX",
                zip_code=f"7870{i}",
                county=f"County{i}",
            ),
            description=f"Ops at location {i}",
            annual_gross_receipts=Decimal(f"{(i + 1) * 500_000}"),
        )
        for i in range(n)
    ]
    return CustomerSubmission(
        business_name="Acme",
        locations=locs,
        lob_details=lob_details or CommercialAutoDetails(),
    )


# ---------------------------------------------------------------------------
# Topology: 8 stems × 4 slots = 32 new aliases on 125
# ---------------------------------------------------------------------------

_STEMS = (
    "CommercialStructure_PhysicalAddress_LineOne",
    "CommercialStructure_PhysicalAddress_LineTwo",
    "CommercialStructure_PhysicalAddress_CityName",
    "CommercialStructure_PhysicalAddress_StateOrProvinceCode",
    "CommercialStructure_PhysicalAddress_PostalCode",
    "CommercialStructure_PhysicalAddress_CountyName",
    "CommercialStructure_AnnualRevenueAmount",
    "BuildingOccupancy_OperationsDescription",
)


@pytest.mark.parametrize("stem", _STEMS)
def test_stem_has_all_four_slots_on_125(stem):
    aliases = _FORM_ALIASES["125"]
    for letter in "ABCD":
        assert f"{stem}_{letter}" in aliases, f"{stem}_{letter} missing"


def test_32_new_location_aliases_present():
    aliases = _FORM_ALIASES["125"]
    count = sum(
        1
        for stem in _STEMS
        for letter in "ABCD"
        if f"{stem}_{letter}" in aliases
    )
    assert count == 32


# ---------------------------------------------------------------------------
# Address fan-out across slots
# ---------------------------------------------------------------------------

def test_location_a_fills_from_index_0():
    m = map_submission_to_form(_sub_with_locations(1), "125")
    assert m["CommercialStructure_PhysicalAddress_LineOne_A"] == "100 Main St"
    assert m["CommercialStructure_PhysicalAddress_CityName_A"] == "City0"
    assert m["CommercialStructure_PhysicalAddress_StateOrProvinceCode_A"] == "TX"
    assert m["CommercialStructure_PhysicalAddress_PostalCode_A"] == "78700"
    assert m["CommercialStructure_PhysicalAddress_CountyName_A"] == "County0"


def test_all_four_slots_fill_independently():
    m = map_submission_to_form(_sub_with_locations(4), "125")
    assert m["CommercialStructure_PhysicalAddress_LineOne_A"] == "100 Main St"
    assert m["CommercialStructure_PhysicalAddress_LineOne_B"] == "101 Main St"
    assert m["CommercialStructure_PhysicalAddress_LineOne_C"] == "102 Main St"
    assert m["CommercialStructure_PhysicalAddress_LineOne_D"] == "103 Main St"


def test_line_two_optional_per_slot():
    """Only odd-indexed locations have line_two. Missing ones must be omitted,
    not rendered as 'None'."""
    m = map_submission_to_form(_sub_with_locations(4), "125")
    assert "CommercialStructure_PhysicalAddress_LineTwo_A" not in m
    assert m["CommercialStructure_PhysicalAddress_LineTwo_B"] == "Suite 1"
    assert "CommercialStructure_PhysicalAddress_LineTwo_C" not in m
    assert m["CommercialStructure_PhysicalAddress_LineTwo_D"] == "Suite 3"


# ---------------------------------------------------------------------------
# Per-slot non-address fields
# ---------------------------------------------------------------------------

def test_annual_revenue_formatted_as_money_per_slot():
    m = map_submission_to_form(_sub_with_locations(2), "125")
    assert m["CommercialStructure_AnnualRevenueAmount_A"] == "500,000"
    assert m["CommercialStructure_AnnualRevenueAmount_B"] == "1,000,000"


def test_operations_description_per_slot():
    m = map_submission_to_form(_sub_with_locations(2), "125")
    assert m["BuildingOccupancy_OperationsDescription_A"] == "Ops at location 0"
    assert m["BuildingOccupancy_OperationsDescription_B"] == "Ops at location 1"


# ---------------------------------------------------------------------------
# Edge cases: empty / truncation / cross-LOB
# ---------------------------------------------------------------------------

def test_empty_locations_list_emits_nothing():
    s = CustomerSubmission(
        business_name="Acme", locations=[],
        lob_details=CommercialAutoDetails(),
    )
    m = map_submission_to_form(s, "125")
    for stem in _STEMS:
        for letter in "ABCD":
            assert f"{stem}_{letter}" not in m


def test_locations_beyond_slot_d_truncate_silently():
    """Form 125 only has 4 physical rows; extras never emit."""
    s = _sub_with_locations(7)
    m = map_submission_to_form(s, "125")
    # A-D fill; E / F / G don't exist as widget names
    assert m["CommercialStructure_PhysicalAddress_CityName_D"] == "City3"
    for bad_letter in "EFG":
        assert (
            f"CommercialStructure_PhysicalAddress_CityName_{bad_letter}"
            not in m
        )


def test_location_with_partial_address_only_emits_set_fields():
    s = CustomerSubmission(
        business_name="Acme",
        locations=[Location(address=Address(city="Austin", state="TX"))],
        lob_details=CommercialAutoDetails(),
    )
    m = map_submission_to_form(s, "125")
    assert m["CommercialStructure_PhysicalAddress_CityName_A"] == "Austin"
    assert m["CommercialStructure_PhysicalAddress_StateOrProvinceCode_A"] == "TX"
    assert "CommercialStructure_PhysicalAddress_LineOne_A" not in m


@pytest.mark.parametrize("lob_details", [
    CommercialAutoDetails(),
    GeneralLiabilityDetails(),
    WorkersCompDetails(),
])
def test_locations_fill_regardless_of_lob(lob_details):
    """locations[] is top-level on CustomerSubmission — mapping works for
    every LOB, not just commercial auto."""
    s = _sub_with_locations(1, lob_details=lob_details)
    m = map_submission_to_form(s, "125")
    assert m["CommercialStructure_PhysicalAddress_CityName_A"] == "City0"


# ---------------------------------------------------------------------------
# Shared resolver invariant — all 8 stems have one canonical resolver each
# ---------------------------------------------------------------------------

def test_each_location_stem_has_one_canonical_resolver_per_slot():
    """For each (stem, letter) pair, the schema_key points to exactly one
    registered resolver in _SCHEMA_RESOLVERS. A typo in array_aliases would
    manifest as a KeyError on _lookup_resolver, caught here with the
    specific (stem, letter) that broke."""
    aliases = _FORM_ALIASES["125"]
    for stem in _STEMS:
        for letter in "ABCD":
            widget = f"{stem}_{letter}"
            schema_key = aliases.get(widget)
            assert schema_key is not None, f"{widget} not in aliases"
            try:
                _lookup_resolver(schema_key)
            except KeyError:
                pytest.fail(f"{widget} → unresolved {schema_key!r}")


def test_location_schema_keys_reference_locations_at_root():
    """Paths should start with 'locations[N].' — not 'lob_details.locations'
    — because Location is a top-level list on CustomerSubmission."""
    aliases = _FORM_ALIASES["125"]
    for stem in _STEMS:
        for letter in "ABCD":
            key = aliases[f"{stem}_{letter}"]
            assert key.startswith("locations["), (
                f"{stem}_{letter} → {key!r} — expected top-level locations path"
            )


# ---------------------------------------------------------------------------
# End-to-end PDF fill
# ---------------------------------------------------------------------------

def test_location_fill_end_to_end_on_125():
    pytest.importorskip("fitz")
    s = _sub_with_locations(3)
    mapped = map_submission_to_form(s, "125")
    pdf_bytes, res = fill_form("125", mapped)
    assert res.unknown_fields == ()
    assert res.error_count == 0
    assert pdf_bytes.startswith(b"%PDF-")


def test_every_new_location_widget_exists_in_spec():
    """Belt-and-suspenders: the ground-truth audit will catch this too, but
    keeping a localized invariant here points at the specific offender."""
    spec = load_form_spec("125")
    aliases = _FORM_ALIASES["125"]
    for stem in _STEMS:
        for letter in "ABCD":
            widget = f"{stem}_{letter}"
            if widget in aliases:
                assert widget in spec.fields, (
                    f"{widget} missing from 125 spec"
                )


# ---------------------------------------------------------------------------
# Deferred widgets stay unmapped (paper trail)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("widget", [
    "CommercialStructure_InsuredInterest_OwnerIndicator_A",
    "CommercialStructure_InsuredInterest_TenantIndicator_A",
    "CommercialStructure_RiskLocation_InsideCityLimitsIndicator_A",
    "CommercialStructure_RiskLocation_OutsideCityLimitsIndicator_A",
    "CommercialStructure_Question_ABBCode_A",
    "BuildingOccupancy_OccupiedArea_A",
    "BuildingOccupancy_OpenToPublicArea_A",
])
def test_deferred_location_widget_stays_unmapped(widget):
    """These widgets are intentionally unmapped; schema doesn't yet have
    the fields they'd bind to. If you want one in, ADD a Location schema
    field first, then delete this test row."""
    assert widget not in _FORM_ALIASES["125"]
