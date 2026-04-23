"""Tests for location fan-out on ACORD 130/131/159 (P10.S.9)."""
from __future__ import annotations

import pytest

from accord_ai.forms import fill_form
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
        Location(address=Address(
            line_one=f"{100 + i} Main St",
            line_two=f"Suite {i}" if i % 2 else None,
            city=f"City{i}",
            state="TX",
            zip_code=f"75{i:03d}",
            county=f"County{i}",
        ))
        for i in range(n)
    ]
    return CustomerSubmission(
        business_name="Acme",
        locations=locs,
        lob_details=lob_details or CommercialAutoDetails(),
    )


# ---------------------------------------------------------------------------
# Form 130 — Location_ prefix, 3 slots, all 6 leaves
# ---------------------------------------------------------------------------

_130_LEAVES = (
    "LineOne", "LineTwo", "CityName",
    "StateOrProvinceCode", "PostalCode", "CountyName",
)


@pytest.mark.parametrize("leaf", _130_LEAVES)
@pytest.mark.parametrize("letter", "ABC")
def test_130_address_alias_present(leaf, letter):
    assert (
        f"Location_PhysicalAddress_{leaf}_{letter}"
        in _FORM_ALIASES["130"]
    )


def test_130_exactly_3_slots_per_leaf():
    for leaf in _130_LEAVES:
        slots = [
            letter for letter in "ABCDEFGHIJKLMN"
            if f"Location_PhysicalAddress_{leaf}_{letter}"
            in _FORM_ALIASES["130"]
        ]
        assert slots == list("ABC"), f"{leaf}: expected A-C, got {slots}"


def test_130_fills_three_slots():
    s = _sub_with_locations(3)
    m = map_submission_to_form(s, "130")
    for i, letter in enumerate("ABC"):
        assert (
            m[f"Location_PhysicalAddress_LineOne_{letter}"]
            == f"{100 + i} Main St"
        )
        assert m[f"Location_PhysicalAddress_CityName_{letter}"] == f"City{i}"
        assert (
            m[f"Location_PhysicalAddress_StateOrProvinceCode_{letter}"]
            == "TX"
        )
        assert (
            m[f"Location_PhysicalAddress_CountyName_{letter}"]
            == f"County{i}"
        )


def test_130_line_two_optional_per_slot():
    s = _sub_with_locations(3)
    m = map_submission_to_form(s, "130")
    assert "Location_PhysicalAddress_LineTwo_A" not in m   # index 0, even → None
    assert m["Location_PhysicalAddress_LineTwo_B"] == "Suite 1"
    assert "Location_PhysicalAddress_LineTwo_C" not in m


def test_130_truncates_beyond_slot_c():
    s = _sub_with_locations(5)
    m = map_submission_to_form(s, "130")
    assert m["Location_PhysicalAddress_LineOne_A"] == "100 Main St"
    assert m["Location_PhysicalAddress_LineOne_C"] == "102 Main St"
    assert "Location_PhysicalAddress_LineOne_D" not in m
    assert "Location_PhysicalAddress_LineOne_E" not in m


# ---------------------------------------------------------------------------
# Form 131 — CommercialStructure_ prefix, 6 slots, NO LineTwo / CountyName
# ---------------------------------------------------------------------------

_131_LEAVES = ("LineOne", "CityName", "StateOrProvinceCode", "PostalCode")


@pytest.mark.parametrize("leaf", _131_LEAVES)
@pytest.mark.parametrize("letter", "ABCDEF")
def test_131_address_alias_present(leaf, letter):
    assert (
        f"CommercialStructure_PhysicalAddress_{leaf}_{letter}"
        in _FORM_ALIASES["131"]
    )


@pytest.mark.parametrize("absent_leaf", ("LineTwo", "CountyName"))
def test_131_omits_absent_leaves(absent_leaf):
    """131 doesn't have LineTwo or CountyName widgets — must NOT emit aliases."""
    for letter in "ABCDEF":
        key = f"CommercialStructure_PhysicalAddress_{absent_leaf}_{letter}"
        assert key not in _FORM_ALIASES["131"]


def test_131_fills_all_six_slots():
    s = _sub_with_locations(6)
    m = map_submission_to_form(s, "131")
    for i, letter in enumerate("ABCDEF"):
        assert (
            m[f"CommercialStructure_PhysicalAddress_LineOne_{letter}"]
            == f"{100 + i} Main St"
        )
        assert (
            m[f"CommercialStructure_PhysicalAddress_CityName_{letter}"]
            == f"City{i}"
        )


def test_131_truncates_beyond_slot_f():
    s = _sub_with_locations(10)
    m = map_submission_to_form(s, "131")
    assert (
        m["CommercialStructure_PhysicalAddress_LineOne_F"] == "105 Main St"
    )
    assert "CommercialStructure_PhysicalAddress_LineOne_G" not in m


# ---------------------------------------------------------------------------
# Form 159 — CommercialStructure_ prefix, 14 slots, 4 leaves
# ---------------------------------------------------------------------------

_159_LEAVES = _131_LEAVES


@pytest.mark.parametrize("leaf", _159_LEAVES)
@pytest.mark.parametrize("letter", "ABCDEFGHIJKLMN")
def test_159_address_alias_present(leaf, letter):
    assert (
        f"CommercialStructure_PhysicalAddress_{leaf}_{letter}"
        in _FORM_ALIASES["159"]
    )


@pytest.mark.parametrize("absent_leaf", ("LineTwo", "CountyName"))
def test_159_omits_absent_leaves(absent_leaf):
    for letter in "ABCDEFGHIJKLMN":
        key = f"CommercialStructure_PhysicalAddress_{absent_leaf}_{letter}"
        assert key not in _FORM_ALIASES["159"]


def test_159_fills_slot_n_from_index_13():
    """14th location (letter N, index 13) must fill correctly — the edge."""
    s = _sub_with_locations(14)
    m = map_submission_to_form(s, "159")
    assert (
        m["CommercialStructure_PhysicalAddress_LineOne_N"] == "113 Main St"
    )
    assert (
        m["CommercialStructure_PhysicalAddress_CityName_N"] == "City13"
    )


def test_159_truncates_beyond_slot_n():
    """Locations beyond the 14-slot cap silently truncate."""
    s = _sub_with_locations(20)
    m = map_submission_to_form(s, "159")
    assert (
        m["CommercialStructure_PhysicalAddress_LineOne_N"] == "113 Main St"
    )
    # letter O (index 14) is beyond the form's 14-slot cap — must not emit
    assert "CommercialStructure_PhysicalAddress_LineOne_O" not in m


# ---------------------------------------------------------------------------
# Cross-form resolver reuse — the "collect once, fill everywhere" invariant
# ---------------------------------------------------------------------------

def test_locations_0_city_shares_one_resolver_across_all_four_forms():
    """Same schema path = same canonical resolver across 125/130/131/159.
    Proves register_scalar's idempotency is working for array expansion."""
    schema_key = "locations[0].address.city"
    resolver_ids = set()
    for form, widget in (
        ("125", "CommercialStructure_PhysicalAddress_CityName_A"),
        ("130", "Location_PhysicalAddress_CityName_A"),
        ("131", "CommercialStructure_PhysicalAddress_CityName_A"),
        ("159", "CommercialStructure_PhysicalAddress_CityName_A"),
    ):
        alias_value = _FORM_ALIASES[form][widget]
        assert alias_value == schema_key, (
            f"{form}/{widget} → {alias_value!r}, expected {schema_key!r}"
        )
        resolver_ids.add(id(_lookup_resolver(alias_value)))
    assert len(resolver_ids) == 1, (
        f"schema key {schema_key!r} resolves to {len(resolver_ids)} distinct "
        "resolver objects across forms — cross-form reuse broken"
    )


def test_location_3_resolver_shared_between_131_and_159():
    """130 only has slots A-C (index 0-2); 131 goes to F (5); 159 goes to N
    (13). For slot index 3 specifically, confirm 131 and 159 share one
    resolver (130 doesn't have it)."""
    schema_key = "locations[3].address.city"
    resolver_ids = set()
    for form, widget in (
        ("131", "CommercialStructure_PhysicalAddress_CityName_D"),
        ("159", "CommercialStructure_PhysicalAddress_CityName_D"),
    ):
        alias_value = _FORM_ALIASES[form][widget]
        assert alias_value == schema_key
        resolver_ids.add(id(_lookup_resolver(alias_value)))
    assert len(resolver_ids) == 1


# ---------------------------------------------------------------------------
# Cross-LOB fill (locations is top-level, works regardless of LOB)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("form_number", ("130", "131", "159"))
@pytest.mark.parametrize("lob_details", [
    CommercialAutoDetails(),
    GeneralLiabilityDetails(),
    WorkersCompDetails(),
])
def test_locations_fill_regardless_of_lob(form_number, lob_details):
    s = _sub_with_locations(1, lob_details=lob_details)
    m = map_submission_to_form(s, form_number)
    # All three forms share CityName in their A slot — proves top-level
    # locations path works for every LOB combination.
    if form_number == "130":
        assert m["Location_PhysicalAddress_CityName_A"] == "City0"
    else:
        assert (
            m["CommercialStructure_PhysicalAddress_CityName_A"] == "City0"
        )


# ---------------------------------------------------------------------------
# Integrity: every new alias resolves
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("form_number", ("130", "131", "159"))
def test_all_location_aliases_resolve(form_number):
    for widget, schema_key in _FORM_ALIASES[form_number].items():
        try:
            _lookup_resolver(schema_key)
        except KeyError:
            pytest.fail(f"{form_number}/{widget} → unresolved {schema_key!r}")


# ---------------------------------------------------------------------------
# End-to-end PDF fill
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("form_number,slot_count", [
    ("130", 3), ("131", 6), ("159", 14),
])
def test_location_fill_end_to_end(form_number, slot_count):
    pytest.importorskip("fitz")
    s = _sub_with_locations(slot_count)
    mapped = map_submission_to_form(s, form_number)
    pdf_bytes, res = fill_form(form_number, mapped)
    assert res.unknown_fields == ()
    assert res.error_count == 0
    assert pdf_bytes.startswith(b"%PDF-")
