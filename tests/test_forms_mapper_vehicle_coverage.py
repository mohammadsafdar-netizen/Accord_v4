"""Tests for per-vehicle coverage fan-out on ACORD 127/129 (P10.S.10a)."""
from __future__ import annotations

from decimal import Decimal

import pytest

from accord_ai.forms import fill_form
from accord_ai.forms.mapper import (
    _FORM_ALIASES,
    _lookup_resolver,
    map_submission_to_form,
)
from accord_ai.schema import (
    CommercialAutoDetails,
    CustomerSubmission,
    Vehicle,
    VehicleCoverage,
)


_CORE_STEMS = (
    "Vehicle_Coverage_LiabilityIndicator",
    "Vehicle_Coverage_CollisionIndicator",
    "Vehicle_Coverage_ComprehensiveIndicator",
    "Vehicle_Collision_DeductibleAmount",
    "Vehicle_Coverage_ComprehensiveOrSpecifiedCauseOfLossDeductibleAmount",
    "Vehicle_Coverage_MedicalPaymentsIndicator",
    "Vehicle_Coverage_UninsuredMotoristsIndicator",
    "Vehicle_Coverage_UnderinsuredMotoristsIndicator",
    "Vehicle_Coverage_TowingAndLabourIndicator",
    "Vehicle_Coverage_RentalReimbursementIndicator",
)


def _vehicle_with_full_coverage(vin="1FUJGLDR0LLAB1234") -> Vehicle:
    return Vehicle(
        year=2024, make="Freightliner", model="Cascadia", vin=vin,
        coverage=VehicleCoverage(
            liability=True,
            collision=True,
            comprehensive=True,
            collision_deductible_amount=Decimal("1000"),
            comprehensive_deductible_amount=Decimal("500"),
            medical_payments=True,
            uninsured_motorists=True,
            underinsured_motorists=False,    # deliberately False
            towing_labour=True,
            rental_reimbursement=False,      # deliberately False
        ),
    )


def _sub(vehicles) -> CustomerSubmission:
    return CustomerSubmission(
        business_name="Acme",
        lob_details=CommercialAutoDetails(vehicles=vehicles),
    )


# ---------------------------------------------------------------------------
# Alias topology
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stem", _CORE_STEMS)
@pytest.mark.parametrize("letter", "ABCD")
def test_127_alias_present(stem, letter):
    assert f"{stem}_{letter}" in _FORM_ALIASES["127"]


@pytest.mark.parametrize("stem", _CORE_STEMS)
@pytest.mark.parametrize("letter", "ABCDE")
def test_129_alias_present(stem, letter):
    assert f"{stem}_{letter}" in _FORM_ALIASES["129"]


def test_127_has_40_new_coverage_aliases():
    aliases = _FORM_ALIASES["127"]
    count = sum(
        1 for s in _CORE_STEMS for L in "ABCD" if f"{s}_{L}" in aliases
    )
    assert count == 40


def test_129_has_50_new_coverage_aliases():
    aliases = _FORM_ALIASES["129"]
    count = sum(
        1 for s in _CORE_STEMS for L in "ABCDE" if f"{s}_{L}" in aliases
    )
    assert count == 50


# ---------------------------------------------------------------------------
# Fill behavior — checkbox semantics
# ---------------------------------------------------------------------------

def test_true_coverage_emits_checkbox_true():
    s = _sub([_vehicle_with_full_coverage()])
    m = map_submission_to_form(s, "129")
    assert m["Vehicle_Coverage_LiabilityIndicator_A"] is True
    assert m["Vehicle_Coverage_CollisionIndicator_A"] is True
    assert m["Vehicle_Coverage_TowingAndLabourIndicator_A"] is True


def test_false_coverage_emits_nothing():
    """fmt_checkbox returns None for False — widget stays unchecked."""
    s = _sub([_vehicle_with_full_coverage()])
    m = map_submission_to_form(s, "129")
    assert "Vehicle_Coverage_UnderinsuredMotoristsIndicator_A" not in m
    assert "Vehicle_Coverage_RentalReimbursementIndicator_A" not in m


def test_unset_coverage_emits_nothing():
    s = _sub([Vehicle(
        vin="1FUJGLDR0LLAB1234",
        coverage=VehicleCoverage(liability=True),
    )])
    m = map_submission_to_form(s, "129")
    assert m["Vehicle_Coverage_LiabilityIndicator_A"] is True
    # All other coverages unset → None in schema → omitted
    for stem in _CORE_STEMS:
        if stem == "Vehicle_Coverage_LiabilityIndicator":
            continue
        key = f"{stem}_A"
        if "Indicator" in stem:
            assert key not in m


def test_no_coverage_block_emits_nothing():
    s = _sub([Vehicle(vin="1FUJGLDR0LLAB1234")])   # coverage=None
    m = map_submission_to_form(s, "129")
    for stem in _CORE_STEMS:
        assert f"{stem}_A" not in m


# ---------------------------------------------------------------------------
# Deductible amounts — money formatting
# ---------------------------------------------------------------------------

def test_collision_deductible_formatted_as_money():
    s = _sub([_vehicle_with_full_coverage()])
    m = map_submission_to_form(s, "129")
    assert m["Vehicle_Collision_DeductibleAmount_A"] == "1,000"


def test_comprehensive_deductible_formatted_as_money():
    s = _sub([_vehicle_with_full_coverage()])
    m = map_submission_to_form(s, "129")
    assert (
        m["Vehicle_Coverage_ComprehensiveOrSpecifiedCauseOfLossDeductibleAmount_A"]
        == "500"
    )


def test_zero_deductible_still_emits():
    """Decimal('0') is not None — must still render (0 is a valid deductible)."""
    s = _sub([Vehicle(
        vin="1FUJGLDR0LLAB1234",
        coverage=VehicleCoverage(collision_deductible_amount=Decimal("0")),
    )])
    m = map_submission_to_form(s, "129")
    assert m["Vehicle_Collision_DeductibleAmount_A"] == "0"


# ---------------------------------------------------------------------------
# Multi-vehicle — each slot independent
# ---------------------------------------------------------------------------

def test_multi_vehicle_independent_coverages():
    s = _sub([
        Vehicle(vin="V1", coverage=VehicleCoverage(liability=True, collision=False)),
        Vehicle(vin="V2", coverage=VehicleCoverage(liability=False, collision=True)),
        Vehicle(vin="V3", coverage=VehicleCoverage(
            liability=True, collision=True,
            collision_deductible_amount=Decimal("2500"),
        )),
    ])
    m = map_submission_to_form(s, "129")
    assert m["Vehicle_Coverage_LiabilityIndicator_A"] is True
    assert "Vehicle_Coverage_LiabilityIndicator_B" not in m    # False → omitted
    assert m["Vehicle_Coverage_LiabilityIndicator_C"] is True

    assert "Vehicle_Coverage_CollisionIndicator_A" not in m
    assert m["Vehicle_Coverage_CollisionIndicator_B"] is True
    assert m["Vehicle_Coverage_CollisionIndicator_C"] is True

    assert m["Vehicle_Collision_DeductibleAmount_C"] == "2,500"


def test_vehicles_truncate_beyond_slot_count():
    """127 caps at 4 slots, 129 caps at 5. Index 5+ must not emit."""
    vehicles = [
        _vehicle_with_full_coverage(vin=f"V{i:017d}") for i in range(7)
    ]
    s = _sub(vehicles)
    m127 = map_submission_to_form(s, "127")
    m129 = map_submission_to_form(s, "129")
    assert m127.get("Vehicle_Coverage_LiabilityIndicator_D") is True
    assert "Vehicle_Coverage_LiabilityIndicator_E" not in m127
    assert m129.get("Vehicle_Coverage_LiabilityIndicator_E") is True
    assert "Vehicle_Coverage_LiabilityIndicator_F" not in m129


# ---------------------------------------------------------------------------
# Cross-form resolver reuse — the core "collect once" guarantee
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("schema_path", [
    "lob_details.vehicles[0].coverage.liability",
    "lob_details.vehicles[0].coverage.collision_deductible_amount",
    "lob_details.vehicles[3].coverage.medical_payments",
])
def test_vehicle_coverage_resolver_shared_between_127_and_129(schema_path):
    """Every coverage field that exists on BOTH forms must resolve to the
    same resolver object — the "fill everywhere" invariant for vehicle
    coverage."""
    forms_using = []
    for form in ("127", "129"):
        for widget, key in _FORM_ALIASES[form].items():
            if key == schema_path:
                forms_using.append((form, widget))

    # Both forms should have exactly one widget mapping to this path
    # (slots 0-3 exist on 127 and 129).
    assert len(forms_using) == 2, (
        f"{schema_path!r} should map on both 127 and 129, got {forms_using}"
    )
    resolver_ids = {
        id(_lookup_resolver(_FORM_ALIASES[f][w])) for f, w in forms_using
    }
    assert len(resolver_ids) == 1


def test_slot_4_only_exists_on_129():
    """vehicles[4] (letter E) is on 129 only — 127 caps at D."""
    aliases_127 = _FORM_ALIASES["127"]
    aliases_129 = _FORM_ALIASES["129"]
    for stem in _CORE_STEMS:
        assert f"{stem}_E" not in aliases_127
        assert f"{stem}_E" in aliases_129


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("form_number", ("127", "129"))
def test_all_coverage_aliases_resolve(form_number):
    aliases = _FORM_ALIASES[form_number]
    for widget, key in aliases.items():
        if any(widget.startswith(s) for s in _CORE_STEMS):
            try:
                _lookup_resolver(key)
            except KeyError:
                pytest.fail(f"{form_number}/{widget} → unresolved {key!r}")


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("form_number", ("127", "129"))
def test_coverage_fill_end_to_end(form_number):
    pytest.importorskip("fitz")
    s = _sub([
        _vehicle_with_full_coverage(),
        _vehicle_with_full_coverage(vin="V2"),
    ])
    mapped = map_submission_to_form(s, form_number)
    pdf_bytes, res = fill_form(form_number, mapped)
    assert res.unknown_fields == ()
    assert res.error_count == 0
    assert pdf_bytes.startswith(b"%PDF-")
