"""Tests for the L3 scorer (P10.S.11a)."""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from accord_ai.eval import score_submission
from accord_ai.eval.scorer import (
    _normalize_for_compare,
    _normalize_v3_expected,
    _resolve_v4_path,
)
from accord_ai.schema import (
    Address,
    CommercialAutoDetails,
    CustomerSubmission,
    Driver,
    PolicyDates,
    Vehicle,
)


# ---------------------------------------------------------------------------
# _resolve_v4_path — walks dicts, lists, and attribute-style objects
# ---------------------------------------------------------------------------

def test_resolve_flat_path_on_dict():
    d = {"business_name": "Acme"}
    assert _resolve_v4_path(d, "business_name") == "Acme"


def test_resolve_nested_path():
    d = {"mailing_address": {"city": "Austin"}}
    assert _resolve_v4_path(d, "mailing_address.city") == "Austin"


def test_resolve_list_index():
    d = {
        "lob_details": {
            "drivers": [{"first_name": "Alice"}, {"first_name": "Bob"}],
        },
    }
    assert _resolve_v4_path(d, "lob_details.drivers[1].first_name") == "Bob"


def test_resolve_missing_returns_none():
    assert _resolve_v4_path({}, "nonexistent.field") is None
    assert _resolve_v4_path({"a": None}, "a.b") is None
    assert _resolve_v4_path({"a": []}, "a[0].b") is None


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def test_normalize_int_to_string():
    assert _normalize_for_compare(22) == "22"


def test_normalize_decimal_to_string():
    assert _normalize_for_compare(Decimal("1000")) == "1000"


def test_normalize_bool():
    assert _normalize_for_compare(True) == "true"
    assert _normalize_for_compare(False) == "false"


def test_normalize_date_iso():
    assert _normalize_for_compare(date(1988, 3, 15)) == "1988-03-15"


def test_normalize_mdy_expected_to_iso():
    assert _normalize_v3_expected("03/15/1988") == "1988-03-15"
    assert _normalize_v3_expected("3/5/2026") == "2026-03-05"


def test_normalize_string_lower_strip():
    assert _normalize_v3_expected("  Austin  ") == "austin"


def test_normalize_empty_to_none():
    assert _normalize_for_compare(None) is None
    assert _normalize_for_compare("") is None
    assert _normalize_for_compare([]) is None
    assert _normalize_for_compare({}) is None
    assert _normalize_v3_expected("") is None


# ---------------------------------------------------------------------------
# score_submission — happy paths
# ---------------------------------------------------------------------------

def test_perfect_score_simple_submission():
    sub = CustomerSubmission(
        business_name="Acme",
        ein="12-3456789",
        phone="512-555-0100",
    )
    result = score_submission("test-perfect", sub, {
        "business.business_name": "Acme",
        "business.tax_id":        "12-3456789",
        "business.phone":         "512-555-0100",
    })
    assert result.total_expected == 3
    assert result.translated    == 3
    assert result.matched       == 3
    assert result.precision == 1.0
    assert result.recall    == 1.0
    assert result.f1        == 1.0


def test_half_score_missing_fields():
    sub = CustomerSubmission(business_name="Acme")
    result = score_submission("test-half", sub, {
        "business.business_name": "Acme",
        "business.tax_id":        "12-3456789",
    })
    assert result.matched == 1
    assert result.precision == 0.5
    assert result.recall    == 0.5


def test_zero_score_no_match():
    sub = CustomerSubmission(business_name="Different")
    result = score_submission("test-zero", sub, {
        "business.business_name": "Acme",
    })
    assert result.matched == 0
    assert result.precision == 0.0
    assert result.recall == 0.0
    # Confirm reason diagnostic is populated.
    assert result.comparisons[0].reason == "mismatch"


def test_mmddyyyy_date_matches_iso_in_submission():
    """v3 expected '03/15/1988' must match v4's date(1988, 3, 15)."""
    sub = CustomerSubmission(
        lob_details=CommercialAutoDetails(drivers=[
            Driver(first_name="Alice", date_of_birth=date(1988, 3, 15)),
        ]),
    )
    result = score_submission("test-date", sub, {
        "drivers[0].dob": "03/15/1988",
    })
    assert result.matched == 1


def test_string_int_cross_type_match():
    """v3 expected '2022', v4 has int 2022 — must match."""
    sub = CustomerSubmission(
        lob_details=CommercialAutoDetails(vehicles=[
            Vehicle(year=2022),
        ]),
    )
    result = score_submission("test-int", sub, {
        "vehicles[0].year": "2022",
    })
    assert result.matched == 1


# ---------------------------------------------------------------------------
# full_name split scoring
# ---------------------------------------------------------------------------

def test_full_name_split_both_match():
    sub = CustomerSubmission(
        lob_details=CommercialAutoDetails(drivers=[
            Driver(first_name="Marcus", last_name="Whitfield"),
        ]),
    )
    result = score_submission("test-split", sub, {
        "drivers[0].full_name": "Marcus Whitfield",
    })
    # One v3 path → two v4 pairs → 2 translated, 2 matched
    assert result.total_expected == 1
    assert result.translated    == 2
    assert result.matched       == 2
    assert result.precision == 1.0


def test_full_name_split_only_first_matches():
    sub = CustomerSubmission(
        lob_details=CommercialAutoDetails(drivers=[
            Driver(first_name="Marcus", last_name="WrongLast"),
        ]),
    )
    result = score_submission("test-split-partial", sub, {
        "drivers[0].full_name": "Marcus Whitfield",
    })
    assert result.translated == 2
    assert result.matched    == 1


# ---------------------------------------------------------------------------
# Untranslatable paths don't break scoring
# ---------------------------------------------------------------------------

def test_untranslatable_paths_tracked_separately():
    sub = CustomerSubmission(business_name="Acme")
    result = score_submission("test-untrans", sub, {
        "business.business_name": "Acme",
        "cyber_info.limit":       "1000000",    # culled
        "business.fax":           "512-555-0000",
    })
    assert "cyber_info.limit" in result.untranslatable_paths
    assert "business.fax" in result.untranslatable_paths
    # Precision unaffected (untranslatable paths not counted in translated).
    assert result.translated == 1
    assert result.matched    == 1
    assert result.precision  == 1.0
    # Recall denominator = total_expected (3), matched only 1 → 1/3.
    assert result.recall == pytest.approx(1 / 3, rel=0.01)


# ---------------------------------------------------------------------------
# Empty expected vs present actual — must not silently pass (reviewer H1)
# ---------------------------------------------------------------------------

def test_empty_expected_with_actual_is_mismatch():
    """Prior impl passed silently when expected was None/empty but the v4
    submission had a concrete value. That hides real divergences from the
    cutover-readiness number. It must be a mismatch."""
    sub = CustomerSubmission(business_name="Acme")
    result = score_submission("test-empty-exp", sub, {
        "business.business_name": None,
    })
    assert result.matched == 0
    assert result.comparisons[0].reason == "unexpected_value"


def test_empty_expected_empty_actual_still_matches():
    sub = CustomerSubmission()  # business_name unset
    result = score_submission("test-empty-both", sub, {
        "business.business_name": None,
    })
    assert result.matched == 1
    assert result.comparisons[0].reason == "ok"


# ---------------------------------------------------------------------------
# Top-level list counts (@count: prefix)
# ---------------------------------------------------------------------------

def test_top_level_vehicles_count_matches():
    sub = CustomerSubmission(
        lob_details=CommercialAutoDetails(vehicles=[Vehicle(year=2022)]),
    )
    result = score_submission("test-count-vehicles", sub, {"vehicles": 1})
    assert result.matched == 1
    assert result.comparisons[0].v4_path == "@count:lob_details.vehicles"


def test_top_level_vehicles_count_mismatch():
    sub = CustomerSubmission(
        lob_details=CommercialAutoDetails(vehicles=[Vehicle(year=2022)]),
    )
    result = score_submission("test-count-mismatch", sub, {"vehicles": 3})
    assert result.matched == 0


def test_top_level_drivers_count_zero_when_empty():
    sub = CustomerSubmission()
    result = score_submission("test-count-empty", sub, {"drivers": 0})
    assert result.matched == 1


# ---------------------------------------------------------------------------
# _ANY and _LIST sentinels
# ---------------------------------------------------------------------------

def test_any_sentinel_matches_any_value():
    sub = CustomerSubmission(nature_of_business="Residential plumbing")
    result = score_submission("test-any", sub, {
        "business.nature_of_business": "_ANY",
    })
    assert result.matched == 1


def test_any_sentinel_fails_on_missing():
    sub = CustomerSubmission()
    result = score_submission("test-any-missing", sub, {
        "business.nature_of_business": "_ANY",
    })
    assert result.matched == 0
    assert result.comparisons[0].reason == "missing"


def test_list_sentinel_matches_non_empty():
    sub = CustomerSubmission(
        lob_details=CommercialAutoDetails(vehicles=[Vehicle(year=2022)]),
    )
    result = score_submission("test-list-vehicles", sub, {"vehicles": "_LIST"})
    assert result.matched == 1


def test_list_sentinel_fails_on_empty():
    sub = CustomerSubmission()
    result = score_submission("test-list-empty", sub, {"vehicles": "_LIST"})
    assert result.matched == 0


# ---------------------------------------------------------------------------
# auto_info.* routing (reviewer B1: fake fields must be untranslatable)
# ---------------------------------------------------------------------------

def test_auto_info_use_type_goes_to_vehicle_not_coverage():
    sub = CustomerSubmission(
        lob_details=CommercialAutoDetails(
            vehicles=[Vehicle(use_type="service")],
        ),
    )
    result = score_submission("test-use-type", sub, {
        "auto_info.use_type": "service",
    })
    assert result.matched == 1
    assert result.comparisons[0].v4_path == "lob_details.vehicles[0].use_type"


def test_auto_info_hazmat_goes_to_details_not_coverage():
    sub = CustomerSubmission(
        lob_details=CommercialAutoDetails(hazmat=False),
    )
    result = score_submission("test-hazmat", sub, {
        "auto_info.hazmat": "false",
    })
    assert result.matched == 1
    assert result.comparisons[0].v4_path == "lob_details.hazmat"


@pytest.mark.parametrize("fake_field", [
    "dash_cam",
    "telematics",
    "driver_training",
    "trailer_interchange",
    "states_of_operation",
    "cargo_type",
    "hired_vehicle_count",
])
def test_auto_info_fake_fields_untranslatable(fake_field):
    """These v3 auto_info.* keys have no v4 schema home — the scorer must
    surface them via untranslatable_paths so we see the gap, not silently
    score them as missing."""
    sub = CustomerSubmission(
        lob_details=CommercialAutoDetails(vehicles=[Vehicle(use_type="service")]),
    )
    result = score_submission("test-fake-auto-info", sub, {
        f"auto_info.{fake_field}": "whatever",
    })
    assert f"auto_info.{fake_field}" in result.untranslatable_paths


# ---------------------------------------------------------------------------
# ScoreResult shape
# ---------------------------------------------------------------------------

def test_score_result_to_dict():
    sub = CustomerSubmission(business_name="Acme")
    result = score_submission("test-shape", sub, {
        "business.business_name": "Acme",
    })
    d = result.to_dict()
    assert d["scenario_id"] == "test-shape"
    assert d["matched"] == 1
    assert "comparisons" in d
    assert d["comparisons"][0]["matched"] is True
    assert d["comparisons"][0]["v3_path"] == "business.business_name"
    assert d["comparisons"][0]["v4_path"] == "business_name"


def test_score_result_is_frozen():
    sub = CustomerSubmission(business_name="Acme")
    result = score_submission("x", sub, {"business.business_name": "Acme"})
    with pytest.raises((AttributeError, TypeError)):
        result.precision = 0.5  # type: ignore


# ---------------------------------------------------------------------------
# Real scenario — mini integration test using the actual standard.yaml shape
# ---------------------------------------------------------------------------

def _solo_plumber_expected() -> dict:
    """Load the real v3 scenario's expected block from disk.

    The scenarios live in the neighboring dual/ sibling repo. We resolve the
    path relative to this test file so the layout is portable across
    checkouts (not hard-coded to one developer's absolute path); if the
    sibling tree isn't present (e.g. CI with only accord_v4 cloned) we skip
    — the test checks production path_map coverage and has no value without
    the real YAML.
    """
    candidates = [
        # Sibling repo layout: accord_v4/tests/.. → ../../dual/dual/eval/...
        Path(__file__).resolve().parents[2] / "dual" / "dual" / "eval" / "scenarios" / "01_solo_plumber.yaml",
        # Absolute fallback for environments where the sibling isn't laid
        # out symmetrically.
        Path("/home/inevoai/Development/Accord-Model-Building/Custom_model_fa_pf/dual/dual/eval/scenarios/01_solo_plumber.yaml"),
    ]
    for path in candidates:
        if path.exists():
            return yaml.safe_load(path.read_text())["expected"]
    pytest.skip("v3 scenarios not available (sibling dual/ tree missing)")


def test_real_scenario_shape_standard_solo_plumber():
    """Port of the real 01_solo_plumber.yaml expected block. Acts as a
    production smoke test: a populated v4 submission matching the scenario
    must score cleanly (no untranslatable paths from the real set, high
    precision/recall).

    This replaces a synthesized expected block — which could silently drift
    from production scenario shape as v3 evolved. Loading the actual YAML
    keeps the scorer honest against what the benchmark actually runs."""
    expected = _solo_plumber_expected()

    # Build a v4 submission that answers every translatable key in `expected`.
    # Fields: auto_info.use_type → vehicles[0].use_type; top-level
    # vehicles/drivers → counts; prior_insurance → _LIST sentinel (WC-only
    # field, so on commercial_auto this WILL be untranslatable-on-resolution
    # and score as missing — that's real signal, not a bug).
    sub = CustomerSubmission(
        business_name="Mike's Plumbing LLC",
        entity_type="llc",
        ein="75-1234567",
        annual_revenue=Decimal("400000"),
        nature_of_business="Residential plumbing",
        mailing_address=Address(city="Dallas", state="TX"),
        policy_dates=PolicyDates(effective_date=date(2026, 7, 1)),
        policy_status="renewal",
        contacts=[{"full_name": "Mike Johnson"}],  # business.contact_name → contacts[0].full_name
        lob_details=CommercialAutoDetails(
            vehicles=[Vehicle(
                year=2022, make="Ford", model="Transit 250",
                vin="1FTBW3XG5NKA12345",
                use_type="service",
                garage_address=Address(city="Dallas", state="TX", zip_code="75201"),
            )],
            drivers=[Driver(
                first_name="Mike", last_name="Johnson",
                date_of_birth=date(1985, 5, 14),
                license_number="12345678",
                license_state="TX",
                years_experience=18,
            )],
        ),
    )
    result = score_submission("solo-plumber", sub, expected)

    # prior_insurance is the one known schema gap: v3 tracks prior insurance
    # on every LOB, v4 only on WC. Assert the miss is PRESENT (not just
    # tolerated) so the day v4 adds CA prior_insurance support and this
    # starts passing, the test breaks and forces explicit removal of the
    # whitelist — the gap gets locked-in-and-visible rather than silently
    # ignored.
    prior_insurance_comparisons = [
        c for c in result.comparisons if c.v3_path == "prior_insurance"
    ]
    assert len(prior_insurance_comparisons) == 1, (
        "Expected exactly one prior_insurance comparison in the scored set"
    )
    assert not prior_insurance_comparisons[0].matched, (
        "prior_insurance now matches on a CA scenario — v4 has grown CA "
        "prior_insurance support; remove this whitelist and the related "
        "note in path_map._COUNT_ROUTES."
    )

    # Every OTHER expected path must translate and match.
    unexpected_misses = [
        (c.v3_path, c.reason) for c in result.comparisons
        if not c.matched and c.v3_path != "prior_insurance"
    ]
    assert not unexpected_misses, (
        f"Unexpected failing comparisons: {unexpected_misses}"
    )
    # No production path should become untranslatable unannounced — the v3
    # path-map is the contract surface and must cover every real key.
    assert result.untranslatable_paths == (), (
        f"Untranslatable production paths: {result.untranslatable_paths}"
    )
