"""Tests for v3 → v4 path translation (P10.S.11a)."""
from __future__ import annotations

import pytest

from accord_ai.eval.path_map import translate


# ---------------------------------------------------------------------------
# Business namespace flattens to root
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("v3_path,v4_path", [
    ("business.business_name",                 "business_name"),
    ("business.dba",                           "dba"),
    ("business.tax_id",                        "ein"),
    ("business.phone",                         "phone"),
    ("business.email",                         "email"),
    ("business.website",                       "website"),
    ("business.entity_type",                   "entity_type"),
    ("business.naics",                         "naics_code"),
    ("business.sic",                           "sic_code"),
    ("business.employee_count",                "full_time_employees"),
    ("business.operations_description",        "operations_description"),
    ("business.mailing_address.city",          "mailing_address.city"),
    ("business.mailing_address.state",         "mailing_address.state"),
    ("business.mailing_address.zip_code",      "mailing_address.zip_code"),
])
def test_business_flattens_to_root(v3_path, v4_path):
    pairs = translate(v3_path, "some_value")
    assert pairs == [(v4_path, "some_value")]


# ---------------------------------------------------------------------------
# Vehicles / drivers shift under lob_details
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("v3_path,v4_path", [
    ("vehicles[0].year",                       "lob_details.vehicles[0].year"),
    ("vehicles[0].make",                       "lob_details.vehicles[0].make"),
    ("vehicles[0].model",                      "lob_details.vehicles[0].model"),
    ("vehicles[0].vin",                        "lob_details.vehicles[0].vin"),
    ("vehicles[0].use_type",                   "lob_details.vehicles[0].use_type"),
    ("vehicles[2].year",                       "lob_details.vehicles[2].year"),
    ("drivers[0].license_number",              "lob_details.drivers[0].license_number"),
    ("drivers[0].license_state",               "lob_details.drivers[0].license_state"),
    ("drivers[1].years_experience",            "lob_details.drivers[1].years_experience"),
])
def test_fleet_shifts_under_lob_details(v3_path, v4_path):
    pairs = translate(v3_path, "x")
    assert pairs == [(v4_path, "x")]


# ---------------------------------------------------------------------------
# Leaf-name renames
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("v3_path,v4_path", [
    ("vehicles[0].radius",                     "lob_details.vehicles[0].radius_of_travel"),
    ("vehicles[0].garaging_address.city",      "lob_details.vehicles[0].garage_address.city"),
    ("vehicles[0].garaging_address.state",     "lob_details.vehicles[0].garage_address.state"),
    ("vehicles[0].garaging_address.zip",       "lob_details.vehicles[0].garage_address.zip_code"),
    ("drivers[0].dob",                         "lob_details.drivers[0].date_of_birth"),
])
def test_leaf_renames(v3_path, v4_path):
    pairs = translate(v3_path, "x")
    assert pairs == [(v4_path, "x")]


# ---------------------------------------------------------------------------
# full_name splits into first + last
# ---------------------------------------------------------------------------

def test_full_name_splits_two_parts():
    pairs = translate("drivers[0].full_name", "Marcus Whitfield")
    assert pairs == [
        ("lob_details.drivers[0].first_name", "Marcus"),
        ("lob_details.drivers[0].last_name",  "Whitfield"),
    ]


def test_full_name_splits_three_parts():
    """First + last; middle silently dropped (v4 Driver has middle_initial,
    but full name usually renders as first + last on the form)."""
    pairs = translate("drivers[0].full_name", "Mary Ann Smith")
    assert pairs == [
        ("lob_details.drivers[0].first_name", "Mary"),
        ("lob_details.drivers[0].last_name",  "Smith"),
    ]


def test_full_name_single_token():
    """Only first — no last_name pair emitted."""
    pairs = translate("drivers[0].full_name", "Madonna")
    assert pairs == [
        ("lob_details.drivers[0].first_name", "Madonna"),
    ]


def test_full_name_second_driver_index_preserved():
    pairs = translate("drivers[2].full_name", "Alice Jones")
    assert pairs == [
        ("lob_details.drivers[2].first_name", "Alice"),
        ("lob_details.drivers[2].last_name",  "Jones"),
    ]


def test_full_name_empty_value_no_pairs():
    assert translate("drivers[0].full_name", "") == []
    assert translate("drivers[0].full_name", None) == []


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("v3_path,v4_path", [
    ("policy.effective_date",                  "policy_dates.effective_date"),
    ("policy.expiration_date",                 "policy_dates.expiration_date"),
    ("policy.status",                          "policy_status"),
])
def test_policy_paths(v3_path, v4_path):
    pairs = translate(v3_path, "x")
    assert pairs == [(v4_path, "x")]


# ---------------------------------------------------------------------------
# Auto coverage
# ---------------------------------------------------------------------------

def test_auto_info_maps_to_coverage():
    pairs = translate("auto_info.liability_limit_csl", "1000000")
    assert pairs == [("lob_details.coverage.liability_limit_csl", "1000000")]


def test_auto_info_hired_auto():
    pairs = translate("auto_info.hired_auto", "true")
    assert pairs == [("lob_details.coverage.hired_auto", "true")]


# ---------------------------------------------------------------------------
# Producer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("v3_path,v4_path", [
    ("producer.agency_name",                   "producer.agency_name"),
    ("producer.mailing_address.city",          "producer.mailing_address.city"),
])
def test_producer_paths(v3_path, v4_path):
    pairs = translate(v3_path, "x")
    assert pairs == [(v4_path, "x")]


# ---------------------------------------------------------------------------
# Untranslatable paths — empty list, not error
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("v3_path", [
    "totally_unknown_field",
    "cyber_info.coverage_limit",       # culled in v4
    "business.fax",                    # culled
    "policy.deposit_amount",           # culled
])
def test_untranslatable_returns_empty(v3_path):
    assert translate(v3_path, "x") == []


def test_untranslatable_bogus_path_returns_empty():
    """Translator never raises on unknown paths — returns [] so the scorer
    can report them via untranslatable_paths."""
    result = translate("whatever.bogus.path", "some value")
    assert result == []


# ---------------------------------------------------------------------------
# Top-level list counts (v3 scenarios write bare `vehicles: 1`)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("v3_key,v4_path", [
    ("vehicles",        "@count:lob_details.vehicles"),
    ("drivers",         "@count:lob_details.drivers"),
    ("loss_history",    "@count:loss_history"),
    ("prior_insurance", "@count:lob_details.prior_insurance"),
])
def test_top_level_counts_emit_count_prefix(v3_key, v4_path):
    pairs = translate(v3_key, 1)
    assert pairs == [(v4_path, 1)]


# ---------------------------------------------------------------------------
# auto_info.* routing (reviewer B1)
# ---------------------------------------------------------------------------

def test_auto_info_use_type_routes_to_first_vehicle():
    pairs = translate("auto_info.use_type", "service")
    assert pairs == [("lob_details.vehicles[0].use_type", "service")]


def test_auto_info_hazmat_routes_to_details_level():
    pairs = translate("auto_info.hazmat", False)
    assert pairs == [("lob_details.hazmat", False)]


def test_auto_info_farthest_zone_routes_to_first_vehicle():
    pairs = translate("auto_info.farthest_zone", "zone_3")
    assert pairs == [("lob_details.vehicles[0].farthest_zone", "zone_3")]


@pytest.mark.parametrize("coverage_field", [
    "liability_limit_csl", "bi_per_person", "bi_per_accident",
    "pd_per_accident", "uim_limit", "medpay_limit",
    "comp_deductible", "coll_deductible", "hired_auto", "non_owned_auto",
])
def test_auto_info_real_coverage_fields_route_to_coverage(coverage_field):
    pairs = translate(f"auto_info.{coverage_field}", "x")
    assert pairs == [(f"lob_details.coverage.{coverage_field}", "x")]


@pytest.mark.parametrize("fake_field", [
    "dash_cam", "telematics", "driver_training", "trailer_interchange",
    "states_of_operation", "cargo_type", "hired_vehicle_count",
])
def test_auto_info_fake_fields_untranslatable(fake_field):
    """Pre-fix behavior routed every auto_info.* to coverage.*, creating
    phantom fields. These must now be untranslatable."""
    assert translate(f"auto_info.{fake_field}", "x") == []
