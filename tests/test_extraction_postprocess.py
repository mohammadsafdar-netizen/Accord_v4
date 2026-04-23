"""Phase A step 2 — postprocess pipeline unit tests.

Five composable functions: ``unfold_dot_keys``, ``strip_empty``,
``drop_phantom_list_items``, ``coerce_list_fields``, ``cap_list_entries``.
Plus the orchestrator ``run_postprocess`` that runs them in order.
"""
from __future__ import annotations

import pytest

from accord_ai.extraction.postprocess import (
    cap_list_entries,
    coerce_list_fields,
    drop_phantom_list_items,
    fix_garage_address_misparse,
    inject_lob_discriminator,
    promote_v3_flat_fields,
    run_postprocess,
    strip_empty,
    unfold_dot_keys,
    _normalize_state_list,
)


# ---------------------------------------------------------------------------
# unfold_dot_keys
# ---------------------------------------------------------------------------

def test_unfold_no_dots_is_noop():
    d = {"business_name": "Acme", "ein": "12-3456789"}
    assert unfold_dot_keys(d) == d


def test_unfold_single_level_dot_key():
    d = {"mailing_address.city": "Dallas"}
    assert unfold_dot_keys(d) == {"mailing_address": {"city": "Dallas"}}


def test_unfold_deep_path():
    d = {"a.b.c.d": 1}
    assert unfold_dot_keys(d) == {"a": {"b": {"c": {"d": 1}}}}


def test_unfold_mixed_keys_preserves_non_dotted():
    d = {"business_name": "Acme", "mailing_address.city": "Dallas"}
    out = unfold_dot_keys(d)
    assert out["business_name"] == "Acme"
    assert out["mailing_address"]["city"] == "Dallas"


def test_unfold_collision_overwrites_non_dict_intermediate():
    """If the same path has both a flat and dotted form, the flat one wins
    on dict-creation but the dotted one places its leaf."""
    d = {"a.b": 1, "a.c": 2}
    assert unfold_dot_keys(d) == {"a": {"b": 1, "c": 2}}


# ---------------------------------------------------------------------------
# strip_empty
# ---------------------------------------------------------------------------

def test_strip_drops_none():
    assert strip_empty({"a": None, "b": 1}) == {"b": 1}


def test_strip_drops_empty_string():
    assert strip_empty({"a": "", "b": "  ", "c": "x"}) == {"c": "x"}


def test_strip_drops_empty_dict():
    assert strip_empty({"a": {}, "b": {"x": 1}}) == {"b": {"x": 1}}


def test_strip_drops_empty_list():
    assert strip_empty({"a": [], "b": [1]}) == {"b": [1]}


def test_strip_keeps_zero_and_false():
    """0 and False are real values, not empty."""
    assert strip_empty({"a": 0, "b": False, "c": None}) == {"a": 0, "b": False}


def test_strip_recurses_into_nested_dicts():
    d = {"address": {"city": "Austin", "zip_code": ""}}
    assert strip_empty(d) == {"address": {"city": "Austin"}}


def test_strip_filters_empty_items_from_list():
    d = {"items": [{}, {"x": 1}, None, "", "  "]}
    assert strip_empty(d) == {"items": [{"x": 1}]}


def test_strip_drops_list_that_becomes_empty_after_filter():
    d = {"items": [{}, None, ""]}
    assert strip_empty(d) == {}


# ---------------------------------------------------------------------------
# drop_phantom_list_items
# ---------------------------------------------------------------------------

def test_phantom_drop_vehicle_with_no_identity():
    """A vehicle with no vin/make/model/year is a correction artifact —
    drop unless the existing state has exactly one to inherit from."""
    delta = {"lob_details": {"vehicles": [{"radius_of_travel": 100}]}}
    current = {"lob_details": {"vehicles": []}}
    drop_phantom_list_items(delta, current)
    assert delta["lob_details"]["vehicles"] == []


def test_phantom_keep_vehicle_with_vin():
    delta = {"lob_details": {"vehicles": [{"vin": "1FT12345", "year": 2024}]}}
    current = {"lob_details": {"vehicles": []}}
    drop_phantom_list_items(delta, current)
    assert delta["lob_details"]["vehicles"] == [
        {"vin": "1FT12345", "year": 2024},
    ]


def test_phantom_merges_into_single_existing_vehicle():
    """User says 'actually the year is 2022' → LLM emits {year: 2022} as a
    new vehicle. The existing single vehicle gives identity; we merge
    so the correction lands on the right entity."""
    delta = {"lob_details": {"vehicles": [{"year": 2022}]}}
    current = {"lob_details": {"vehicles": [
        {"vin": "1FT12345", "make": "Ford", "model": "F250"},
    ]}}
    drop_phantom_list_items(delta, current)
    veh = delta["lob_details"]["vehicles"]
    assert len(veh) == 1
    assert veh[0]["vin"] == "1FT12345"
    assert veh[0]["year"] == 2022


def test_phantom_does_not_merge_when_multiple_existing():
    """Ambiguous which existing vehicle to inherit from — drop the phantom."""
    delta = {"lob_details": {"vehicles": [{"year": 2022}]}}
    current = {"lob_details": {"vehicles": [
        {"vin": "111", "make": "Ford"},
        {"vin": "222", "make": "Chevy"},
    ]}}
    drop_phantom_list_items(delta, current)
    assert delta["lob_details"]["vehicles"] == []


def test_phantom_drop_driver_no_identity():
    delta = {"lob_details": {"drivers": [{"sex": "F"}]}}
    drop_phantom_list_items(delta, {"lob_details": {"drivers": []}})
    assert delta["lob_details"]["drivers"] == []


def test_phantom_drop_driver_with_license_kept():
    delta = {"lob_details": {"drivers": [{"license_number": "X123"}]}}
    drop_phantom_list_items(delta, {"lob_details": {"drivers": []}})
    assert delta["lob_details"]["drivers"] == [{"license_number": "X123"}]


def test_phantom_drop_loss_history_no_description():
    delta = {"loss_history": [{"amount_paid": 5000}]}
    drop_phantom_list_items(delta, {"loss_history": []})
    assert delta["loss_history"] == []


def test_phantom_drop_handles_missing_lob_details_gracefully():
    """No lob_details on delta → no work to do, no exception."""
    delta = {"business_name": "Acme"}
    drop_phantom_list_items(delta, {})
    assert delta == {"business_name": "Acme"}


# ---------------------------------------------------------------------------
# coerce_list_fields / _normalize_state_list
# ---------------------------------------------------------------------------

def test_normalize_state_list_handles_uppercase_codes_with_spaces():
    assert _normalize_state_list("NE IA MO") == ["NE", "IA", "MO"]


def test_normalize_state_list_handles_commas():
    assert _normalize_state_list("TX, OK, NM") == ["TX", "OK", "NM"]


def test_normalize_state_list_handles_full_names():
    assert _normalize_state_list(
        "Texas, Oklahoma, New Mexico"
    ) == ["TX", "OK", "NM"]


def test_normalize_state_list_handles_and_separator():
    assert _normalize_state_list("TX and OK") == ["TX", "OK"]


def test_normalize_state_list_drops_unknown_tokens():
    """Unknown tokens silently dropped; valid neighbors survive."""
    assert _normalize_state_list("TX, Atlantis, NY") == ["TX", "NY"]


def test_normalize_state_list_returns_none_for_non_string():
    """Already-list inputs return None (caller leaves them alone)."""
    assert _normalize_state_list(["TX", "OK"]) is None
    assert _normalize_state_list(None) is None


def test_normalize_state_list_returns_none_for_empty_string():
    assert _normalize_state_list("") is None
    assert _normalize_state_list("   ") is None


def test_coerce_list_fields_states_of_operation():
    delta = {"lob_details": {"states_of_operation": "TX OK NM"}}
    coerce_list_fields(delta)
    assert delta["lob_details"]["states_of_operation"] == ["TX", "OK", "NM"]


def test_coerce_list_fields_leaves_already_list_alone():
    delta = {"lob_details": {"states_of_operation": ["TX", "OK"]}}
    coerce_list_fields(delta)
    assert delta["lob_details"]["states_of_operation"] == ["TX", "OK"]


def test_coerce_list_fields_no_lob_details_is_noop():
    delta = {"business_name": "Acme"}
    coerce_list_fields(delta)
    assert delta == {"business_name": "Acme"}


# ---------------------------------------------------------------------------
# cap_list_entries
# ---------------------------------------------------------------------------

def test_cap_vehicles_under_limit_unchanged():
    delta = {"lob_details": {"vehicles": [{"vin": "A"}, {"vin": "B"}]}}
    cap_list_entries(delta)
    assert len(delta["lob_details"]["vehicles"]) == 2


def test_cap_vehicles_over_limit_trimmed():
    delta = {"lob_details": {"vehicles": [{"vin": str(i)} for i in range(7)]}}
    cap_list_entries(delta)
    assert len(delta["lob_details"]["vehicles"]) == 3
    # First 3 preserved, in order.
    assert [v["vin"] for v in delta["lob_details"]["vehicles"]] == ["0", "1", "2"]


def test_cap_drivers_over_limit_trimmed():
    delta = {"lob_details": {"drivers": [{"license_number": str(i)} for i in range(5)]}}
    cap_list_entries(delta)
    assert len(delta["lob_details"]["drivers"]) == 3


def test_cap_loss_history_over_limit_trimmed():
    delta = {"loss_history": [{"description": str(i)} for i in range(5)]}
    cap_list_entries(delta)
    assert len(delta["loss_history"]) == 3


def test_cap_handles_missing_paths():
    delta = {"business_name": "Acme"}
    cap_list_entries(delta)
    assert delta == {"business_name": "Acme"}


# ---------------------------------------------------------------------------
# run_postprocess (full pipeline)
# ---------------------------------------------------------------------------

def test_run_postprocess_full_pipeline():
    """End-to-end: dot-keys unfold, empties strip, phantom drops, state
    list coerces, lists cap. Single LLM-output dict goes through all 5."""
    delta = {
        # dot-key form (step 1)
        "mailing_address.city": "Austin",
        "mailing_address.state": "TX",
        # empty values (step 2)
        "ein": None,
        "phone": "",
        # phantom vehicle (step 3) — no identity, will be dropped
        "lob_details.vehicles": [{"radius_of_travel": 50}],
        # string state list (step 4)
        "lob_details.states_of_operation": "TX OK",
        # over-cap loss list (step 5)
        "loss_history": [{"description": str(i)} for i in range(5)],
    }
    out = run_postprocess(delta, current_state={"lob_details": {"vehicles": []}})
    assert out["mailing_address"]["city"] == "Austin"
    assert out["mailing_address"]["state"] == "TX"
    assert "ein" not in out
    assert "phone" not in out
    assert out["lob_details"]["vehicles"] == []
    assert out["lob_details"]["states_of_operation"] == ["TX", "OK"]
    assert len(out["loss_history"]) == 3


def test_run_postprocess_empty_delta():
    """Empty input returns empty (or near-empty after stripping)."""
    out = run_postprocess({}, current_state={})
    assert out == {}


def test_run_postprocess_idempotent_on_clean_delta():
    delta = {
        "business_name": "Acme",
        "lob_details": {
            "vehicles": [{"vin": "X", "year": 2024}],
            "states_of_operation": ["TX"],
        },
    }
    once = run_postprocess(dict(delta), current_state={})
    twice = run_postprocess(dict(once), current_state={})
    assert once == twice


# ---------------------------------------------------------------------------
# Step 3A — phantom-merge inherits FULL current item, not just identity
# ---------------------------------------------------------------------------

def test_phantom_merge_inherits_full_vehicle_fields():
    """Correction turn emits {year: 2023} only → phantom-merge must
    inherit make/model/vin AND garage_address/use_type/etc. from the
    existing vehicle. Without this, apply_diff's list-replace loses
    non-identity fields like garage_address on correction turns."""
    delta = {"lob_details": {"vehicles": [{"year": 2023}]}}
    current = {
        "lob_details": {
            "lob": "commercial_auto",
            "vehicles": [{
                "vin": "ABC123",
                "make": "Toyota",
                "model": "Tacoma",
                "year": 2022,
                "garage_address": {"city": "Tucson", "state": "AZ"},
                "use_type": "service",
            }],
        },
    }
    drop_phantom_list_items(delta, current)
    merged = delta["lob_details"]["vehicles"][0]
    # Correction applied
    assert merged["year"] == 2023
    # Identity preserved
    assert merged["vin"] == "ABC123"
    assert merged["make"] == "Toyota"
    # Non-identity context preserved (the bug this fix addresses)
    assert merged["garage_address"] == {"city": "Tucson", "state": "AZ"}
    assert merged["use_type"] == "service"


def test_phantom_merge_prefers_diff_value_over_current():
    """If both diff and current have the same key, diff wins."""
    delta = {"lob_details": {"vehicles": [{"year": 2023, "color": "red"}]}}
    current = {
        "lob_details": {
            "vehicles": [{"vin": "V", "year": 2022, "color": "blue"}],
        },
    }
    drop_phantom_list_items(delta, current)
    v = delta["lob_details"]["vehicles"][0]
    assert v["year"] == 2023
    assert v["color"] == "red"


def test_phantom_dropped_when_no_single_identity_bearing_current_item():
    """Phantom + no/multiple current items with identity → drop phantom."""
    delta = {"lob_details": {"vehicles": [{"year": 2023}]}}
    current = {
        "lob_details": {
            "vehicles": [
                {"vin": "A", "make": "Ford"},
                {"vin": "B", "make": "Chevy"},
            ],
        },
    }
    drop_phantom_list_items(delta, current)
    # Phantom dropped — can't decide which vehicle to merge into.
    assert delta["lob_details"]["vehicles"] == []


# ---------------------------------------------------------------------------
# Step 6 — inject_lob_discriminator
# ---------------------------------------------------------------------------

def test_inject_lob_fills_missing_discriminator_from_current_state():
    delta = {"lob_details": {"vehicles": [{"vin": "X"}]}}
    current = {"lob_details": {"lob": "commercial_auto"}}
    inject_lob_discriminator(delta, current)
    assert delta["lob_details"]["lob"] == "commercial_auto"


def test_inject_lob_preserves_existing_discriminator():
    delta = {"lob_details": {"lob": "commercial_auto", "vehicles": []}}
    current = {"lob_details": {"lob": "general_liability"}}
    inject_lob_discriminator(delta, current)
    # Delta's explicit value wins.
    assert delta["lob_details"]["lob"] == "commercial_auto"


def test_inject_lob_noop_when_current_has_no_lob():
    delta = {"lob_details": {"vehicles": []}}
    current = {}
    inject_lob_discriminator(delta, current)
    assert "lob" not in delta["lob_details"]


def test_inject_lob_noop_when_delta_has_no_lob_details():
    delta = {"business_name": "Acme"}
    current = {"lob_details": {"lob": "commercial_auto"}}
    inject_lob_discriminator(delta, current)
    assert "lob_details" not in delta


def test_inject_lob_ignores_unknown_lob_values():
    """Defensive: if current_state carries a bogus lob string, don't
    propagate it into the delta where it would fail validation
    differently."""
    delta = {"lob_details": {"vehicles": []}}
    current = {"lob_details": {"lob": "fake_lob"}}
    inject_lob_discriminator(delta, current)
    assert "lob" not in delta["lob_details"]


# ---------------------------------------------------------------------------
# Step 6a — fix_garage_address_misparse
# ---------------------------------------------------------------------------

def test_fix_garage_address_swaps_mis_slotted_city_state():
    """LLM emitted line_one="Tucson", city="AZ" (state in city slot) →
    shift: line_one=None, city="Tucson", state="AZ"."""
    delta = {
        "lob_details": {
            "vehicles": [{
                "vin": "V",
                "garage_address": {
                    "line_one": "Tucson",
                    "city": "AZ",
                    "zip_code": "85701",
                },
            }],
        },
    }
    fix_garage_address_misparse(delta)
    ga = delta["lob_details"]["vehicles"][0]["garage_address"]
    assert ga["line_one"] is None
    assert ga["city"] == "Tucson"
    assert ga["state"] == "AZ"
    assert ga["zip_code"] == "85701"


def test_fix_garage_address_leaves_real_street_alone():
    """line_one="123 Main St" looks like a street → don't shift."""
    delta = {
        "lob_details": {
            "vehicles": [{
                "vin": "V",
                "garage_address": {
                    "line_one": "123 Main St",
                    "city": "TX",   # looks suspicious but line_one is a real street
                    "zip_code": "78701",
                },
            }],
        },
    }
    fix_garage_address_misparse(delta)
    ga = delta["lob_details"]["vehicles"][0]["garage_address"]
    assert ga["line_one"] == "123 Main St"
    # city is left alone — we only shift when line_one is clearly not a street.
    assert ga["city"] == "TX"


def test_fix_garage_address_leaves_well_parsed_address_alone():
    delta = {
        "lob_details": {
            "vehicles": [{
                "vin": "V",
                "garage_address": {
                    "line_one": "123 Main St",
                    "city": "Austin",
                    "state": "TX",
                    "zip_code": "78701",
                },
            }],
        },
    }
    fix_garage_address_misparse(delta)
    ga = delta["lob_details"]["vehicles"][0]["garage_address"]
    assert ga == {
        "line_one": "123 Main St",
        "city": "Austin",
        "state": "TX",
        "zip_code": "78701",
    }


def test_fix_garage_address_no_vehicles_is_noop():
    delta = {"business_name": "Acme"}
    fix_garage_address_misparse(delta)
    assert delta == {"business_name": "Acme"}


# ---------------------------------------------------------------------------
# Step 0 — promote_v3_flat_fields
# ---------------------------------------------------------------------------

def test_promote_flat_vehicles_to_lob_details():
    """LLM emitted v3-style top-level ``vehicles`` because SYSTEM_V2
    lacks the harness path-translation table. Schema's extra='ignore'
    drops them silently — this promotion saves the turn."""
    delta = {
        "business_name": "Iron Horse",
        "vehicles": [{"year": 2024, "vin": "V1", "make": "Kenworth"}],
    }
    promote_v3_flat_fields(delta)
    assert "vehicles" not in delta
    assert delta["lob_details"]["vehicles"] == [
        {"year": 2024, "vin": "V1", "make": "Kenworth"},
    ]


def test_promote_flat_drivers_to_lob_details():
    delta = {"drivers": [{"license_number": "X123", "first_name": "Jane"}]}
    promote_v3_flat_fields(delta)
    assert "drivers" not in delta
    assert delta["lob_details"]["drivers"] == [
        {"license_number": "X123", "first_name": "Jane"},
    ]


def test_promote_flat_concatenates_with_existing_lob_details_list():
    """If lob_details.vehicles already exists, concatenate."""
    delta = {
        "lob_details": {"vehicles": [{"vin": "A"}]},
        "vehicles": [{"vin": "B"}],
    }
    promote_v3_flat_fields(delta)
    assert "vehicles" not in delta
    assert delta["lob_details"]["vehicles"] == [{"vin": "A"}, {"vin": "B"}]


def test_promote_flat_scalar_hazmat():
    delta = {"hazmat": False}
    promote_v3_flat_fields(delta)
    assert "hazmat" not in delta
    assert delta["lob_details"]["hazmat"] is False


def test_promote_flat_scalar_preserves_existing_nested_value():
    """Nested explicit value takes precedence over flat fallback —
    the LLM meant to set the nested one."""
    delta = {
        "lob_details": {"hazmat": True},
        "hazmat": False,
    }
    promote_v3_flat_fields(delta)
    assert delta["lob_details"]["hazmat"] is True
    assert "hazmat" not in delta


def test_promote_flat_noop_when_no_flat_fields():
    delta = {"business_name": "Acme", "ein": "12-3456789"}
    before = dict(delta)
    promote_v3_flat_fields(delta)
    assert delta == before


def test_promote_flat_ignores_non_list_vehicles():
    """If vehicles is somehow not a list (rare LLM mistake), don't
    promote — silently drop to avoid creating malformed data."""
    delta = {"vehicles": "nonsense"}
    promote_v3_flat_fields(delta)
    # The bogus value is popped; no lob_details.vehicles is created.
    assert "vehicles" not in delta
    assert "lob_details" not in delta


def test_run_postprocess_integrates_promote_flat():
    """Full pipeline: v3-flat vehicles + phantom + negation-style → works."""
    delta = {
        "vehicles": [{"year": 2024, "vin": "V1", "make": "Kenworth"}],
    }
    out = run_postprocess(delta, current_state={})
    assert "vehicles" not in out
    assert out["lob_details"]["vehicles"][0]["year"] == 2024


def test_fix_garage_address_does_not_shift_if_state_already_set():
    """If state is already populated, the mis-slotted pattern doesn't
    apply — something else is going on, leave it alone."""
    delta = {
        "lob_details": {
            "vehicles": [{
                "vin": "V",
                "garage_address": {
                    "line_one": "Tucson",
                    "city": "AZ",
                    "state": "AZ",    # already set
                },
            }],
        },
    }
    fix_garage_address_misparse(delta)
    ga = delta["lob_details"]["vehicles"][0]["garage_address"]
    assert ga["line_one"] == "Tucson"
    assert ga["city"] == "AZ"
    assert ga["state"] == "AZ"
