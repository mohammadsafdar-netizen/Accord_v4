"""6.a — apply_diff on scalars (loose removal protection)."""
import pytest

from accord_ai.core.diff import apply_diff
from accord_ai.schema import CustomerSubmission


def test_empty_diff_returns_equivalent_current():
    current = CustomerSubmission(business_name="Acme")
    result = apply_diff(current, CustomerSubmission())
    assert result.business_name == "Acme"


def test_scalar_in_diff_replaces_current():
    current = CustomerSubmission(business_name="Old")
    diff = CustomerSubmission(business_name="New")
    result = apply_diff(current, diff)
    assert result.business_name == "New"


def test_scalar_explicitly_none_in_diff_is_noop():
    """Loose removal protection: diff.business_name=None leaves current alone."""
    current = CustomerSubmission(business_name="Acme")
    diff = CustomerSubmission(business_name=None)
    result = apply_diff(current, diff)
    assert result.business_name == "Acme"


def test_unset_fields_in_diff_dont_affect_current():
    """Only fields in diff.model_fields_set are touched."""
    current = CustomerSubmission(business_name="Acme", ein="12-3456789", phone="555-1234")
    diff = CustomerSubmission(business_name="New")   # ein/phone NOT in diff.model_fields_set
    result = apply_diff(current, diff)
    assert result.business_name == "New"
    assert result.ein == "12-3456789"
    assert result.phone == "555-1234"


def test_multiple_scalars_replaced_in_one_diff():
    current = CustomerSubmission()
    diff = CustomerSubmission(
        business_name="Acme",
        ein="12-3456789",
        phone="555-1234",
        email="ops@acme.com",
    )
    result = apply_diff(current, diff)
    assert result.business_name == "Acme"
    assert result.ein == "12-3456789"
    assert result.phone == "555-1234"
    assert result.email == "ops@acme.com"


def test_apply_diff_returns_new_instance_not_mutation():
    current = CustomerSubmission(business_name="Acme")
    diff = CustomerSubmission(business_name="New")
    result = apply_diff(current, diff)
    assert result is not current
    assert current.business_name == "Acme"   # original untouched
    assert result.business_name == "New"


def test_diff_with_all_none_fields_is_full_noop():
    current = CustomerSubmission(business_name="Acme", ein="12-3456789")
    diff = CustomerSubmission(business_name=None, ein=None, phone=None)
    result = apply_diff(current, diff)
    assert result.business_name == "Acme"
    assert result.ein == "12-3456789"


def test_scalar_typo_correction_end_to_end():
    """Real case — LLM extracts 'Acme Corp' turn 1, user says 'actually Acme Corporation'."""
    current = CustomerSubmission(business_name="Acme Corp")
    diff = CustomerSubmission(business_name="Acme Corporation")
    result = apply_diff(current, diff)
    assert result.business_name == "Acme Corporation"


def test_result_roundtrips_through_json():
    """Merged submission must be a valid Pydantic instance."""
    current = CustomerSubmission(business_name="Acme")
    diff = CustomerSubmission(ein="12-3456789")
    result = apply_diff(current, diff)
    restored = CustomerSubmission.model_validate(result.model_dump(mode="json"))
    assert restored.business_name == "Acme"
    assert restored.ein == "12-3456789"


def test_json_null_treated_as_explicit_none():
    """LLM emits {"business_name": null} — must be skipped, not blank the field."""
    current = CustomerSubmission(business_name="Acme")
    diff = CustomerSubmission.model_validate({"business_name": None})
    # model_fields_set contains 'business_name' — but value is None → skipped
    result = apply_diff(current, diff)
    assert result.business_name == "Acme"


# ============================================================
# 6.a lock-in — flagged during 6.a review
# ============================================================

def test_fields_set_contract_is_consistent_across_construction_paths():
    """Both kwargs-with-None and validate-from-dict-with-null must add the
    field to model_fields_set — apply_diff's None-skip contract depends on
    both paths behaving identically. If Pydantic ever changes this, the
    test fires before we hit weird behavior in 6.b/c/d.
    """
    via_kwargs = CustomerSubmission(business_name=None)
    via_validate = CustomerSubmission.model_validate({"business_name": None})
    assert "business_name" in via_kwargs.model_fields_set
    assert "business_name" in via_validate.model_fields_set

    # Same apply_diff behavior either way
    current = CustomerSubmission(business_name="Acme")
    assert apply_diff(current, via_kwargs).business_name == "Acme"
    assert apply_diff(current, via_validate).business_name == "Acme"


# ============================================================
# 6.b — nested-model recursive merge
# ============================================================

from datetime import date


def test_address_on_unset_current_is_set_wholesale():
    current = CustomerSubmission()
    diff = CustomerSubmission(business_address={"city": "Detroit", "state": "MI"})
    result = apply_diff(current, diff)
    assert result.business_address is not None
    assert result.business_address.city == "Detroit"
    assert result.business_address.state == "MI"


def test_nested_merge_preserves_unchanged_fields():
    """Diff sets zip_code only — line_one/city/state survive."""
    current = CustomerSubmission(business_address={
        "line_one": "123 Main", "city": "Detroit", "state": "MI",
    })
    diff = CustomerSubmission(business_address={"zip_code": "48201"})
    result = apply_diff(current, diff)
    assert result.business_address.line_one == "123 Main"
    assert result.business_address.city == "Detroit"
    assert result.business_address.state == "MI"
    assert result.business_address.zip_code == "48201"


def test_nested_scalar_replace_corrects_typo():
    current = CustomerSubmission(business_address={"city": "Detrit"})
    diff = CustomerSubmission(business_address={"city": "Detroit"})
    result = apply_diff(current, diff)
    assert result.business_address.city == "Detroit"


def test_nested_scalar_none_is_noop():
    """diff.business_address.city=None must NOT clear current's value."""
    current = CustomerSubmission(business_address={"city": "Detroit"})
    diff = CustomerSubmission(business_address={"city": None})
    result = apply_diff(current, diff)
    assert result.business_address.city == "Detroit"


def test_nested_model_explicitly_none_in_diff_is_noop():
    """diff.business_address=None is the 'I didn't say anything about address' signal."""
    current = CustomerSubmission(business_address={"city": "Detroit"})
    diff = CustomerSubmission(business_address=None)
    result = apply_diff(current, diff)
    assert result.business_address is not None
    assert result.business_address.city == "Detroit"


def test_policy_dates_merge_preserves_effective_date():
    current = CustomerSubmission(policy_dates={"effective_date": "2026-05-01"})
    diff = CustomerSubmission(policy_dates={"expiration_date": "2027-05-01"})
    result = apply_diff(current, diff)
    assert result.policy_dates.effective_date == date(2026, 5, 1)
    assert result.policy_dates.expiration_date == date(2027, 5, 1)


def test_business_and_mailing_address_are_independent():
    current = CustomerSubmission(business_address={"city": "Detroit"})
    diff = CustomerSubmission(mailing_address={"city": "Ann Arbor"})
    result = apply_diff(current, diff)
    assert result.business_address.city == "Detroit"
    assert result.mailing_address.city == "Ann Arbor"


def test_nested_merge_does_not_mutate_current():
    """Returned tree must not share references with `current`."""
    current = CustomerSubmission(business_address={"city": "Detrit"})
    diff = CustomerSubmission(business_address={"city": "Detroit"})
    result = apply_diff(current, diff)
    # current untouched
    assert current.business_address.city == "Detrit"
    # result has the correction
    assert result.business_address.city == "Detroit"
    # the two address objects are distinct
    assert result.business_address is not current.business_address


def test_nested_inside_lob_details_merges_when_same_lob():
    """CommercialAutoDetails → CommercialAutoDetails: recursive merge on the
    inner coverage model (6.c will refine list fields; this test uses
    only scalars inside coverage)."""
    current = CustomerSubmission(lob_details={
        "lob": "commercial_auto",
        "coverage": {"liability_limit_csl": 1_000_000},
    })
    diff = CustomerSubmission(lob_details={
        "lob": "commercial_auto",
        "coverage": {"comp_deductible": 500},
    })
    result = apply_diff(current, diff)
    assert result.lob_details.lob == "commercial_auto"
    assert result.lob_details.coverage.liability_limit_csl == 1_000_000
    assert result.lob_details.coverage.comp_deductible == 500


# ============================================================
# 6.c — list merge (replace-if-longer-or-equal)
# ============================================================

def test_top_level_list_longer_in_diff_replaces():
    current = CustomerSubmission(
        additional_interests=[{"name": "Ally", "role": "lienholder"}],
    )
    diff = CustomerSubmission(additional_interests=[
        {"name": "Ally", "role": "lienholder"},
        {"name": "Landlord LLC", "role": "additional_insured"},
    ])
    result = apply_diff(current, diff)
    assert [ai.name for ai in result.additional_interests] == ["Ally", "Landlord LLC"]


def test_top_level_list_equal_length_in_diff_replaces():
    """Equal-length still replaces — handles typo correction in a list."""
    current = CustomerSubmission(additional_interests=[{"name": "Aly Bank"}])
    diff = CustomerSubmission(additional_interests=[{"name": "Ally Bank"}])
    result = apply_diff(current, diff)
    assert result.additional_interests[0].name == "Ally Bank"


def test_top_level_list_shorter_in_diff_keeps_current():
    """LLM accidentally dropped items — protect current."""
    current = CustomerSubmission(additional_interests=[
        {"name": "Ally Bank", "role": "lienholder"},
        {"name": "Landlord LLC", "role": "additional_insured"},
        {"name": "Vendor X", "role": "additional_insured"},
    ])
    diff = CustomerSubmission(additional_interests=[
        {"name": "Ally Bank", "role": "lienholder"},
    ])
    result = apply_diff(current, diff)
    assert len(result.additional_interests) == 3
    assert {ai.name for ai in result.additional_interests} == {
        "Ally Bank", "Landlord LLC", "Vendor X",
    }


def test_empty_list_in_diff_keeps_non_empty_current():
    current = CustomerSubmission(loss_history=[
        {"date_of_loss": "2024-03-15", "type_of_loss": "collision", "amount_paid": 8500},
    ])
    diff = CustomerSubmission(loss_history=[])
    result = apply_diff(current, diff)
    assert len(result.loss_history) == 1


def test_list_on_empty_current_accepts_any_diff_list():
    """Default list is [] (len 0). Any non-empty diff list replaces (len >= 0)."""
    current = CustomerSubmission()
    assert current.additional_interests == []
    diff = CustomerSubmission(additional_interests=[{"name": "Ally Bank"}])
    result = apply_diff(current, diff)
    assert len(result.additional_interests) == 1


def test_list_inside_lob_details_merge_adds_new_driver():
    """Drivers use identity-based merge: a new driver is added alongside the existing one.

    Old 6.c replace-if-longer contract replaced entirely when diff list was longer.
    New merge contract: existing Alice (matched by license) stays; Bob is added.
    """
    current = CustomerSubmission(lob_details={
        "lob": "commercial_auto",
        "drivers": [{"first_name": "Alice", "license_number": "A001"}],
    })
    diff = CustomerSubmission(lob_details={
        "lob": "commercial_auto",
        "drivers": [
            {"first_name": "Alice", "license_number": "A001"},
            {"first_name": "Bob",   "license_number": "B002"},
        ],
    })
    result = apply_diff(current, diff)
    names = [d.first_name for d in result.lob_details.drivers]
    assert "Alice" in names
    assert "Bob" in names
    assert len(result.lob_details.drivers) == 2


def test_list_inside_lob_details_preserves_unmentioned_drivers():
    """Key property of the new merge: drivers NOT in the diff are NOT dropped.

    Old 6.c behavior: a shorter diff list kept current unchanged (only by luck —
    it hit the len < len branch). New merge behavior: Carol and Bob are preserved
    because they are never mentioned in the diff, so they have no incoming
    counterpart to overwrite them.
    """
    current = CustomerSubmission(lob_details={
        "lob": "commercial_auto",
        "drivers": [
            {"first_name": "Alice", "license_number": "A001"},
            {"first_name": "Bob",   "license_number": "B002"},
            {"first_name": "Carol", "license_number": "C003"},
        ],
    })
    diff = CustomerSubmission(lob_details={
        "lob": "commercial_auto",
        "drivers": [{"first_name": "Alice", "license_number": "A001", "years_experience": 5}],
    })
    result = apply_diff(current, diff)
    # All three drivers are preserved; Alice gets her years_experience updated
    assert len(result.lob_details.drivers) == 3
    alice = next(d for d in result.lob_details.drivers if d.first_name == "Alice")
    assert alice.years_experience == 5


def test_list_merge_does_not_mutate_current():
    current = CustomerSubmission(additional_interests=[{"name": "Ally"}])
    diff = CustomerSubmission(additional_interests=[{"name": "Ally"}, {"name": "Landlord"}])
    result = apply_diff(current, diff)
    assert len(current.additional_interests) == 1       # unchanged
    assert len(result.additional_interests) == 2
    assert result.additional_interests is not current.additional_interests


# ============================================================
# 6.d — LOB transition rejection
# ============================================================

from accord_ai.core.diff import LobTransitionError


def test_lob_transition_ca_to_gl_raises():
    current = CustomerSubmission(lob_details={
        "lob": "commercial_auto",
        "drivers": [{"first_name": "Alice"}],
    })
    diff = CustomerSubmission(lob_details={
        "lob": "general_liability",
        "employee_count": 25,
    })
    with pytest.raises(LobTransitionError):
        apply_diff(current, diff)


def test_lob_transition_gl_to_wc_raises():
    current = CustomerSubmission(lob_details={"lob": "general_liability", "employee_count": 12})
    diff = CustomerSubmission(lob_details={"lob": "workers_comp", "experience_mod": 0.95})
    with pytest.raises(LobTransitionError):
        apply_diff(current, diff)


def test_lob_transition_ca_to_wc_raises():
    current = CustomerSubmission(lob_details={"lob": "commercial_auto"})
    diff = CustomerSubmission(lob_details={"lob": "workers_comp", "experience_mod": 1.0})
    with pytest.raises(LobTransitionError):
        apply_diff(current, diff)


def test_lob_transition_error_message_names_both_types():
    current = CustomerSubmission(lob_details={"lob": "commercial_auto"})
    diff = CustomerSubmission(lob_details={"lob": "general_liability", "employee_count": 1})
    with pytest.raises(LobTransitionError) as exc_info:
        apply_diff(current, diff)
    msg = str(exc_info.value)
    assert "CommercialAutoDetails" in msg
    assert "GeneralLiabilityDetails" in msg


def test_lob_details_on_unset_current_accepts_any_lob():
    """First-time set isn't a transition — no current to preserve."""
    current = CustomerSubmission()
    assert current.lob_details is None
    diff = CustomerSubmission(lob_details={"lob": "workers_comp", "experience_mod": 0.85})
    result = apply_diff(current, diff)
    assert result.lob_details.lob == "workers_comp"


def test_lob_details_none_in_diff_preserves_current_lob():
    """Loose removal protection: explicit None in diff is a no-op."""
    current = CustomerSubmission(lob_details={"lob": "commercial_auto"})
    diff = CustomerSubmission(lob_details=None)
    result = apply_diff(current, diff)
    assert result.lob_details.lob == "commercial_auto"


def test_lob_transition_does_not_partially_apply_other_fields():
    """Raise must happen before any field mutation — current stays pristine."""
    current = CustomerSubmission(
        business_name="Acme",
        lob_details={"lob": "commercial_auto"},
    )
    diff = CustomerSubmission(
        business_name="Acme Corporation",
        lob_details={"lob": "general_liability", "employee_count": 5},
    )
    with pytest.raises(LobTransitionError):
        apply_diff(current, diff)
    # Original untouched
    assert current.business_name == "Acme"
    assert current.lob_details.lob == "commercial_auto"


def test_lob_transition_error_is_subclass_of_valueerror():
    """Callers can catch ValueError as a general 'bad input' handler."""
    assert issubclass(LobTransitionError, ValueError)
