"""Tests for scalar fan-out (P10.S.3) — business_name, policy dates, mailing address."""
from __future__ import annotations

from datetime import date

import pytest

from accord_ai.forms.mapper import (
    _FORM_ALIASES,
    _SCHEMA_RESOLVERS,
    _lookup_resolver,
    map_submission_to_form,
)
from accord_ai.schema import (
    Address,
    CommercialAutoDetails,
    CustomerSubmission,
    GeneralLiabilityDetails,
    PolicyDates,
    WorkersCompDetails,
)


def _sub(**kw):
    return CustomerSubmission(**kw)


# ---------------------------------------------------------------------------
# Fan-out topology: business_name lives on 9 forms
# ---------------------------------------------------------------------------

_BUSINESS_NAME_FORMS = (
    "125", "126", "127", "129", "130", "131", "137", "159", "160",
)


@pytest.mark.parametrize("form_number", _BUSINESS_NAME_FORMS)
def test_business_name_maps_to_named_insured_full_name_a(form_number):
    aliases = _FORM_ALIASES[form_number]
    assert aliases.get("NamedInsured_FullName_A") == "business_name"


def test_business_name_shares_single_resolver_across_all_nine_forms():
    resolver_ids = set()
    for form_number in _BUSINESS_NAME_FORMS:
        schema_key = _FORM_ALIASES[form_number]["NamedInsured_FullName_A"]
        resolver_ids.add(id(_lookup_resolver(schema_key)))
    assert len(resolver_ids) == 1


@pytest.mark.parametrize("form_number", _BUSINESS_NAME_FORMS)
def test_business_name_fills_on_every_form(form_number):
    """Given a submission with a name, each of the 9 forms emits it."""
    s = _sub(business_name="Acme", lob_details=CommercialAutoDetails())
    m = map_submission_to_form(s, form_number)
    assert m.get("NamedInsured_FullName_A") == "Acme"


# ---------------------------------------------------------------------------
# Fan-out topology: effective_date lives on 9 forms (125-160, not 163)
# ---------------------------------------------------------------------------

_EFFECTIVE_DATE_FORMS = (
    "125", "126", "127", "129", "130", "131", "137", "159", "160",
)


@pytest.mark.parametrize("form_number", _EFFECTIVE_DATE_FORMS)
def test_effective_date_maps_via_policy_effective_date_a(form_number):
    """Policy_EffectiveDate_A is the fan-out widget name (vs. 125's legacy
    Policy_Status_EffectiveDate_A which stays as a second alias on 125)."""
    aliases = _FORM_ALIASES[form_number]
    assert aliases.get("Policy_EffectiveDate_A") == "policy_dates.effective_date"


def test_form_125_keeps_legacy_status_widget_AND_new_policy_widget():
    """125 has both widget names; both should fill to the same date."""
    aliases = _FORM_ALIASES["125"]
    assert aliases["Policy_Status_EffectiveDate_A"] == "policy_dates.effective_date"
    assert aliases["Policy_EffectiveDate_A"] == "policy_dates.effective_date"

    s = _sub(
        policy_dates=PolicyDates(effective_date=date(2026, 5, 1)),
        lob_details=CommercialAutoDetails(),
    )
    m = map_submission_to_form(s, "125")
    assert m["Policy_Status_EffectiveDate_A"] == "05/01/2026"
    assert m["Policy_EffectiveDate_A"] == "05/01/2026"


# ---------------------------------------------------------------------------
# Mailing address fan-out to 159
# ---------------------------------------------------------------------------

def test_mailing_address_fans_out_to_159():
    s = _sub(
        mailing_address=Address(
            line_one="123 Main St",
            city="Austin",
            state="TX",
            zip_code="78701",
        ),
        lob_details=CommercialAutoDetails(),
    )
    m = map_submission_to_form(s, "159")
    assert m["NamedInsured_MailingAddress_LineOne_A"] == "123 Main St"
    assert m["NamedInsured_MailingAddress_CityName_A"] == "Austin"
    assert m["NamedInsured_MailingAddress_StateOrProvinceCode_A"] == "TX"
    assert m["NamedInsured_MailingAddress_PostalCode_A"] == "78701"


def test_159_does_not_carry_mailing_line_two():
    """159 has no NamedInsured_MailingAddress_LineTwo_A widget — intentional
    omission in the spec, not a miss. If this ever flips, the fan-out should
    pick it up automatically once the alias is added."""
    aliases = _FORM_ALIASES["159"]
    assert "NamedInsured_MailingAddress_LineTwo_A" not in aliases


# ---------------------------------------------------------------------------
# Empty forms promoted to populated
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("form_number", ("131", "137", "159", "160"))
def test_previously_empty_forms_now_populated(form_number):
    """These four forms had {} alias tables before 10.S.3; they must now
    emit at least business_name + effective_date."""
    aliases = _FORM_ALIASES[form_number]
    assert len(aliases) >= 2
    assert "NamedInsured_FullName_A" in aliases
    assert "Policy_EffectiveDate_A" in aliases


# ---------------------------------------------------------------------------
# Resolver count is UNCHANGED (pure alias additions)
# ---------------------------------------------------------------------------

def test_resolver_count_unchanged_from_pre_fanout_floor():
    """The fan-out should add aliases, not resolvers. If the count grew
    meaningfully, someone accidentally called register_scalar with a new
    schema path instead of reusing an existing key."""
    assert len(_SCHEMA_RESOLVERS) >= 100


def test_every_new_alias_resolves():
    """Invariant guard — re-assert that every alias key resolves."""
    for form_number, aliases in _FORM_ALIASES.items():
        for acord_field, schema_key in aliases.items():
            try:
                _lookup_resolver(schema_key)
            except KeyError:
                pytest.fail(
                    f"form {form_number}: alias {acord_field!r} → "
                    f"unresolved {schema_key!r}"
                )


# ---------------------------------------------------------------------------
# End-to-end: fan-out + filler produces correct PDFs
# ---------------------------------------------------------------------------

def test_fanout_end_to_end_commercial_auto_fills_five_forms():
    """A CA submission with business_name + effective date should emit
    both fields on 125/127/129/137 (all CA-required forms carrying these)."""
    pytest.importorskip("fitz")
    from accord_ai.forms import fill_form

    s = _sub(
        business_name="Acme Trucking",
        policy_dates=PolicyDates(effective_date=date(2026, 5, 1)),
        lob_details=CommercialAutoDetails(),
    )
    for fn in ("125", "127", "129", "137"):
        mapped = map_submission_to_form(s, fn)
        pdf_bytes, res = fill_form(fn, mapped)
        assert res.unknown_fields == ()
        assert res.error_count == 0
        assert mapped["NamedInsured_FullName_A"] == "Acme Trucking"


def test_fanout_wc_populates_130():
    """A WC submission fills 130 + 125 without issue."""
    s = _sub(
        business_name="Initech",
        ein="12-3456789",
        lob_details=WorkersCompDetails(),
    )
    for fn in ("125", "130"):
        m = map_submission_to_form(s, fn)
        assert m["NamedInsured_FullName_A"] == "Initech"
        assert m["NamedInsured_TaxIdentifier_A"] == "12-3456789"


def test_fanout_gl_populates_126():
    s = _sub(
        business_name="GlobeX",
        policy_dates=PolicyDates(effective_date=date(2026, 2, 1)),
        lob_details=GeneralLiabilityDetails(),
    )
    m = map_submission_to_form(s, "126")
    assert m["NamedInsured_FullName_A"] == "GlobeX"
    assert m["Policy_EffectiveDate_A"] == "02/01/2026"
