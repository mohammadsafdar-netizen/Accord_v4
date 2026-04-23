"""Tests for the schema → ACORD field mapper (P10.A.3)."""
from __future__ import annotations

from datetime import date

import pytest

from accord_ai.forms import (
    UnknownFormError,
    fill_form,
    load_form_spec,
    map_submission,
    map_submission_to_form,
)
from accord_ai.forms.mapper import (
    _FORM_ALIASES,
    _resolve,
    fmt_checkbox,
    fmt_date,
    fmt_money,
    fmt_str,
    path,
)
from accord_ai.schema import (
    Address,
    CommercialAutoDetails,
    CustomerSubmission,
    GeneralLiabilityCoverage,
    GeneralLiabilityDetails,
    PolicyDates,
    WorkersCompCoverage,
    WorkersCompDetails,
)


# ---------------------------------------------------------------------------
# Path resolver
# ---------------------------------------------------------------------------

def test_resolve_scalar():
    s = CustomerSubmission(business_name="Acme")
    assert _resolve(s, "business_name") == "Acme"


def test_resolve_nested():
    s = CustomerSubmission(mailing_address=Address(city="Austin"))
    assert _resolve(s, "mailing_address.city") == "Austin"


def test_resolve_missing_intermediate_returns_none():
    s = CustomerSubmission()   # mailing_address is None
    assert _resolve(s, "mailing_address.city") is None


def test_resolve_missing_leaf_returns_none():
    s = CustomerSubmission(mailing_address=Address())
    assert _resolve(s, "mailing_address.city") is None


def test_resolve_unknown_attribute_returns_none():
    s = CustomerSubmission()
    assert _resolve(s, "nonexistent_attribute") is None


def test_resolve_rejects_invalid_segment():
    # Brackets became valid in 3b (list indexing), so the invalid example
    # needs to be something the new regex also rejects — a segment starting
    # with a digit is the simplest.
    with pytest.raises(ValueError):
        _resolve(CustomerSubmission(), "foo.0bad_segment")


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def test_fmt_str_strips_and_empties_to_none():
    assert fmt_str("  hi  ") == "hi"
    assert fmt_str("") is None
    assert fmt_str("   ") is None
    assert fmt_str(None) is None


def test_fmt_date_mdy():
    assert fmt_date(date(2026, 3, 5)) == "03/05/2026"
    assert fmt_date(None) is None


def test_fmt_money_comma_grouped_no_dollar():
    assert fmt_money(1_234_567) == "1,234,567"
    assert fmt_money(0) == "0"
    assert fmt_money(None) is None


def test_fmt_checkbox_only_true_emits():
    assert fmt_checkbox(True) is True
    assert fmt_checkbox(False) is None
    assert fmt_checkbox(None) is None


def test_path_helper_applies_formatter():
    resolver = path("policy_dates.effective_date", fmt_date)
    s = CustomerSubmission(
        policy_dates=PolicyDates(effective_date=date(2026, 1, 1)),
    )
    assert resolver(s) == "01/01/2026"


# ---------------------------------------------------------------------------
# Form 125 mapping — shared fields
# ---------------------------------------------------------------------------

def _populated_submission_ca() -> CustomerSubmission:
    return CustomerSubmission(
        business_name="Acme Trucking LLC",
        dba="Acme Transport",
        ein="12-3456789",
        email="ops@acme.test",
        phone="512-555-0100",
        mailing_address=Address(
            line_one="123 Main St",
            city="Austin",
            state="TX",
            zip_code="78701",
        ),
        policy_dates=PolicyDates(
            effective_date=date(2026, 5, 1),
            expiration_date=date(2027, 5, 1),
        ),
        lob_details=CommercialAutoDetails(),
    )


def test_form_125_emits_identity_fields():
    m = map_submission_to_form(_populated_submission_ca(), "125")
    assert m["NamedInsured_FullName_A"] == "Acme Trucking LLC"
    assert m["NamedInsured_TaxIdentifier_A"] == "12-3456789"
    assert m["NamedInsured_Contact_PrimaryEmailAddress_A"] == "ops@acme.test"
    assert m["NamedInsured_Contact_PrimaryPhoneNumber_A"] == "512-555-0100"


def test_form_125_emits_address():
    m = map_submission_to_form(_populated_submission_ca(), "125")
    assert m["NamedInsured_MailingAddress_LineOne_A"] == "123 Main St"
    assert m["NamedInsured_MailingAddress_CityName_A"] == "Austin"
    assert m["NamedInsured_MailingAddress_StateOrProvinceCode_A"] == "TX"
    assert m["NamedInsured_MailingAddress_PostalCode_A"] == "78701"
    # None → omitted (line_two not populated).
    assert "NamedInsured_MailingAddress_LineTwo_A" not in m


def test_form_125_emits_policy_dates_formatted():
    m = map_submission_to_form(_populated_submission_ca(), "125")
    assert m["Policy_Status_EffectiveDate_A"] == "05/01/2026"
    assert m["Policy_ExpirationDate_A"] == "05/01/2027"


def test_form_125_ca_lob_indicator():
    m = map_submission_to_form(_populated_submission_ca(), "125")
    assert m["Policy_LineOfBusiness_BusinessAutoIndicator_A"] is True
    assert "Policy_LineOfBusiness_CommercialGeneralLiability_A" not in m


def test_form_125_gl_lob_indicator():
    s = CustomerSubmission(
        business_name="Acme",
        lob_details=GeneralLiabilityDetails(),
    )
    m = map_submission_to_form(s, "125")
    assert m["Policy_LineOfBusiness_CommercialGeneralLiability_A"] is True
    assert "Policy_LineOfBusiness_BusinessAutoIndicator_A" not in m


def test_form_125_no_lob_details_no_indicator():
    s = CustomerSubmission(business_name="Acme")
    m = map_submission_to_form(s, "125")
    assert "Policy_LineOfBusiness_BusinessAutoIndicator_A" not in m
    assert "Policy_LineOfBusiness_CommercialGeneralLiability_A" not in m
    assert m["NamedInsured_FullName_A"] == "Acme"


def test_empty_submission_produces_empty_mapping():
    s = CustomerSubmission()
    m = map_submission_to_form(s, "125")
    assert m == {}


# ---------------------------------------------------------------------------
# Form 126 (GL) mapping
# ---------------------------------------------------------------------------

def test_form_126_claims_made_true_emits_claims_made_only():
    s = CustomerSubmission(
        lob_details=GeneralLiabilityDetails(
            coverage=GeneralLiabilityCoverage(
                claims_made_basis=True,
                each_occurrence_limit=1_000_000,
                general_aggregate_limit=2_000_000,
            ),
        ),
    )
    m = map_submission_to_form(s, "126")
    assert m["GeneralLiability_ClaimsMadeIndicator_A"] is True
    assert "GeneralLiability_OccurrenceIndicator_A" not in m
    assert m["GeneralLiability_EachOccurrence_LimitAmount_A"] == "1,000,000"
    assert m["GeneralLiability_GeneralAggregate_LimitAmount_A"] == "2,000,000"


def test_form_126_claims_made_false_emits_occurrence_only():
    s = CustomerSubmission(
        lob_details=GeneralLiabilityDetails(
            coverage=GeneralLiabilityCoverage(claims_made_basis=False),
        ),
    )
    m = map_submission_to_form(s, "126")
    assert m["GeneralLiability_OccurrenceIndicator_A"] is True
    assert "GeneralLiability_ClaimsMadeIndicator_A" not in m


def test_form_126_claims_made_unset_emits_neither():
    s = CustomerSubmission(lob_details=GeneralLiabilityDetails())
    m = map_submission_to_form(s, "126")
    assert "GeneralLiability_ClaimsMadeIndicator_A" not in m
    assert "GeneralLiability_OccurrenceIndicator_A" not in m


# ---------------------------------------------------------------------------
# Form 130 (WC) mapping
# ---------------------------------------------------------------------------

def test_form_130_emits_employers_liability_limits():
    s = CustomerSubmission(
        policy_dates=PolicyDates(effective_date=date(2026, 1, 1)),
        lob_details=WorkersCompDetails(
            coverage=WorkersCompCoverage(
                employers_liability_per_accident=500_000,
                employers_liability_per_employee=500_000,
                employers_liability_per_policy=500_000,
            ),
        ),
    )
    m = map_submission_to_form(s, "130")
    assert m["Policy_EffectiveDate_A"] == "01/01/2026"
    el_accident = "WorkersCompensationEmployersLiability_EmployersLiability_EachAccidentLimitAmount_A"
    assert m[el_accident] == "500,000"


# ---------------------------------------------------------------------------
# map_submission (LOB dispatch)
# ---------------------------------------------------------------------------

def test_map_submission_ca_populates_125_and_129():
    s = _populated_submission_ca()
    out = map_submission(s)
    assert set(out.keys()) == {"125", "127", "129", "137", "163"}
    assert out["125"]["NamedInsured_FullName_A"] == "Acme Trucking LLC"
    assert out["129"]["Policy_EffectiveDate_A"] == "05/01/2026"
    # 137 was {} in 3b — fan-out (10.S.3) now emits business_name + effective_date.
    assert out["137"]["NamedInsured_FullName_A"] == "Acme Trucking LLC"
    assert out["137"]["Policy_EffectiveDate_A"] == "05/01/2026"
    # 163 still deferred (coord-layout in 10.S.10) — empty is expected.
    assert out["163"] == {}


def test_map_submission_gl_populates_125_and_126():
    s = CustomerSubmission(
        business_name="GlobeX",
        policy_dates=PolicyDates(effective_date=date(2026, 2, 1)),
        lob_details=GeneralLiabilityDetails(
            coverage=GeneralLiabilityCoverage(each_occurrence_limit=1_000_000),
        ),
    )
    out = map_submission(s)
    assert set(out.keys()) == {"125", "126"}
    assert out["126"]["GeneralLiability_EachOccurrence_LimitAmount_A"] == "1,000,000"


def test_map_submission_wc_populates_125_and_130():
    s = CustomerSubmission(
        business_name="Initech",
        lob_details=WorkersCompDetails(),
    )
    out = map_submission(s)
    assert set(out.keys()) == {"125", "130"}


def test_map_submission_no_lob_returns_empty():
    s = CustomerSubmission(business_name="Acme")
    assert map_submission(s) == {}


# ---------------------------------------------------------------------------
# Unknown forms + integrity checks
# ---------------------------------------------------------------------------

def test_map_submission_to_form_unknown_form_raises():
    with pytest.raises(UnknownFormError):
        map_submission_to_form(_populated_submission_ca(), "999")


def test_every_mapped_field_exists_in_registry_spec():
    """Invariant: every ACORD field name in a mapping must be a real widget
    on the corresponding blank PDF. If a template version changes and a
    field disappears, this test surfaces it before production."""
    for form_number, mapping in _FORM_ALIASES.items():
        if not mapping:
            continue
        spec = load_form_spec(form_number)
        for acord_field in mapping:
            assert acord_field in spec.fields, (
                f"form {form_number}: mapped field {acord_field!r} not in spec"
            )


def test_resolver_exception_is_swallowed(caplog):
    """A buggy resolver must not kill the whole mapping.

    Post-10.S.2: alias values are schema keys (strings), not callables —
    inject via _COMPUTED_RESOLVERS + a @-prefixed alias so the mapper
    looks it up through _lookup_resolver.
    """
    import logging
    from accord_ai.forms.mapper import _COMPUTED_RESOLVERS, _FORM_ALIASES

    _COMPUTED_RESOLVERS["@bogus"] = lambda s: 1 / 0      # noqa: E731
    original = dict(_FORM_ALIASES["125"])
    _FORM_ALIASES["125"]["__bogus_field__"] = "@bogus"
    try:
        caplog.set_level(logging.ERROR, logger="accord_ai.forms.mapper")
        result = map_submission_to_form(_populated_submission_ca(), "125")
        assert "__bogus_field__" not in result
        assert result["NamedInsured_FullName_A"] == "Acme Trucking LLC"
    finally:
        _COMPUTED_RESOLVERS.pop("@bogus", None)
        _FORM_ALIASES["125"].clear()
        _FORM_ALIASES["125"].update(original)


# ---------------------------------------------------------------------------
# End-to-end: mapper → filler
# ---------------------------------------------------------------------------

def test_mapper_output_feeds_filler_without_unknowns():
    """Integration: fill a real ACORD 125 with mapper output and verify
    the filler reports zero unknown fields (every mapped name exists)."""
    pytest.importorskip("fitz")
    s = _populated_submission_ca()
    mapped = map_submission_to_form(s, "125")
    assert mapped  # precondition

    pdf_bytes, res = fill_form("125", mapped)
    assert res.unknown_fields == ()
    assert res.filled_count == len(mapped)
    assert pdf_bytes.startswith(b"%PDF-")
