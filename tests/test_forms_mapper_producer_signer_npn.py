"""Tests for producer signer + NPN fan-out (P10.S.6)."""
from __future__ import annotations

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
    Producer,
)


def _sub(npn="NPN-8675309", signer="Jane Broker"):
    return CustomerSubmission(
        business_name="Acme",
        producer=Producer(
            agency_name="Brokerage Co",
            national_producer_number=npn,
            authorized_representative=signer,
        ),
        lob_details=CommercialAutoDetails(),
    )


# ---------------------------------------------------------------------------
# Schema additions
# ---------------------------------------------------------------------------

def test_producer_has_national_producer_number_field():
    p = Producer(national_producer_number="12345678")
    assert p.national_producer_number == "12345678"


def test_producer_has_authorized_representative_field():
    p = Producer(authorized_representative="Signatory Name")
    assert p.authorized_representative == "Signatory Name"


def test_producer_new_fields_default_to_none():
    p = Producer()
    assert p.national_producer_number is None
    assert p.authorized_representative is None


def test_producer_npn_distinct_from_producer_code():
    """Both fields exist independently — they're different identifiers."""
    p = Producer(producer_code="CUST-1", national_producer_number="NPN-2")
    assert p.producer_code == "CUST-1"
    assert p.national_producer_number == "NPN-2"


# ---------------------------------------------------------------------------
# NPN fan-out: 7 forms
# ---------------------------------------------------------------------------

_NPN_FORMS = ("125", "126", "127", "130", "131", "137", "160")


@pytest.mark.parametrize("form_number", _NPN_FORMS)
def test_npn_alias_present(form_number):
    assert (
        _FORM_ALIASES[form_number].get("Producer_NationalIdentifier_A")
        == "producer.national_producer_number"
    )


@pytest.mark.parametrize("form_number", _NPN_FORMS)
def test_npn_fills_on_every_form(form_number):
    m = map_submission_to_form(_sub(), form_number)
    assert m["Producer_NationalIdentifier_A"] == "NPN-8675309"


def test_npn_shares_single_resolver_across_all_seven_forms():
    resolver_ids = set()
    for fn in _NPN_FORMS:
        schema_key = _FORM_ALIASES[fn]["Producer_NationalIdentifier_A"]
        resolver_ids.add(id(_lookup_resolver(schema_key)))
    assert len(resolver_ids) == 1


@pytest.mark.parametrize("form_number", ("129", "159"))
def test_npn_absent_on_forms_lacking_widget(form_number):
    """129 and 159 don't carry Producer_NationalIdentifier_A."""
    assert "Producer_NationalIdentifier_A" not in _FORM_ALIASES[form_number]


# ---------------------------------------------------------------------------
# Signer fan-out: 5 forms
# ---------------------------------------------------------------------------

_SIGNER_FORMS = ("125", "126", "127", "131", "160")


@pytest.mark.parametrize("form_number", _SIGNER_FORMS)
def test_signer_alias_present(form_number):
    assert (
        _FORM_ALIASES[form_number].get(
            "Producer_AuthorizedRepresentative_FullName_A"
        )
        == "producer.authorized_representative"
    )


@pytest.mark.parametrize("form_number", _SIGNER_FORMS)
def test_signer_fills_on_every_form(form_number):
    m = map_submission_to_form(_sub(), form_number)
    assert m["Producer_AuthorizedRepresentative_FullName_A"] == "Jane Broker"


def test_signer_shares_single_resolver_across_all_five_forms():
    resolver_ids = set()
    for fn in _SIGNER_FORMS:
        schema_key = _FORM_ALIASES[fn][
            "Producer_AuthorizedRepresentative_FullName_A"
        ]
        resolver_ids.add(id(_lookup_resolver(schema_key)))
    assert len(resolver_ids) == 1


@pytest.mark.parametrize("form_number", ("129", "130", "137", "159"))
def test_signer_absent_on_forms_lacking_widget(form_number):
    """129/130/137/159 don't carry the authorized-representative name widget."""
    assert (
        "Producer_AuthorizedRepresentative_FullName_A"
        not in _FORM_ALIASES[form_number]
    )


# ---------------------------------------------------------------------------
# Signer field is distinct from contact_name — independent fills
# ---------------------------------------------------------------------------

def test_signer_and_contact_name_are_independent():
    """Fill both; confirm distinct widgets get distinct values."""
    s = CustomerSubmission(
        business_name="Acme",
        producer=Producer(
            agency_name="Brokerage Co",
            contact_name="Contact Person",
            authorized_representative="Signatory Name",
        ),
        lob_details=CommercialAutoDetails(),
    )
    m = map_submission_to_form(s, "125")
    assert m["Producer_ContactPerson_FullName_A"] == "Contact Person"
    assert m["Producer_AuthorizedRepresentative_FullName_A"] == "Signatory Name"


def test_npn_and_producer_code_are_independent():
    s = CustomerSubmission(
        business_name="Acme",
        producer=Producer(
            producer_code="CUST-INTERNAL",
            national_producer_number="NPN-EXTERNAL",
        ),
        lob_details=CommercialAutoDetails(),
    )
    m = map_submission_to_form(s, "125")
    assert m["Producer_CustomerIdentifier_A"] == "CUST-INTERNAL"
    assert m["Producer_NationalIdentifier_A"] == "NPN-EXTERNAL"


# ---------------------------------------------------------------------------
# Negative: missing producer or missing field emits no widget
# ---------------------------------------------------------------------------

def test_missing_signer_emits_no_signer_widget():
    s = CustomerSubmission(
        business_name="Acme",
        producer=Producer(agency_name="Brokerage Co"),  # no signer
        lob_details=CommercialAutoDetails(),
    )
    m = map_submission_to_form(s, "125")
    assert "Producer_AuthorizedRepresentative_FullName_A" not in m


def test_missing_npn_emits_no_npn_widget():
    s = CustomerSubmission(
        business_name="Acme",
        producer=Producer(agency_name="Brokerage Co"),  # no NPN
        lob_details=CommercialAutoDetails(),
    )
    m = map_submission_to_form(s, "125")
    assert "Producer_NationalIdentifier_A" not in m


# ---------------------------------------------------------------------------
# Integrity sweep — every new alias resolves
# ---------------------------------------------------------------------------

def test_every_new_alias_resolves():
    new_widgets = (
        "Producer_NationalIdentifier_A",
        "Producer_AuthorizedRepresentative_FullName_A",
    )
    for form_number, aliases in _FORM_ALIASES.items():
        for widget in new_widgets:
            if widget in aliases:
                try:
                    _lookup_resolver(aliases[widget])
                except KeyError:
                    pytest.fail(
                        f"{form_number}/{widget} → unresolved {aliases[widget]!r}"
                    )


# ---------------------------------------------------------------------------
# End-to-end: both fields fill real PDFs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("form_number", ("125", "126", "127", "131", "160"))
def test_signer_end_to_end_fill(form_number):
    pytest.importorskip("fitz")
    mapped = map_submission_to_form(_sub(), form_number)
    pdf_bytes, res = fill_form(form_number, mapped)
    assert res.unknown_fields == ()
    assert res.error_count == 0
    assert pdf_bytes.startswith(b"%PDF-")


@pytest.mark.parametrize("form_number", _NPN_FORMS)
def test_npn_end_to_end_fill(form_number):
    pytest.importorskip("fitz")
    mapped = map_submission_to_form(_sub(), form_number)
    pdf_bytes, res = fill_form(form_number, mapped)
    assert res.unknown_fields == ()
