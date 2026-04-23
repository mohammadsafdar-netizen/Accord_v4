"""Tests for producer fan-out (P10.S.5)."""
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
    Producer,
)


def _sub_with_producer() -> CustomerSubmission:
    return CustomerSubmission(
        business_name="Acme Trucking",
        producer=Producer(
            agency_name="Brokerage Co",
            contact_name="Jane Broker",
            phone="512-555-0100",
            email="jane@broker.test",
            producer_code="CUST-12345",
            license_number="TX-987654",
            mailing_address=Address(
                line_one="500 Oak St",
                line_two="Suite 300",
                city="Dallas",
                state="TX",
                zip_code="75201",
            ),
        ),
        lob_details=CommercialAutoDetails(),
    )


# ---------------------------------------------------------------------------
# Fan-out topology: Producer_FullName_A on 9 forms
# ---------------------------------------------------------------------------

_PRODUCER_FULLNAME_FORMS = (
    "125", "126", "127", "129", "130", "131", "137", "159", "160",
)


@pytest.mark.parametrize("form_number", _PRODUCER_FULLNAME_FORMS)
def test_producer_fullname_alias_present(form_number):
    assert _FORM_ALIASES[form_number].get("Producer_FullName_A") == "producer.agency_name"


def test_producer_fullname_shares_one_resolver_across_all_nine_forms():
    resolver_ids = set()
    for form_number in _PRODUCER_FULLNAME_FORMS:
        schema_key = _FORM_ALIASES[form_number]["Producer_FullName_A"]
        resolver_ids.add(id(_lookup_resolver(schema_key)))
    assert len(resolver_ids) == 1


@pytest.mark.parametrize("form_number", _PRODUCER_FULLNAME_FORMS)
def test_producer_fullname_fills_on_every_form(form_number):
    s = _sub_with_producer()
    m = map_submission_to_form(s, form_number)
    assert m["Producer_FullName_A"] == "Brokerage Co"


# ---------------------------------------------------------------------------
# Producer_CustomerIdentifier_A on same 9 forms (producer.producer_code)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("form_number", _PRODUCER_FULLNAME_FORMS)
def test_producer_customer_identifier_fills(form_number):
    s = _sub_with_producer()
    m = map_submission_to_form(s, form_number)
    assert m["Producer_CustomerIdentifier_A"] == "CUST-12345"


# ---------------------------------------------------------------------------
# Producer_StateLicenseIdentifier_A on 5 forms (not 129/130/137/159)
# ---------------------------------------------------------------------------

_PRODUCER_LICENSE_FORMS = ("125", "126", "127", "131", "160")


@pytest.mark.parametrize("form_number", _PRODUCER_LICENSE_FORMS)
def test_producer_license_number_fills(form_number):
    s = _sub_with_producer()
    m = map_submission_to_form(s, form_number)
    assert m["Producer_StateLicenseIdentifier_A"] == "TX-987654"


@pytest.mark.parametrize("form_number", ("129", "130", "137", "159"))
def test_producer_license_not_on_forms_lacking_widget(form_number):
    """129/130/137/159 don't carry Producer_StateLicenseIdentifier_A — the
    alias must NOT appear (would fail the registry-integrity test)."""
    assert "Producer_StateLicenseIdentifier_A" not in _FORM_ALIASES[form_number]


# ---------------------------------------------------------------------------
# Producer contact block (full_name, phone, email) on 125/130/159
# ---------------------------------------------------------------------------

_PRODUCER_CONTACT_FORMS = ("125", "130", "159")


@pytest.mark.parametrize("form_number", _PRODUCER_CONTACT_FORMS)
def test_producer_contact_block_fills(form_number):
    s = _sub_with_producer()
    m = map_submission_to_form(s, form_number)
    assert m["Producer_ContactPerson_FullName_A"] == "Jane Broker"
    assert m["Producer_ContactPerson_PhoneNumber_A"] == "512-555-0100"
    assert m["Producer_ContactPerson_EmailAddress_A"] == "jane@broker.test"


# ---------------------------------------------------------------------------
# Producer mailing address on 125/130/159 (all 5 leaves)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("form_number", _PRODUCER_CONTACT_FORMS)
def test_producer_mailing_address_fills(form_number):
    s = _sub_with_producer()
    m = map_submission_to_form(s, form_number)
    assert m["Producer_MailingAddress_LineOne_A"] == "500 Oak St"
    assert m["Producer_MailingAddress_LineTwo_A"] == "Suite 300"
    assert m["Producer_MailingAddress_CityName_A"] == "Dallas"
    assert m["Producer_MailingAddress_StateOrProvinceCode_A"] == "TX"
    assert m["Producer_MailingAddress_PostalCode_A"] == "75201"


# ---------------------------------------------------------------------------
# No-producer submissions emit nothing (producer fields omitted, not blank)
# ---------------------------------------------------------------------------

def test_missing_producer_emits_no_producer_widgets():
    s = CustomerSubmission(
        business_name="Acme",
        lob_details=CommercialAutoDetails(),
    )
    m = map_submission_to_form(s, "125")
    producer_widgets = {k for k in m if k.startswith("Producer_")}
    assert producer_widgets == set()


def test_partial_producer_emits_only_set_fields():
    """Producer with only agency_name — other producer widgets omitted."""
    s = CustomerSubmission(
        business_name="Acme",
        producer=Producer(agency_name="Just An Agency"),
        lob_details=CommercialAutoDetails(),
    )
    m = map_submission_to_form(s, "125")
    assert m["Producer_FullName_A"] == "Just An Agency"
    assert "Producer_ContactPerson_FullName_A" not in m
    assert "Producer_MailingAddress_LineOne_A" not in m


# ---------------------------------------------------------------------------
# Cross-form resolver reuse invariant
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("schema_key", [
    "producer.agency_name",
    "producer.producer_code",
    "producer.license_number",
    "producer.contact_name",
    "producer.phone",
    "producer.email",
    "producer.mailing_address.line_one",
    "producer.mailing_address.city",
])
def test_producer_resolver_shared_across_every_form_using_it(schema_key):
    """Every form that references `schema_key` must hit the same resolver
    object (guarantees 'collect once, fill everywhere')."""
    resolver_ids = set()
    for form_number, aliases in _FORM_ALIASES.items():
        for acord_field, key in aliases.items():
            if key == schema_key:
                resolver_ids.add(id(_lookup_resolver(key)))
    assert len(resolver_ids) >= 1
    assert len(resolver_ids) == 1, (
        f"{schema_key!r} has {len(resolver_ids)} distinct resolvers"
    )


# ---------------------------------------------------------------------------
# Integrity sweep — every new producer alias resolves
# ---------------------------------------------------------------------------

def test_every_producer_alias_resolves():
    for form_number, aliases in _FORM_ALIASES.items():
        for acord_field, schema_key in aliases.items():
            if not acord_field.startswith("Producer_"):
                continue
            try:
                _lookup_resolver(schema_key)
            except KeyError:
                pytest.fail(
                    f"form {form_number}: {acord_field!r} → "
                    f"unresolved {schema_key!r}"
                )


# ---------------------------------------------------------------------------
# End-to-end: producer fields fill real PDFs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("form_number", ("125", "130", "159"))
def test_producer_end_to_end_fill(form_number):
    pytest.importorskip("fitz")
    s = _sub_with_producer()
    mapped = map_submission_to_form(s, form_number)
    pdf_bytes, res = fill_form(form_number, mapped)
    assert res.unknown_fields == ()
    assert res.error_count == 0
    assert pdf_bytes.startswith(b"%PDF-")
