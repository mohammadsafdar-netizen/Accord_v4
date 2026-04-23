"""Tests for contact fan-out + role-based resolution (P10.S.4)."""
from __future__ import annotations

import pytest

from accord_ai.forms import fill_form
from accord_ai.forms.mapper import (
    _COMPUTED_RESOLVERS,
    _FORM_ALIASES,
    _ROLE_SYNONYMS,
    _lookup_resolver,
    _matches_role,
    contact_by_role,
    fmt_phone,
    map_submission_to_form,
)
from accord_ai.schema import (
    Contact,
    CustomerSubmission,
    WorkersCompDetails,
)


# ---------------------------------------------------------------------------
# _matches_role — canonical + synonym handling
# ---------------------------------------------------------------------------

def test_matches_role_exact_case_insensitive():
    assert _matches_role("accounting", "accounting")
    assert _matches_role("Accounting", "accounting")
    assert _matches_role("  ACCOUNTING  ", "accounting")


def test_matches_role_synonyms():
    assert _matches_role("claims", "claim")
    assert _matches_role("loss", "claim")
    assert _matches_role("accounts", "accounting")
    assert _matches_role("AP", "accounting")


def test_matches_role_unknown_returns_false():
    assert not _matches_role("random_role", "accounting")
    assert not _matches_role(None, "accounting")
    assert not _matches_role("", "accounting")


def test_matches_role_falls_back_to_exact_when_no_synonym_entry():
    """Roles without a synonym table entry match only their exact canonical
    string (case-insensitive). Guards against silent 'anything matches' bugs."""
    assert _matches_role("ceo", "ceo")
    assert _matches_role("CEO", "ceo")
    assert not _matches_role("chief executive", "ceo")


# ---------------------------------------------------------------------------
# contact_by_role — happy path + edge cases
# ---------------------------------------------------------------------------

def test_contact_by_role_finds_first_match():
    sub = CustomerSubmission(contacts=[
        Contact(full_name="Alice", phone="512-555-0001", role="primary"),
        Contact(full_name="Bob",   phone="512-555-0002", role="accounting"),
        Contact(full_name="Carol", phone="512-555-0003", role="accounting"),  # shadowed
    ])
    resolver = contact_by_role("accounting", "full_name")
    assert resolver(sub) == "Bob"


def test_contact_by_role_synonym_matching():
    sub = CustomerSubmission(contacts=[
        Contact(full_name="Dana", email="d@x.test", role="claims"),    # synonym for claim
    ])
    resolver = contact_by_role("claim", "email")
    assert resolver(sub) == "d@x.test"


def test_contact_by_role_returns_none_when_no_match():
    sub = CustomerSubmission(contacts=[
        Contact(full_name="Eve", role="primary"),
    ])
    resolver = contact_by_role("accounting", "full_name")
    assert resolver(sub) is None


def test_contact_by_role_returns_none_when_leaf_missing():
    sub = CustomerSubmission(contacts=[
        Contact(role="accounting"),   # full_name is None
    ])
    resolver = contact_by_role("accounting", "full_name")
    assert resolver(sub) is None


def test_contact_by_role_empty_contacts_list():
    sub = CustomerSubmission(contacts=[])
    resolver = contact_by_role("accounting", "full_name")
    assert resolver(sub) is None


def test_contact_by_role_formatter_applied():
    sub = CustomerSubmission(contacts=[
        Contact(phone="  512-555-0100  ", role="claim"),
    ])
    resolver = contact_by_role("claim", "phone", fmt_phone)
    assert resolver(sub) == "512-555-0100"      # stripped by fmt_phone


# ---------------------------------------------------------------------------
# Registration — the 9 role-specific computed resolvers exist
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", [
    "@contact.accounting.full_name",
    "@contact.accounting.phone",
    "@contact.accounting.email",
    "@contact.claim.full_name",
    "@contact.claim.phone",
    "@contact.claim.email",
    "@contact.inspection.full_name",
    "@contact.inspection.phone",
    "@contact.inspection.email",
])
def test_role_computed_resolver_registered(key):
    assert key in _COMPUTED_RESOLVERS


# ---------------------------------------------------------------------------
# Primary phone/email fan-out (alias only, no new resolvers)
# ---------------------------------------------------------------------------

def test_125_new_primary_phone_alias_uses_existing_phone_resolver():
    """125 now has both NamedInsured_Contact_PrimaryPhoneNumber_A (old)
    AND NamedInsured_Primary_PhoneNumber_A (new). Both must hit the same
    schema key."""
    aliases = _FORM_ALIASES["125"]
    assert aliases["NamedInsured_Contact_PrimaryPhoneNumber_A"] == "phone"
    assert aliases["NamedInsured_Primary_PhoneNumber_A"] == "phone"


def test_130_primary_phone_email_aliases():
    aliases = _FORM_ALIASES["130"]
    assert aliases["NamedInsured_Primary_PhoneNumber_A"] == "phone"
    assert aliases["NamedInsured_Primary_EmailAddress_A"] == "email"


def test_125_populates_both_primary_phone_widgets():
    sub = CustomerSubmission(
        phone="512-555-0100", lob_details=WorkersCompDetails(),
    )
    m = map_submission_to_form(sub, "125")
    assert m["NamedInsured_Contact_PrimaryPhoneNumber_A"] == "512-555-0100"
    assert m["NamedInsured_Primary_PhoneNumber_A"] == "512-555-0100"


# ---------------------------------------------------------------------------
# 130 role-specific fill
# ---------------------------------------------------------------------------

def test_130_role_specific_contacts_fill():
    sub = CustomerSubmission(
        business_name="Acme",
        phone="512-555-0100",
        email="main@acme.test",
        contacts=[
            Contact(full_name="Alice Accounting", phone="512-555-0001",
                    email="ar@acme.test", role="accounting"),
            Contact(full_name="Bob Claims",       phone="512-555-0002",
                    email="claims@acme.test", role="claims"),
            Contact(full_name="Carol Inspector",  phone="512-555-0003",
                    email="uw@acme.test", role="inspection"),
        ],
        lob_details=WorkersCompDetails(),
    )
    m = map_submission_to_form(sub, "130")

    # Primary (alias-only fan-out)
    assert m["NamedInsured_Primary_PhoneNumber_A"] == "512-555-0100"
    assert m["NamedInsured_Primary_EmailAddress_A"] == "main@acme.test"

    # Accounting role-specific
    assert m["NamedInsured_AccountingContact_FullName_A"] == "Alice Accounting"
    assert m["NamedInsured_AccountingContact_PhoneNumber_A"] == "512-555-0001"
    assert m["NamedInsured_AccountingContact_EmailAddress_A"] == "ar@acme.test"

    # Claim role-specific (synonym: "claims" → "claim")
    assert m["NamedInsured_ClaimContact_FullName_A"] == "Bob Claims"
    assert m["NamedInsured_ClaimContact_PhoneNumber_A"] == "512-555-0002"

    # Inspection role-specific
    assert m["NamedInsured_InspectionContact_FullName_A"] == "Carol Inspector"
    assert m["NamedInsured_InspectionContact_EmailAddress_A"] == "uw@acme.test"


def test_130_missing_roles_leaves_widgets_empty():
    sub = CustomerSubmission(
        business_name="Acme",
        contacts=[Contact(full_name="Only Primary", role="primary")],
        lob_details=WorkersCompDetails(),
    )
    m = map_submission_to_form(sub, "130")
    # No accounting/claim/inspection contact → those widgets omitted.
    for widget in (
        "NamedInsured_AccountingContact_FullName_A",
        "NamedInsured_ClaimContact_FullName_A",
        "NamedInsured_InspectionContact_FullName_A",
    ):
        assert widget not in m


# ---------------------------------------------------------------------------
# Invariant sweep
# ---------------------------------------------------------------------------

def test_every_new_alias_resolves():
    for form_number, aliases in _FORM_ALIASES.items():
        for acord_field, schema_key in aliases.items():
            try:
                _lookup_resolver(schema_key)
            except KeyError:
                pytest.fail(
                    f"{form_number}/{acord_field} → unresolved {schema_key!r}"
                )


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

def test_130_fill_end_to_end_with_roles():
    pytest.importorskip("fitz")
    sub = CustomerSubmission(
        business_name="Acme",
        phone="512-555-0100",
        contacts=[
            Contact(full_name="Bob Claims", email="claims@acme.test", role="claim"),
        ],
        lob_details=WorkersCompDetails(),
    )
    mapped = map_submission_to_form(sub, "130")
    pdf_bytes, res = fill_form("130", mapped)
    assert res.unknown_fields == ()
    assert res.error_count == 0
    assert pdf_bytes.startswith(b"%PDF-")


# ---------------------------------------------------------------------------
# Role synonym table doesn't overlap (defensive — prevents "claims" also
# matching "accounting" through a typo)
# ---------------------------------------------------------------------------

def test_role_synonym_table_has_no_cross_overlap():
    seen: dict = {}
    for canonical, synonyms in _ROLE_SYNONYMS.items():
        for syn in synonyms:
            assert syn not in seen, (
                f"synonym {syn!r} appears under both "
                f"{seen[syn]!r} and {canonical!r}"
            )
            seen[syn] = canonical
