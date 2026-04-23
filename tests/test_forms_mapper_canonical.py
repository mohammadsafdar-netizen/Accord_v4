"""Tests for the canonical/alias mapper architecture (P10.S.2)."""
from __future__ import annotations

import pytest

from accord_ai.forms import load_form_spec
from accord_ai.forms.mapper import (
    _COMPUTED_RESOLVERS,
    _FORM_ALIASES,
    _SCHEMA_RESOLVERS,
    _lookup_resolver,
    array_aliases,
    fmt_money,
    fmt_str,
    map_submission_to_form,
    register_computed,
    register_scalar,
)
from accord_ai.schema import (
    CommercialAutoDetails,
    CustomerSubmission,
    GeneralLiabilityDetails,
)


# ---------------------------------------------------------------------------
# Registry + lookup invariants
# ---------------------------------------------------------------------------

def test_register_scalar_is_idempotent():
    key1 = register_scalar("business_name", fmt_str)
    key2 = register_scalar("business_name", fmt_str)
    assert key1 == key2 == "business_name"
    r1 = _SCHEMA_RESOLVERS["business_name"]
    register_scalar("business_name", fmt_str)
    # Same resolver object — no duplication.
    assert _SCHEMA_RESOLVERS["business_name"] is r1


def test_register_scalar_conflict_warns_and_keeps_first(caplog):
    """configure_logging() (invoked by test_api fixtures) disables propagate
    on the accord_ai logger, so plain caplog misses child-logger records in
    suite-order. Attach the caplog handler directly to 'accord_ai.forms.mapper'."""
    import logging
    mapper_logger = logging.getLogger("accord_ai.forms.mapper")
    mapper_logger.addHandler(caplog.handler)
    original_level = mapper_logger.level
    mapper_logger.setLevel(logging.DEBUG)
    try:
        register_scalar("__test_conflict__", fmt_str)
        register_scalar("__test_conflict__", fmt_money)
        assert any(
            "ignoring conflicting formatter" in r.getMessage()
            for r in caplog.records
        )
        # First registration wins — formatter name is still the original.
        first = _SCHEMA_RESOLVERS["__test_conflict__"]
        assert getattr(first, "_formatter_name", None) == "fmt_str"
    finally:
        mapper_logger.removeHandler(caplog.handler)
        mapper_logger.setLevel(original_level)
        _SCHEMA_RESOLVERS.pop("__test_conflict__", None)


def test_register_computed_returns_prefixed_key():
    key = register_computed("__test_computed__", lambda s: True)
    assert key == "@__test_computed__"
    assert _COMPUTED_RESOLVERS[key] is not None
    _COMPUTED_RESOLVERS.pop(key, None)


def test_lookup_resolver_handles_both_namespaces():
    assert (
        _lookup_resolver("business_name")
        is _SCHEMA_RESOLVERS["business_name"]
    )
    assert (
        _lookup_resolver("@lob.commercial_auto")
        is _COMPUTED_RESOLVERS["@lob.commercial_auto"]
    )


def test_lookup_resolver_unknown_raises():
    with pytest.raises(KeyError):
        _lookup_resolver("nonexistent.path")
    with pytest.raises(KeyError):
        _lookup_resolver("@nonexistent")


# ---------------------------------------------------------------------------
# "Collect once, fill everywhere" — the whole point of the refactor
# ---------------------------------------------------------------------------

def test_business_name_resolver_shared_across_forms():
    """Same schema key on every form that carries NamedInsured_FullName_A
    → same resolver object in the registry. This is the invariant that
    makes 'collect once, fill everywhere' actually work."""
    keys_using_business_name = []
    for form_number, aliases in _FORM_ALIASES.items():
        for acord_field, schema_key in aliases.items():
            if schema_key == "business_name":
                keys_using_business_name.append((form_number, acord_field))

    # business_name is on 125 today; 10.S.3 will fan it out. Assertion is
    # "at least one form uses it, and every use resolves to the same
    # resolver object" — the guarantee holds as more forms adopt it.
    assert len(keys_using_business_name) >= 1
    resolvers = {
        id(_lookup_resolver(_FORM_ALIASES[f][a]))
        for f, a in keys_using_business_name
    }
    assert len(resolvers) == 1, "business_name has divergent resolvers"


def test_cross_form_policy_effective_date_is_shared():
    """Policy effective date is on 125/126/129/130. All must share one resolver."""
    resolver_ids = set()
    for form_number in ("125", "126", "129", "130"):
        for acord_field, schema_key in _FORM_ALIASES[form_number].items():
            if schema_key == "policy_dates.effective_date":
                resolver_ids.add(id(_lookup_resolver(schema_key)))
    assert len(resolver_ids) == 1


# ---------------------------------------------------------------------------
# Alias table integrity — every key resolves
# ---------------------------------------------------------------------------

def test_every_alias_key_has_a_resolver():
    for form_number, aliases in _FORM_ALIASES.items():
        for acord_field, schema_key in aliases.items():
            try:
                _lookup_resolver(schema_key)
            except KeyError:
                pytest.fail(
                    f"form {form_number}: alias {acord_field!r} → "
                    f"unresolved key {schema_key!r}"
                )


def test_every_alias_acord_field_exists_in_registry_spec():
    """Carried over from 3a/3b: every mapped widget name must be a real
    widget on the blank PDF."""
    for form_number, aliases in _FORM_ALIASES.items():
        if not aliases:
            continue
        spec = load_form_spec(form_number)
        for acord_field in aliases:
            assert acord_field in spec.fields, (
                f"form {form_number}: mapped field {acord_field!r} "
                f"not in spec"
            )


# ---------------------------------------------------------------------------
# array_aliases — registers + returns alias entries
# ---------------------------------------------------------------------------

def test_array_aliases_registers_canonical_resolvers():
    out = array_aliases(
        "__TestStem__", "lob_details.drivers", "first_name", max_count=3,
    )
    assert out == {
        "__TestStem___A": "lob_details.drivers[0].first_name",
        "__TestStem___B": "lob_details.drivers[1].first_name",
        "__TestStem___C": "lob_details.drivers[2].first_name",
    }
    for schema_key in out.values():
        assert schema_key in _SCHEMA_RESOLVERS


def test_array_aliases_reuses_resolvers_across_stems():
    """Two different ACORD stems that reference the same leaf path
    should share the canonical drivers[0].first_name resolver."""
    r1 = _SCHEMA_RESOLVERS.get("lob_details.drivers[0].first_name")
    array_aliases(
        "__AnotherStem__", "lob_details.drivers", "first_name", max_count=1,
    )
    r2 = _SCHEMA_RESOLVERS["lob_details.drivers[0].first_name"]
    # Idempotent — same resolver object if it existed before, or now cached.
    assert r1 is None or r1 is r2


# ---------------------------------------------------------------------------
# Behavioral equivalence — output must match pre-refactor
# ---------------------------------------------------------------------------

def test_form_125_ca_submission_output_unchanged():
    """Concrete expected output — locks behavior against regressions."""
    from datetime import date
    from accord_ai.schema import Address, PolicyDates
    s = CustomerSubmission(
        business_name="Acme Trucking LLC",
        ein="12-3456789",
        email="ops@acme.test",
        phone="512-555-0100",
        mailing_address=Address(
            line_one="123 Main St", city="Austin",
            state="TX", zip_code="78701",
        ),
        policy_dates=PolicyDates(
            effective_date=date(2026, 5, 1),
            expiration_date=date(2027, 5, 1),
        ),
        lob_details=CommercialAutoDetails(),
    )
    m = map_submission_to_form(s, "125")
    assert m["NamedInsured_FullName_A"] == "Acme Trucking LLC"
    assert m["NamedInsured_TaxIdentifier_A"] == "12-3456789"
    assert m["NamedInsured_MailingAddress_CityName_A"] == "Austin"
    assert m["Policy_Status_EffectiveDate_A"] == "05/01/2026"
    assert m["Policy_LineOfBusiness_BusinessAutoIndicator_A"] is True
    assert "Policy_LineOfBusiness_CommercialGeneralLiability_A" not in m


def test_form_126_gl_occurrence_computed_resolver():
    s = CustomerSubmission(
        lob_details=GeneralLiabilityDetails(),    # claims_made_basis = None
    )
    m = map_submission_to_form(s, "126")
    # Both indicators omitted when claims_made_basis is unset.
    assert "GeneralLiability_ClaimsMadeIndicator_A" not in m
    assert "GeneralLiability_OccurrenceIndicator_A" not in m


# ---------------------------------------------------------------------------
# Registry snapshot
# ---------------------------------------------------------------------------

def test_canonical_resolver_count_at_floor():
    """The refactor should end up with at least N canonical resolvers —
    one per distinct schema path referenced by any alias table.

    Floor: ~13 scalars + 7 driver stems × 8 slots + 5 vehicle stems × 5 slots
    + 4 WC-class stems × 14 slots = ~13 + 56 + 25 + 56 = ~150.
    """
    assert len(_SCHEMA_RESOLVERS) >= 100, (
        f"canonical resolver count {len(_SCHEMA_RESOLVERS)} below floor"
    )
