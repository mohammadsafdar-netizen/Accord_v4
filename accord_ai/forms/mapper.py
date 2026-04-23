"""Schema → ACORD field mapping (P10.S.2 — canonical resolvers + alias tables).

Structure:
  * _SCHEMA_RESOLVERS — one resolver per schema path. "Collect once."
  * _COMPUTED_RESOLVERS — resolvers keyed by @name, for derived values that
    aren't a simple schema walk (LOB discriminator indicators, mutex-style
    checkboxes that key off another field's truthiness).
  * _FORM_{N} — alias table: {acord_field_name: schema_key | @name}.
    The same schema key reused across forms = fill-everywhere.

Helpers:
  * register_scalar(schema_path, formatter) — ensure a resolver is
    registered, return the schema_key to use in alias tables. Idempotent.
  * register_computed(name, resolver_fn) — register under @name, return key.
  * array_aliases(stem, list_path, leaf, max_count, formatter) — register
    N array-slot resolvers AND return the alias dict to merge into the form.

Adding a new field is a 1-line change on most forms (alias entry only —
resolver gets registered by register_scalar side-effect). The canonical
resolver registry makes the fan-out topology explicit.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Callable, Dict, Mapping, Optional, Union

from accord_ai.forms.registry import forms_for_lob, load_form_spec
from accord_ai.logging_config import get_logger
from accord_ai.schema import CustomerSubmission

_logger = get_logger("forms.mapper")

FieldValue = Union[str, bool]
Resolver = Callable[[CustomerSubmission], Optional[FieldValue]]


# ---------------------------------------------------------------------------
# Path resolver (unchanged from 3b)
# ---------------------------------------------------------------------------

_SEGMENT_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\[(\d+)\])?$")


def _resolve(obj: Any, path_: str) -> Any:
    segments = []
    for raw in path_.split("."):
        m = _SEGMENT_RE.match(raw)
        if not m:
            raise ValueError(f"invalid path segment: {raw!r} in {path_!r}")
        segments.append((m.group(1), m.group(2)))

    current = obj
    for attr, idx in segments:
        if current is None:
            return None
        current = getattr(current, attr, None)
        if idx is not None:
            if current is None:
                return None
            try:
                current = current[int(idx)]
            except (IndexError, TypeError, KeyError):
                return None
    return current


# ---------------------------------------------------------------------------
# Formatters (unchanged from 3a)
# ---------------------------------------------------------------------------

def fmt_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def fmt_date(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, datetime):
        v = v.date()
    if isinstance(v, date):
        return v.strftime("%m/%d/%Y")
    return fmt_str(v)


def fmt_money(v: Any) -> Optional[str]:
    if v is None:
        return None
    try:
        return f"{int(v):,}"
    except (TypeError, ValueError):
        return fmt_str(v)


def fmt_int(v: Any) -> Optional[str]:
    if v is None:
        return None
    try:
        return str(int(v))
    except (TypeError, ValueError):
        return fmt_str(v)


def fmt_float(v: Any) -> Optional[str]:
    if v is None:
        return None
    try:
        return f"{float(v):g}"
    except (TypeError, ValueError):
        return fmt_str(v)


def fmt_checkbox(v: Any) -> Optional[bool]:
    return True if v is True else None


def fmt_phone(v: Any) -> Optional[str]:
    return fmt_str(v)


# ---------------------------------------------------------------------------
# Resolver builder — the 95% case.
# ---------------------------------------------------------------------------

def path(
    schema_path: str,
    formatter: Callable[[Any], Optional[FieldValue]] = fmt_str,
) -> Resolver:
    def _get(sub: CustomerSubmission) -> Optional[FieldValue]:
        return formatter(_resolve(sub, schema_path))
    _get.__name__ = f"path({schema_path})"
    return _get


# ---------------------------------------------------------------------------
# Canonical resolver registry
# ---------------------------------------------------------------------------

_SCHEMA_RESOLVERS: Dict[str, Resolver] = {}
_COMPUTED_RESOLVERS: Dict[str, Resolver] = {}


def register_scalar(
    schema_path: str,
    formatter: Callable[[Any], Optional[FieldValue]] = fmt_str,
) -> str:
    """Register (or confirm) a resolver for `schema_path`. Returns the key
    to place in an alias table.

    Idempotent: re-registering with the same formatter is a no-op.
    Registering with a different formatter logs a warning and keeps the
    first registration — two forms trying to render the same schema path
    with conflicting formatting is a programmer error, not a runtime
    situation to silently resolve.
    """
    new_fmt = getattr(formatter, "__name__", repr(formatter))
    existing = _SCHEMA_RESOLVERS.get(schema_path)
    if existing is None:
        resolver = path(schema_path, formatter)
        resolver._formatter_name = new_fmt   # type: ignore[attr-defined]
        _SCHEMA_RESOLVERS[schema_path] = resolver
        return schema_path

    existing_fmt = getattr(existing, "_formatter_name", None)
    if existing_fmt is not None and existing_fmt != new_fmt:
        _logger.warning(
            "register_scalar: %r already registered with %s, "
            "ignoring conflicting formatter %s",
            schema_path, existing_fmt, new_fmt,
        )
    # Non-conflict re-registration: keep the original resolver + formatter
    # (first-write-wins so the warning path doesn't quietly mask drift).
    return schema_path


def register_computed(name: str, resolver: Resolver) -> str:
    """Register a computed resolver under '@name'. Returns the key."""
    if not name or name.startswith("."):
        raise ValueError(f"invalid computed name: {name!r}")
    key = f"@{name}"
    _COMPUTED_RESOLVERS[key] = resolver
    return key


def _lookup_resolver(key: str) -> Resolver:
    """Resolve either a schema path or a @-prefixed computed key.

    Raises KeyError on unknown — an alias table referencing a missing key
    is a bug that should surface loudly, not fall through to a silent no-op.
    """
    if key.startswith("@"):
        return _COMPUTED_RESOLVERS[key]
    return _SCHEMA_RESOLVERS[key]


def array_aliases(
    stem: str,
    list_path: str,
    leaf_path: str,
    *,
    max_count: int,
    formatter: Callable[[Any], Optional[FieldValue]] = fmt_str,
) -> Dict[str, str]:
    """Register N array-slot resolvers + return {stem_A: schema_key, ...}.

    Example::

        array_aliases("Driver_GivenName", "lob_details.drivers",
                      "first_name", max_count=8)

    Registers resolvers for lob_details.drivers[0..7].first_name and returns
    {"Driver_GivenName_A": "lob_details.drivers[0].first_name", ..., "_H": …}.
    """
    if max_count < 1 or max_count > 26:
        raise ValueError(f"max_count must be 1..26, got {max_count}")

    out: Dict[str, str] = {}
    for i in range(max_count):
        letter = chr(ord("A") + i)
        schema_key = f"{list_path}[{i}].{leaf_path}"
        register_scalar(schema_key, formatter)
        out[f"{stem}_{letter}"] = schema_key
    return out


# ---------------------------------------------------------------------------
# Computed resolvers — LOB discriminator + mutex checkboxes
# ---------------------------------------------------------------------------

def _lob_is(target_lob: str) -> Resolver:
    def _get(sub: CustomerSubmission) -> Optional[bool]:
        details = sub.lob_details
        if details is None:
            return None
        return True if details.lob == target_lob else None
    _get.__name__ = f"lob_is({target_lob})"
    return _get


_LOB_CA = register_computed("lob.commercial_auto",   _lob_is("commercial_auto"))
_LOB_GL = register_computed("lob.general_liability", _lob_is("general_liability"))
_LOB_WC = register_computed("lob.workers_comp",      _lob_is("workers_comp"))


def _gl_occurrence_basis(sub: CustomerSubmission) -> Optional[bool]:
    """GL occurrence indicator = claims-made is explicitly False (not None)."""
    cm = _resolve(sub, "lob_details.coverage.claims_made_basis")
    return True if cm is False else None


_GL_OCC = register_computed("gl.occurrence", _gl_occurrence_basis)


# ---------------------------------------------------------------------------
# Violation resolvers (P10.S.7)
# ---------------------------------------------------------------------------
#
# ACORD 127's AccidentConviction widget family has only an _A suffix — the
# form records a single violation record per submission, not a nested array.
# Semantically this is "the most recent / most relevant violation reported
# with this application."
#
# We pick the first violation from the first driver who has one. Arbitrary
# but deterministic, and matches v3's drivers[0].violations[0] convention
# when the LLM respects slot ordering.

def _first_violation_leaf(
    leaf_path: str,
    formatter: Callable[[Any], Optional[FieldValue]] = fmt_str,
) -> Resolver:
    """Return a resolver that picks the first non-empty violation across
    drivers[] and walks leaf_path. Returns None when no driver has a
    violation recorded or the leaf itself is None."""
    def _get(sub: CustomerSubmission) -> Optional[FieldValue]:
        details = sub.lob_details
        if details is None or not hasattr(details, "drivers"):
            return None
        for driver in details.drivers or []:
            violations = getattr(driver, "violations", None) or []
            if violations:
                return formatter(_resolve(violations[0], leaf_path))
        return None
    _get.__name__ = f"first_violation({leaf_path})"
    return _get


def _first_violation_place_combined(
    sub: CustomerSubmission,
) -> Optional[str]:
    """PlaceOfIncident on ACORD 127 is a single free-text field. Combine
    city + state from the first violation into 'city, state' so brokers
    don't have to re-type either."""
    details = sub.lob_details
    if details is None or not hasattr(details, "drivers"):
        return None
    for driver in details.drivers or []:
        violations = getattr(driver, "violations", None) or []
        if violations:
            v = violations[0]
            city  = (getattr(v, "location_city", None)  or "").strip()
            state = (getattr(v, "location_state", None) or "").strip()
            combined = ", ".join(p for p in (city, state) if p)
            return combined or None
    return None


_VIOLATION_DATE        = register_computed(
    "violation.first.occurred_on",
    _first_violation_leaf("occurred_on", fmt_date),
)
_VIOLATION_DESCRIPTION = register_computed(
    "violation.first.description",
    _first_violation_leaf("description"),
)
_VIOLATION_PLACE       = register_computed(
    "violation.first.place_combined",
    _first_violation_place_combined,
)


# ---------------------------------------------------------------------------
# Contact-by-role resolver (P10.S.4)
# ---------------------------------------------------------------------------
#
# ACORD 130 carries role-specific contact widgets (accounting / claim /
# inspection). The schema models contacts as a free-form List[Contact] with
# an Optional[str] role — the LLM tags each contact as it extracts them.
# At fill time, the mapper finds the first contact whose role matches and
# walks the requested leaf.
#
# Matching is deterministic: case-insensitive equality against a canonical
# role name, plus a small synonym set so "claims" / "loss" map to "claim".
# No fuzzy matching, no Levenshtein — debuggability > flexibility here.

_ROLE_SYNONYMS: Dict[str, tuple] = {
    "accounting": ("accounting", "accounts", "finance", "ar", "ap"),
    "claim":      ("claim", "claims", "loss", "losses"),
    "inspection": ("inspection", "inspector", "underwriting", "uw"),
    "primary":    ("primary", "main", "principal"),
}


def _matches_role(contact_role: Optional[str], canonical: str) -> bool:
    """Match a contact's free-form role against a canonical role name.

    Falls back to exact-match for canonicals without a synonym-table entry
    (e.g. "ceo" matches "ceo" case-insensitively) — otherwise roles outside
    the synonym table could never be selected by name.
    """
    if not contact_role:
        return False
    normalized = contact_role.strip().lower()
    return normalized in _ROLE_SYNONYMS.get(canonical, (canonical,))


def contact_by_role(
    role: str,
    leaf_path: str,
    formatter: Callable[[Any], Optional[FieldValue]] = fmt_str,
) -> Resolver:
    """Build a resolver that returns the first contact matching `role`'s
    leaf value (formatted). Returns None when no matching contact exists
    or the leaf itself is None."""
    def _get(sub: CustomerSubmission) -> Optional[FieldValue]:
        contacts = sub.contacts or []
        for c in contacts:
            if _matches_role(c.role, role):
                return formatter(_resolve(c, leaf_path))
        return None
    _get.__name__ = f"contact[role={role}].{leaf_path}"
    return _get


# Role-specific computed resolvers — keyed under @contact.{role}.{leaf}.
_CONTACT_ACCOUNTING_NAME  = register_computed("contact.accounting.full_name", contact_by_role("accounting", "full_name"))
_CONTACT_ACCOUNTING_PHONE = register_computed("contact.accounting.phone",     contact_by_role("accounting", "phone", fmt_phone))
_CONTACT_ACCOUNTING_EMAIL = register_computed("contact.accounting.email",     contact_by_role("accounting", "email"))

_CONTACT_CLAIM_NAME       = register_computed("contact.claim.full_name",      contact_by_role("claim", "full_name"))
_CONTACT_CLAIM_PHONE      = register_computed("contact.claim.phone",          contact_by_role("claim", "phone", fmt_phone))
_CONTACT_CLAIM_EMAIL      = register_computed("contact.claim.email",          contact_by_role("claim", "email"))

_CONTACT_INSPECTION_NAME  = register_computed("contact.inspection.full_name", contact_by_role("inspection", "full_name"))
_CONTACT_INSPECTION_PHONE = register_computed("contact.inspection.phone",     contact_by_role("inspection", "phone", fmt_phone))
_CONTACT_INSPECTION_EMAIL = register_computed("contact.inspection.email",     contact_by_role("inspection", "email"))


# ---------------------------------------------------------------------------
# Per-form alias tables
#
# Values are either:
#   * a schema path (e.g. "business_name") — looked up in _SCHEMA_RESOLVERS
#   * a @-prefixed key (e.g. "@lob.commercial_auto") — looked up in
#     _COMPUTED_RESOLVERS
#
# register_scalar() is called as a side effect of building the table so
# referenced resolvers exist by the time map_submission_to_form runs.
# ---------------------------------------------------------------------------

_FORM_125: Dict[str, str] = {
    "NamedInsured_FullName_A":                            register_scalar("business_name"),
    "NamedInsured_TaxIdentifier_A":                       register_scalar("ein"),
    "NamedInsured_MailingAddress_LineOne_A":              register_scalar("mailing_address.line_one"),
    "NamedInsured_MailingAddress_LineTwo_A":              register_scalar("mailing_address.line_two"),
    "NamedInsured_MailingAddress_CityName_A":             register_scalar("mailing_address.city"),
    "NamedInsured_MailingAddress_StateOrProvinceCode_A":  register_scalar("mailing_address.state"),
    "NamedInsured_MailingAddress_PostalCode_A":           register_scalar("mailing_address.zip_code"),
    "NamedInsured_Contact_PrimaryPhoneNumber_A":          register_scalar("phone", fmt_phone),
    "NamedInsured_Primary_PhoneNumber_A":                 "phone",   # 10.S.4: second-widget alias
    "NamedInsured_Contact_PrimaryEmailAddress_A":         register_scalar("email"),
    # NB: 125 does NOT carry NamedInsured_Primary_EmailAddress_A — 130 only.
    "Policy_Status_EffectiveDate_A":                      register_scalar("policy_dates.effective_date",  fmt_date),
    "Policy_EffectiveDate_A":                             "policy_dates.effective_date",  # 10.S.3: 125 carries both widget names
    "Policy_ExpirationDate_A":                            register_scalar("policy_dates.expiration_date", fmt_date),
    "Policy_LineOfBusiness_BusinessAutoIndicator_A":      _LOB_CA,
    "Policy_LineOfBusiness_CommercialGeneralLiability_A": _LOB_GL,

    # --- Producer (10.S.5) — full block (125 carries contact + mailing address) ---
    "Producer_FullName_A":                            register_scalar("producer.agency_name"),
    "Producer_CustomerIdentifier_A":                  register_scalar("producer.producer_code"),
    "Producer_StateLicenseIdentifier_A":              register_scalar("producer.license_number"),
    "Producer_ContactPerson_FullName_A":              register_scalar("producer.contact_name"),
    "Producer_ContactPerson_PhoneNumber_A":           register_scalar("producer.phone", fmt_phone),
    "Producer_ContactPerson_EmailAddress_A":          register_scalar("producer.email"),
    "Producer_MailingAddress_LineOne_A":              register_scalar("producer.mailing_address.line_one"),
    "Producer_MailingAddress_LineTwo_A":              register_scalar("producer.mailing_address.line_two"),
    "Producer_MailingAddress_CityName_A":             register_scalar("producer.mailing_address.city"),
    "Producer_MailingAddress_StateOrProvinceCode_A":  register_scalar("producer.mailing_address.state"),
    "Producer_MailingAddress_PostalCode_A":           register_scalar("producer.mailing_address.zip_code"),

    # --- Producer signer + NPN (10.S.6) ---
    "Producer_NationalIdentifier_A":                  register_scalar("producer.national_producer_number"),
    "Producer_AuthorizedRepresentative_FullName_A":   register_scalar("producer.authorized_representative"),

    # --- Locations (10.S.8) — 4 slots A-D per commercial structure. Paths
    # start with `locations[i]` (top-level on CustomerSubmission, not under
    # lob_details), so these fill regardless of LOB.
    **array_aliases("CommercialStructure_PhysicalAddress_LineOne",
                    "locations", "address.line_one",  max_count=4),
    **array_aliases("CommercialStructure_PhysicalAddress_LineTwo",
                    "locations", "address.line_two",  max_count=4),
    **array_aliases("CommercialStructure_PhysicalAddress_CityName",
                    "locations", "address.city",      max_count=4),
    **array_aliases("CommercialStructure_PhysicalAddress_StateOrProvinceCode",
                    "locations", "address.state",     max_count=4),
    **array_aliases("CommercialStructure_PhysicalAddress_PostalCode",
                    "locations", "address.zip_code",  max_count=4),
    **array_aliases("CommercialStructure_PhysicalAddress_CountyName",
                    "locations", "address.county",    max_count=4),
    **array_aliases("CommercialStructure_AnnualRevenueAmount",
                    "locations", "annual_gross_receipts",
                    max_count=4, formatter=fmt_money),
    **array_aliases("BuildingOccupancy_OperationsDescription",
                    "locations", "description",       max_count=4),
}

_FORM_126: Dict[str, str] = {
    "NamedInsured_FullName_A":                                             "business_name",
    "Policy_EffectiveDate_A":                                              register_scalar("policy_dates.effective_date", fmt_date),
    "GeneralLiability_ClaimsMadeIndicator_A":                              register_scalar("lob_details.coverage.claims_made_basis", fmt_checkbox),
    "GeneralLiability_OccurrenceIndicator_A":                              _GL_OCC,
    "GeneralLiability_GeneralAggregate_LimitAmount_A":                     register_scalar("lob_details.coverage.general_aggregate_limit",             fmt_money),
    "GeneralLiability_ProductsAndCompletedOperations_AggregateLimitAmount_A":  register_scalar("lob_details.coverage.products_ops_aggregate_limit",     fmt_money),
    "GeneralLiability_EachOccurrence_LimitAmount_A":                       register_scalar("lob_details.coverage.each_occurrence_limit",               fmt_money),
    "GeneralLiability_PersonalAndAdvertisingInjury_LimitAmount_A":         register_scalar("lob_details.coverage.personal_advertising_injury_limit",   fmt_money),
    "GeneralLiability_FireDamageRentedPremises_EachOccurrenceLimitAmount_A":   register_scalar("lob_details.coverage.damage_to_rented_premises_limit", fmt_money),
    "GeneralLiability_MedicalExpense_EachPersonLimitAmount_A":             register_scalar("lob_details.coverage.medical_expense_limit",               fmt_money),

    # --- Producer (10.S.5) ---
    "Producer_FullName_A":               "producer.agency_name",
    "Producer_CustomerIdentifier_A":     "producer.producer_code",
    "Producer_StateLicenseIdentifier_A": "producer.license_number",
    # --- Producer signer + NPN (10.S.6) ---
    "Producer_NationalIdentifier_A":                "producer.national_producer_number",
    "Producer_AuthorizedRepresentative_FullName_A": "producer.authorized_representative",
}

_FORM_127: Dict[str, str] = {
    "NamedInsured_FullName_A":   "business_name",
    "Policy_EffectiveDate_A":    "policy_dates.effective_date",
    # Expanded 8 → 13 slots (A-M) to match ACORD 127's physical row capacity (P10.S.7).
    **array_aliases("Driver_GivenName",                   "lob_details.drivers", "first_name",       max_count=13),
    **array_aliases("Driver_Surname",                     "lob_details.drivers", "last_name",        max_count=13),
    **array_aliases("Driver_OtherGivenNameInitial",       "lob_details.drivers", "middle_initial",   max_count=13),
    **array_aliases("Driver_BirthDate",                   "lob_details.drivers", "date_of_birth",    max_count=13, formatter=fmt_date),
    **array_aliases("Driver_LicenseNumberIdentifier",     "lob_details.drivers", "license_number",   max_count=13),
    **array_aliases("Driver_LicensedStateOrProvinceCode", "lob_details.drivers", "license_state",    max_count=13),
    **array_aliases("Driver_ExperienceYearCount",         "lob_details.drivers", "years_experience", max_count=13, formatter=fmt_int),

    # --- Producer (10.S.5) ---
    "Producer_FullName_A":               "producer.agency_name",
    "Producer_CustomerIdentifier_A":     "producer.producer_code",
    "Producer_StateLicenseIdentifier_A": "producer.license_number",
    # --- Producer signer + NPN (10.S.6) ---
    "Producer_NationalIdentifier_A":                "producer.national_producer_number",
    "Producer_AuthorizedRepresentative_FullName_A": "producer.authorized_representative",

    # --- Violation (10.S.7) — single slot on the form, computed from the
    # first driver with a recorded violation. Intentionally NOT mapped:
    #   AccidentConviction_ViolationYearCount_A       — computable from occurred_on
    #   AccidentConviction_DriverProducerIdentifier_A — no clean schema fit
    "AccidentConviction_TrafficViolationDate_A":        _VIOLATION_DATE,
    "AccidentConviction_TrafficViolationDescription_A": _VIOLATION_DESCRIPTION,
    "AccidentConviction_PlaceOfIncident_A":             _VIOLATION_PLACE,

    # --- Per-vehicle coverage (P10.S.10a) — 4 slots A-D on 127 ---
    **array_aliases("Vehicle_Coverage_LiabilityIndicator",
                    "lob_details.vehicles", "coverage.liability",
                    max_count=4, formatter=fmt_checkbox),
    **array_aliases("Vehicle_Coverage_CollisionIndicator",
                    "lob_details.vehicles", "coverage.collision",
                    max_count=4, formatter=fmt_checkbox),
    **array_aliases("Vehicle_Coverage_ComprehensiveIndicator",
                    "lob_details.vehicles", "coverage.comprehensive",
                    max_count=4, formatter=fmt_checkbox),
    **array_aliases("Vehicle_Collision_DeductibleAmount",
                    "lob_details.vehicles", "coverage.collision_deductible_amount",
                    max_count=4, formatter=fmt_money),
    **array_aliases("Vehicle_Coverage_ComprehensiveOrSpecifiedCauseOfLossDeductibleAmount",
                    "lob_details.vehicles", "coverage.comprehensive_deductible_amount",
                    max_count=4, formatter=fmt_money),
    **array_aliases("Vehicle_Coverage_MedicalPaymentsIndicator",
                    "lob_details.vehicles", "coverage.medical_payments",
                    max_count=4, formatter=fmt_checkbox),
    **array_aliases("Vehicle_Coverage_UninsuredMotoristsIndicator",
                    "lob_details.vehicles", "coverage.uninsured_motorists",
                    max_count=4, formatter=fmt_checkbox),
    **array_aliases("Vehicle_Coverage_UnderinsuredMotoristsIndicator",
                    "lob_details.vehicles", "coverage.underinsured_motorists",
                    max_count=4, formatter=fmt_checkbox),
    **array_aliases("Vehicle_Coverage_TowingAndLabourIndicator",
                    "lob_details.vehicles", "coverage.towing_labour",
                    max_count=4, formatter=fmt_checkbox),
    **array_aliases("Vehicle_Coverage_RentalReimbursementIndicator",
                    "lob_details.vehicles", "coverage.rental_reimbursement",
                    max_count=4, formatter=fmt_checkbox),
}

_FORM_129: Dict[str, str] = {
    "NamedInsured_FullName_A": "business_name",
    "Policy_EffectiveDate_A":  register_scalar("policy_dates.effective_date", fmt_date),
    **array_aliases("Vehicle_ModelYear",         "lob_details.vehicles", "year",       max_count=5, formatter=fmt_int),
    **array_aliases("Vehicle_ManufacturersName", "lob_details.vehicles", "make",       max_count=5),
    **array_aliases("Vehicle_ModelName",         "lob_details.vehicles", "model",      max_count=5),
    **array_aliases("Vehicle_VINIdentifier",     "lob_details.vehicles", "vin",        max_count=5),
    **array_aliases("Vehicle_BodyCode",          "lob_details.vehicles", "body_type",  max_count=5),

    # --- Producer (10.S.5) — 129 lacks Producer_StateLicenseIdentifier_A ---
    "Producer_FullName_A":           "producer.agency_name",
    "Producer_CustomerIdentifier_A": "producer.producer_code",

    # --- Per-vehicle coverage (P10.S.10a) — 5 slots A-E on 129.
    # Schema paths are identical to 127's; register_scalar is idempotent,
    # so indices 0-3 reuse the resolvers registered by 127's 4-slot call,
    # and index 4 registers fresh.
    **array_aliases("Vehicle_Coverage_LiabilityIndicator",
                    "lob_details.vehicles", "coverage.liability",
                    max_count=5, formatter=fmt_checkbox),
    **array_aliases("Vehicle_Coverage_CollisionIndicator",
                    "lob_details.vehicles", "coverage.collision",
                    max_count=5, formatter=fmt_checkbox),
    **array_aliases("Vehicle_Coverage_ComprehensiveIndicator",
                    "lob_details.vehicles", "coverage.comprehensive",
                    max_count=5, formatter=fmt_checkbox),
    **array_aliases("Vehicle_Collision_DeductibleAmount",
                    "lob_details.vehicles", "coverage.collision_deductible_amount",
                    max_count=5, formatter=fmt_money),
    **array_aliases("Vehicle_Coverage_ComprehensiveOrSpecifiedCauseOfLossDeductibleAmount",
                    "lob_details.vehicles", "coverage.comprehensive_deductible_amount",
                    max_count=5, formatter=fmt_money),
    **array_aliases("Vehicle_Coverage_MedicalPaymentsIndicator",
                    "lob_details.vehicles", "coverage.medical_payments",
                    max_count=5, formatter=fmt_checkbox),
    **array_aliases("Vehicle_Coverage_UninsuredMotoristsIndicator",
                    "lob_details.vehicles", "coverage.uninsured_motorists",
                    max_count=5, formatter=fmt_checkbox),
    **array_aliases("Vehicle_Coverage_UnderinsuredMotoristsIndicator",
                    "lob_details.vehicles", "coverage.underinsured_motorists",
                    max_count=5, formatter=fmt_checkbox),
    **array_aliases("Vehicle_Coverage_TowingAndLabourIndicator",
                    "lob_details.vehicles", "coverage.towing_labour",
                    max_count=5, formatter=fmt_checkbox),
    **array_aliases("Vehicle_Coverage_RentalReimbursementIndicator",
                    "lob_details.vehicles", "coverage.rental_reimbursement",
                    max_count=5, formatter=fmt_checkbox),
}

_FORM_130: Dict[str, str] = {
    "NamedInsured_FullName_A":      "business_name",
    "NamedInsured_TaxIdentifier_A": "ein",
    "Policy_EffectiveDate_A":       register_scalar("policy_dates.effective_date",  fmt_date),
    "Policy_ExpirationDate_A":      register_scalar("policy_dates.expiration_date", fmt_date),
    "WorkersCompensationEmployersLiability_EmployersLiability_EachAccidentLimitAmount_A":
        register_scalar("lob_details.coverage.employers_liability_per_accident", fmt_money),
    "WorkersCompensationEmployersLiability_EmployersLiability_DiseaseEachEmployeeLimitAmount_A":
        register_scalar("lob_details.coverage.employers_liability_per_employee", fmt_money),
    "WorkersCompensationEmployersLiability_EmployersLiability_DiseasePolicyLimitAmount_A":
        register_scalar("lob_details.coverage.employers_liability_per_policy",   fmt_money),
    **array_aliases("WorkersCompensation_RateClass_ClassificationCode",    "lob_details.payroll_by_class", "class_code",     max_count=14),
    **array_aliases("WorkersCompensation_RateClass_DutiesDescription",     "lob_details.payroll_by_class", "description",    max_count=14),
    **array_aliases("WorkersCompensation_RateClass_RemunerationAmount",    "lob_details.payroll_by_class", "payroll",        max_count=14, formatter=fmt_money),
    **array_aliases("WorkersCompensation_RateClass_FullTimeEmployeeCount", "lob_details.payroll_by_class", "employee_count", max_count=14, formatter=fmt_int),

    # --- Primary business contact (10.S.4) ---
    "NamedInsured_Primary_PhoneNumber_A":  "phone",
    "NamedInsured_Primary_EmailAddress_A": "email",

    # --- Role-specific contacts (10.S.4) — resolved from contacts[] by role match ---
    "NamedInsured_AccountingContact_FullName_A":     _CONTACT_ACCOUNTING_NAME,
    "NamedInsured_AccountingContact_PhoneNumber_A":  _CONTACT_ACCOUNTING_PHONE,
    "NamedInsured_AccountingContact_EmailAddress_A": _CONTACT_ACCOUNTING_EMAIL,

    "NamedInsured_ClaimContact_FullName_A":          _CONTACT_CLAIM_NAME,
    "NamedInsured_ClaimContact_PhoneNumber_A":       _CONTACT_CLAIM_PHONE,
    "NamedInsured_ClaimContact_EmailAddress_A":      _CONTACT_CLAIM_EMAIL,

    "NamedInsured_InspectionContact_FullName_A":     _CONTACT_INSPECTION_NAME,
    "NamedInsured_InspectionContact_PhoneNumber_A":  _CONTACT_INSPECTION_PHONE,
    "NamedInsured_InspectionContact_EmailAddress_A": _CONTACT_INSPECTION_EMAIL,

    # --- Producer (10.S.5) — full block; 130 lacks StateLicense widget ---
    "Producer_FullName_A":                            "producer.agency_name",
    "Producer_CustomerIdentifier_A":                  "producer.producer_code",
    "Producer_ContactPerson_FullName_A":              "producer.contact_name",
    "Producer_ContactPerson_PhoneNumber_A":           "producer.phone",
    "Producer_ContactPerson_EmailAddress_A":          "producer.email",
    "Producer_MailingAddress_LineOne_A":              "producer.mailing_address.line_one",
    "Producer_MailingAddress_LineTwo_A":              "producer.mailing_address.line_two",
    "Producer_MailingAddress_CityName_A":             "producer.mailing_address.city",
    "Producer_MailingAddress_StateOrProvinceCode_A":  "producer.mailing_address.state",
    "Producer_MailingAddress_PostalCode_A":           "producer.mailing_address.zip_code",
    # --- Producer signer + NPN (10.S.6) — 130 has NPN but not the signer widget ---
    "Producer_NationalIdentifier_A":                  "producer.national_producer_number",

    # --- Locations (10.S.9) — 3 slots A-C, Location_ convention, all 6 leaves.
    # Resolvers already registered by 125's array_aliases — side-effect
    # adds new slot indices (A-C = index 0-2, same as 125's A-C).
    **array_aliases("Location_PhysicalAddress_LineOne",
                    "locations", "address.line_one",  max_count=3),
    **array_aliases("Location_PhysicalAddress_LineTwo",
                    "locations", "address.line_two",  max_count=3),
    **array_aliases("Location_PhysicalAddress_CityName",
                    "locations", "address.city",      max_count=3),
    **array_aliases("Location_PhysicalAddress_StateOrProvinceCode",
                    "locations", "address.state",     max_count=3),
    **array_aliases("Location_PhysicalAddress_PostalCode",
                    "locations", "address.zip_code",  max_count=3),
    **array_aliases("Location_PhysicalAddress_CountyName",
                    "locations", "address.county",    max_count=3),
}


# --- Fan-out shells (P10.S.3) — business_name + effective_date only ---

_FORM_131: Dict[str, str] = {
    "NamedInsured_FullName_A":   "business_name",
    "Policy_EffectiveDate_A":    "policy_dates.effective_date",

    # --- Producer (10.S.5) ---
    "Producer_FullName_A":               "producer.agency_name",
    "Producer_CustomerIdentifier_A":     "producer.producer_code",
    "Producer_StateLicenseIdentifier_A": "producer.license_number",
    # --- Producer signer + NPN (10.S.6) ---
    "Producer_NationalIdentifier_A":                "producer.national_producer_number",
    "Producer_AuthorizedRepresentative_FullName_A": "producer.authorized_representative",

    # --- Locations (10.S.9) — 6 slots A-F, CommercialStructure_ convention.
    # ACORD 131 has no LineTwo or CountyName widgets — single-line address.
    **array_aliases("CommercialStructure_PhysicalAddress_LineOne",
                    "locations", "address.line_one",  max_count=6),
    **array_aliases("CommercialStructure_PhysicalAddress_CityName",
                    "locations", "address.city",      max_count=6),
    **array_aliases("CommercialStructure_PhysicalAddress_StateOrProvinceCode",
                    "locations", "address.state",     max_count=6),
    **array_aliases("CommercialStructure_PhysicalAddress_PostalCode",
                    "locations", "address.zip_code",  max_count=6),
}

_FORM_137: Dict[str, str] = {
    "NamedInsured_FullName_A":   "business_name",
    "Policy_EffectiveDate_A":    "policy_dates.effective_date",

    # --- Producer (10.S.5) — 137 lacks license widget ---
    "Producer_FullName_A":           "producer.agency_name",
    "Producer_CustomerIdentifier_A": "producer.producer_code",
    # --- Producer signer + NPN (10.S.6) — 137 has NPN only ---
    "Producer_NationalIdentifier_A": "producer.national_producer_number",
}

_FORM_159: Dict[str, str] = {
    "NamedInsured_FullName_A":                           "business_name",
    "NamedInsured_MailingAddress_LineOne_A":             "mailing_address.line_one",
    "NamedInsured_MailingAddress_CityName_A":            "mailing_address.city",
    "NamedInsured_MailingAddress_StateOrProvinceCode_A": "mailing_address.state",
    "NamedInsured_MailingAddress_PostalCode_A":          "mailing_address.zip_code",
    "Policy_EffectiveDate_A":                            "policy_dates.effective_date",

    # --- Producer (10.S.5) — full contact + mailing block ---
    "Producer_FullName_A":                            "producer.agency_name",
    "Producer_CustomerIdentifier_A":                  "producer.producer_code",
    "Producer_ContactPerson_FullName_A":              "producer.contact_name",
    "Producer_ContactPerson_PhoneNumber_A":           "producer.phone",
    "Producer_ContactPerson_EmailAddress_A":          "producer.email",
    "Producer_MailingAddress_LineOne_A":              "producer.mailing_address.line_one",
    "Producer_MailingAddress_LineTwo_A":              "producer.mailing_address.line_two",
    "Producer_MailingAddress_CityName_A":             "producer.mailing_address.city",
    "Producer_MailingAddress_StateOrProvinceCode_A":  "producer.mailing_address.state",
    "Producer_MailingAddress_PostalCode_A":           "producer.mailing_address.zip_code",

    # --- Locations (10.S.9) — 14 slots A-N, CommercialStructure_ convention.
    # ACORD 159 property schedule uses single-line addresses, no LineTwo/County.
    **array_aliases("CommercialStructure_PhysicalAddress_LineOne",
                    "locations", "address.line_one",  max_count=14),
    **array_aliases("CommercialStructure_PhysicalAddress_CityName",
                    "locations", "address.city",      max_count=14),
    **array_aliases("CommercialStructure_PhysicalAddress_StateOrProvinceCode",
                    "locations", "address.state",     max_count=14),
    **array_aliases("CommercialStructure_PhysicalAddress_PostalCode",
                    "locations", "address.zip_code",  max_count=14),
}

_FORM_160: Dict[str, str] = {
    "NamedInsured_FullName_A":   "business_name",
    "Policy_EffectiveDate_A":    "policy_dates.effective_date",

    # --- Producer (10.S.5) ---
    "Producer_FullName_A":               "producer.agency_name",
    "Producer_CustomerIdentifier_A":     "producer.producer_code",
    "Producer_StateLicenseIdentifier_A": "producer.license_number",
    # --- Producer signer + NPN (10.S.6) ---
    "Producer_NationalIdentifier_A":                "producer.national_producer_number",
    "Producer_AuthorizedRepresentative_FullName_A": "producer.authorized_representative",
}


_FORM_ALIASES: Dict[str, Dict[str, str]] = {
    "125": _FORM_125,
    "126": _FORM_126,
    "127": _FORM_127,
    "129": _FORM_129,
    "130": _FORM_130,
    "131": _FORM_131,
    "137": _FORM_137,
    "159": _FORM_159,
    "160": _FORM_160,
    "163": {},   # coord-layout generic Text100 names — 10.S.10
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def map_submission_to_form(
    submission: CustomerSubmission,
    form_number: str,
) -> Dict[str, FieldValue]:
    if form_number not in _FORM_ALIASES:
        load_form_spec(form_number)   # raises UnknownFormError if truly unknown
        return {}

    out: Dict[str, FieldValue] = {}
    aliases = _FORM_ALIASES[form_number]
    for acord_field, schema_key in aliases.items():
        try:
            resolver = _lookup_resolver(schema_key)
        except KeyError:
            _logger.error(
                "mapper: unresolved key form=%s field=%s key=%s",
                form_number, acord_field, schema_key,
            )
            continue
        try:
            value = resolver(submission)
        except Exception as exc:
            _logger.exception(
                "mapper: resolver failed form=%s field=%s key=%s err=%s",
                form_number, acord_field, schema_key, exc,
            )
            continue
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        out[acord_field] = value
    return out


def map_submission(
    submission: CustomerSubmission,
) -> Dict[str, Dict[str, FieldValue]]:
    if submission.lob_details is None:
        return {}
    lob = submission.lob_details.lob
    return {
        form_number: map_submission_to_form(submission, form_number)
        for form_number in forms_for_lob(lob)
    }
