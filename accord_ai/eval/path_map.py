"""v3 scenario path → v4 schema path translator.

v3 expectation YAMLs use paths like `business.business_name`,
`vehicles[0].year`, `drivers[0].full_name`. v4's schema has restructured
hierarchies (business_* is flat at root; vehicles/drivers live under
`lob_details`) and different field names (`dob` → `date_of_birth`,
`radius` → `radius_of_travel`).

This module translates. Most mappings are 1→1. A few (full_name) split
into 2 pairs. Untranslatable paths return []; the caller tracks them
separately rather than erroring, so a v4 schema gap surfaces as "field
could not be scored" rather than a scorer crash.

The mapping set is extensible — adding a new v3 path is a new row in
_RULES. If you find a scenario path that's not covered, the scorer logs
it with the exact scenario id so you know what to add.

First-vehicle convention: v3's `auto_info.*` keys (e.g. `auto_info.use_type`)
are fleet-level in the v3 model. v4 stores `use_type`/`farthest_zone` per
`Vehicle`, so we translate to `lob_details.vehicles[0].*`. This silently
hides divergence across a multi-vehicle fleet with different `use_type`
values. No production v3 scenario exercises that case today; if one lands,
fan the rule out to emit N pairs (one per vehicle) and teach the scorer
to accept per-index match sets.
"""
from __future__ import annotations

import re
from typing import Any, Callable, List, Tuple, Union


# ---------------------------------------------------------------------------
# Rule types
# ---------------------------------------------------------------------------
#
# A rule is either:
#   * str — regex replacement pattern (v4_path_template with \1, \2 etc.)
#   * Callable[[str, Any], List[Tuple[str, Any]]] — custom split/transform
#
# Rules are tried in order; first match wins.

_Rule = Union[str, Callable[[str, Any], List[Tuple[str, Any]]]]


def _split_full_name(v3_path: str, value: Any) -> List[Tuple[str, Any]]:
    """drivers[N].full_name → (first_name, last_name) pair on v4."""
    m = re.match(r"drivers\[(\d+)\]\.full_name", v3_path)
    if not m or not isinstance(value, str):
        return []
    idx = m.group(1)
    parts = value.strip().split()
    if not parts:
        return []
    first = parts[0]
    last  = parts[-1] if len(parts) > 1 else ""
    pairs: List[Tuple[str, Any]] = [
        (f"lob_details.drivers[{idx}].first_name", first),
    ]
    if last:
        pairs.append((f"lob_details.drivers[{idx}].last_name", last))
    return pairs


# Top-level list-count expectations in v3 scenarios (bare key, no dots/brackets).
# v3 writes `vehicles: 1` meaning "one vehicle present"; v4 resolves that from the
# list length. Scorer interprets the `@count:` prefix to mean len(resolved).
#
# LOB-conditional routing note: `prior_insurance` only resolves when the
# submission's lob_details is a WorkersCompensationDetails (v4 only tracks
# prior insurance on WC). On CA/GL/etc. submissions, the path resolves to
# None → count 0 → scores as missing, which is correct given the schema gap.
# Adding a CA prior_insurance field in the future would let this route start
# matching without any change here.
_COUNT_ROUTES = {
    "vehicles":        "@count:lob_details.vehicles",
    "drivers":         "@count:lob_details.drivers",
    "loss_history":    "@count:loss_history",
    "prior_insurance": "@count:lob_details.prior_insurance",
}


def _top_level_list_count(v3_path: str, value: Any) -> List[Tuple[str, Any]]:
    route = _COUNT_ROUTES.get(v3_path)
    if route is None:
        return []
    return [(route, value)]


# ---------------------------------------------------------------------------
# Mapping rules — (v3_regex, v4_template_or_callable)
# ---------------------------------------------------------------------------

_RULES: List[Tuple[re.Pattern, _Rule]] = [
    # --- Splits (handled as callables; must come BEFORE the patterns that
    # would otherwise catch the same prefix) ---
    (re.compile(r"^drivers\[\d+\]\.full_name$"),
     _split_full_name),

    # --- Business → root (v4 flattens) ---
    (re.compile(r"^business\.business_name$"),        r"business_name"),
    (re.compile(r"^business\.dba$"),                  r"dba"),
    (re.compile(r"^business\.tax_id$"),               r"ein"),
    (re.compile(r"^business\.phone$"),                r"phone"),
    (re.compile(r"^business\.email$"),                r"email"),
    (re.compile(r"^business\.website$"),              r"website"),
    (re.compile(r"^business\.entity_type$"),          r"entity_type"),
    (re.compile(r"^business\.naics$"),                r"naics_code"),
    (re.compile(r"^business\.sic$"),                  r"sic_code"),
    (re.compile(r"^business\.years_in_business$"),    r"years_in_business"),
    (re.compile(r"^business\.business_start_date$"),  r"business_start_date"),
    (re.compile(r"^business\.operations_description$"), r"operations_description"),
    (re.compile(r"^business\.nature_of_business$"),   r"nature_of_business"),
    (re.compile(r"^business\.employee_count$"),       r"full_time_employees"),
    (re.compile(r"^business\.full_time_employees$"),  r"full_time_employees"),
    (re.compile(r"^business\.part_time_employees$"),  r"part_time_employees"),
    (re.compile(r"^business\.annual_revenue$"),       r"annual_revenue"),
    (re.compile(r"^business\.annual_payroll$"),       r"annual_payroll"),
    (re.compile(r"^business\.contact_name$"),         r"contacts[0].full_name"),
    (re.compile(r"^business\.mailing_address\.(.+)$"), r"mailing_address.\1"),

    # --- Producer ---
    (re.compile(r"^producer\.agency_name$"),          r"producer.agency_name"),
    (re.compile(r"^producer\.contact_name$"),         r"producer.contact_name"),
    (re.compile(r"^producer\.phone$"),                r"producer.phone"),
    (re.compile(r"^producer\.email$"),                r"producer.email"),
    (re.compile(r"^producer\.license_number$"),       r"producer.license_number"),
    (re.compile(r"^producer\.producer_code$"),        r"producer.producer_code"),
    (re.compile(r"^producer\.mailing_address\.(.+)$"), r"producer.mailing_address.\1"),

    # --- Policy ---
    (re.compile(r"^policy\.effective_date$"),         r"policy_dates.effective_date"),
    (re.compile(r"^policy\.expiration_date$"),        r"policy_dates.expiration_date"),
    (re.compile(r"^policy\.status$"),                 r"policy_status"),
    (re.compile(r"^policy\.billing_plan$"),           r"billing_plan"),
    (re.compile(r"^policy\.payment_plan$"),           r"payment_plan"),

    # --- Vehicles (shift under lob_details, rename a few leaves) ---
    (re.compile(r"^vehicles\[(\d+)\]\.dob$"),
     r"lob_details.vehicles[\1].date_of_birth"),
    (re.compile(r"^vehicles\[(\d+)\]\.radius$"),
     r"lob_details.vehicles[\1].radius_of_travel"),
    (re.compile(r"^vehicles\[(\d+)\]\.garaging_address\.zip$"),
     r"lob_details.vehicles[\1].garage_address.zip_code"),
    (re.compile(r"^vehicles\[(\d+)\]\.garaging_address\.(.+)$"),
     r"lob_details.vehicles[\1].garage_address.\2"),
    (re.compile(r"^vehicles\[(\d+)\]\.(.+)$"),
     r"lob_details.vehicles[\1].\2"),

    # --- Drivers (shift under lob_details, rename dob) ---
    (re.compile(r"^drivers\[(\d+)\]\.dob$"),
     r"lob_details.drivers[\1].date_of_birth"),
    (re.compile(r"^drivers\[(\d+)\]\.mailing_address\.(.+)$"),
     r"lob_details.drivers[\1].mailing_address.\2"),
    (re.compile(r"^drivers\[(\d+)\]\.(.+)$"),
     r"lob_details.drivers[\1].\2"),

    # --- Top-level list-count expectations (bare keys — no dots/brackets) ---
    # v3 writes `vehicles: 1` / `drivers: 1` / `prior_insurance: "_LIST"` etc.
    # These must come before any broader rule that would swallow them (none
    # currently would, but keep them early for clarity).
    (re.compile(r"^(?:vehicles|drivers|loss_history|prior_insurance)$"),
     _top_level_list_count),

    # --- auto_info.* — v3's catch-all namespace covering both vehicle-level
    # and coverage-level facts. We translate only what v4 actually models;
    # everything else is deliberately untranslatable (so it shows up in
    # untranslatable_paths rather than silently scoring as "missing").
    #
    # Rules here are explicit — the previous catch-all routed all auto_info.*
    # to lob_details.coverage.*, which misreported `use_type` (on Vehicle),
    # `hazmat` (on CommercialAutoDetails), and created phantom coverage
    # fields for `dash_cam`, `telematics`, `driver_training`, etc. that
    # don't exist in v4. ---
    #
    # Vehicle-level (first-vehicle convention — v3 collapses across fleet).
    (re.compile(r"^auto_info\.use_type$"),
     r"lob_details.vehicles[0].use_type"),
    (re.compile(r"^auto_info\.farthest_zone$"),
     r"lob_details.vehicles[0].farthest_zone"),
    # CommercialAutoDetails-level.
    (re.compile(r"^auto_info\.hazmat$"),
     r"lob_details.hazmat"),
    # Coverage-level (CommercialAutoCoverage).
    (re.compile(r"^auto_info\.(liability_limit_csl|bi_per_person|bi_per_accident|pd_per_accident|uim_limit|medpay_limit|comp_deductible|coll_deductible|hired_auto|non_owned_auto)$"),
     r"lob_details.coverage.\1"),
    # Anything else under auto_info.* (dash_cam, telematics, driver_training,
    # trailer_interchange, states_of_operation, cargo_type, hired_vehicle_count,
    # etc.) intentionally falls through to untranslatable.

    # --- coverage.* still routes wholesale (tests depend on it + these are
    # all real CommercialAutoCoverage fields by convention). ---
    (re.compile(r"^coverage\.(.+)$"),
     r"lob_details.coverage.\1"),

    # --- Loss history (v4 stores at root) ---
    (re.compile(r"^loss_history\[(\d+)\]\.date$"),
     r"loss_history[\1].date_of_loss"),
    (re.compile(r"^loss_history\[(\d+)\]\.(.+)$"),
     r"loss_history[\1].\2"),

    # --- Additional interests ---
    (re.compile(r"^additional_interests\[(\d+)\]\.(.+)$"),
     r"additional_interests[\1].\2"),
]


def translate(v3_path: str, v3_value: Any) -> List[Tuple[str, Any]]:
    """Return list of (v4_path, v4_value) pairs for one v3 expectation.

    Returns []:
      * if no rule matches (caller treats as "untranslatable", logs)
      * if a split-callable returns [] because the value shape isn't
        usable (e.g. driver full_name isn't a string)

    The returned value is NOT yet normalized for scoring — the scorer
    handles type coercion and string trimming itself.
    """
    for pattern, rule in _RULES:
        m = pattern.match(v3_path)
        if not m:
            continue
        if callable(rule):
            return rule(v3_path, v3_value)
        # String rule — regex substitution on path, identity on value.
        v4_path = pattern.sub(rule, v3_path)
        return [(v4_path, v3_value)]
    return []
