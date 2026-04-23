"""Extraction-delta post-processing pipeline (Phase A step 2).

Ported from accord_ai_v3/extraction/runner.py — five composable steps that
clean LLM-emitted JSON before pydantic validation. Each step is a pure
function on the dict (or in-place mutation where v3 did so), unit-testable
in isolation. The orchestrator ``run_postprocess`` runs them in order.

Steps:

1. ``unfold_dot_keys``         — ``{"a.b.c": 1}``         → ``{"a":{"b":{"c":1}}}``.
   Some LLM outputs collapse paths into dotted keys; pydantic doesn't
   unflatten on its own.

2. ``strip_empty``             — drops None / empty-string / empty-dict /
   empty-list values recursively, including filtering empty entries from
   lists. Saves pydantic from a thicket of present-but-meaningless fields.

3. ``drop_phantom_list_items`` — vehicles/drivers/etc. with no identity
   key (vin/make/license_number/...) are LLM correction artifacts. Drop
   them so they don't become independent ghost entities. If the session
   has exactly one existing matching item, the correction's fields are
   merged into that item's identity (the typical "actually 2022" pattern).

4. ``coerce_list_fields``      — ``"NE IA MO"`` → ``["NE","IA","MO"]``
   for state-list fields (``lob_details.states_of_operation``). The LLM
   sometimes emits delimited strings instead of arrays despite the schema.

5. ``cap_list_entries``        — defence-in-depth: caps vehicles/drivers/
   loss_history at 3 inline. The extractor prompt already says max 3, but
   bulk-dump turns occasionally exceed; downstream code shouldn't see >3
   inline entities (the upload flow handles overflow).

v4 schema notes (vs v3):
  * v3 had vehicles/drivers at root; v4 has ``lob_details.{vehicles,drivers}``.
  * v3's ``operations.states_of_operation``; v4's ``lob_details.states_of_operation``.
  * v3's ``loss_history`` and ``prior_insurance`` are at root; same in v4.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from accord_ai.logging_config import get_logger

_logger = get_logger("extraction.postprocess")


# ---------------------------------------------------------------------------
# Step 0 — promote v3-flat fields under lob_details
# ---------------------------------------------------------------------------

# v3 emitted vehicles/drivers/hazmat at the root of the submission; v4
# moved them under lob_details.* to match the discriminated-union shape.
# When the extractor uses SYSTEM_V2 (without the HARNESS_RULES
# path-translation table), the LLM frequently falls back to v3 flat
# shape. Since CustomerSubmission has `extra='ignore'`, these fields
# are silently dropped and the whole turn's vehicle data is lost.
# Discovered on negation-then-correction turn 2: the vehicle extraction
# JSON had `{"vehicles": [...]}` at root; the schema never saw it.
#
# Moves happen before any other cleanup so later steps see the
# canonical nested shape. List-shape merges (not replacements) preserve
# any lob_details.vehicles the LLM also emitted in the same turn.
_V3_FLAT_PROMOTIONS_LIST = (
    ("vehicles", "vehicles"),
    ("drivers",  "drivers"),
)
# Scalar/object promotions: v3 path → v4 key under lob_details.
_V3_FLAT_PROMOTIONS_SCALAR = (
    ("hazmat",               "hazmat"),
    ("trailer_interchange",  "trailer_interchange"),
    ("driver_training",      "driver_training"),
    ("fleet_use_type",       "fleet_use_type"),
    ("fleet_for_hire",       "fleet_for_hire"),
    ("states_of_operation",  "states_of_operation"),
    ("radius_of_operations", "radius_of_operations"),
    ("vehicle_count",        "vehicle_count"),
    ("driver_count",         "driver_count"),
)


def promote_v3_flat_fields(delta: Dict[str, Any]) -> None:
    """Move v3-style root-level fields under ``lob_details`` in place.

    Side-effects on ``delta``:
      * ``delta.pop("vehicles")`` → ``delta["lob_details"]["vehicles"]``
      * ``delta.pop("drivers")``  → ``delta["lob_details"]["drivers"]``
      * Same for known CA-scalar fields (hazmat, radius_of_operations, ...)

    If lob_details already has a matching list, concatenates. If it has
    a matching scalar, the nested value wins (we assume the LLM's nested
    value is more intentional than the flat fallback)."""
    lob = delta.get("lob_details")
    if not isinstance(lob, dict):
        lob = {}

    lob_created = False
    for flat_key, nested_key in _V3_FLAT_PROMOTIONS_LIST:
        if flat_key not in delta:
            continue
        flat_value = delta.pop(flat_key)
        if not isinstance(flat_value, list):
            continue
        existing = lob.get(nested_key)
        if isinstance(existing, list):
            lob[nested_key] = existing + flat_value
        else:
            lob[nested_key] = flat_value
        lob_created = True
        _logger.info(
            "promoted v3-flat %s→lob_details.%s (%d item(s))",
            flat_key, nested_key, len(flat_value),
        )

    for flat_key, nested_key in _V3_FLAT_PROMOTIONS_SCALAR:
        if flat_key not in delta:
            continue
        flat_value = delta.pop(flat_key)
        # Nested value already present → keep it (explicit > fallback).
        if nested_key in lob:
            continue
        lob[nested_key] = flat_value
        lob_created = True
        _logger.info(
            "promoted v3-flat %s→lob_details.%s",
            flat_key, nested_key,
        )

    if lob_created and "lob_details" not in delta:
        delta["lob_details"] = lob


# ---------------------------------------------------------------------------
# Step 1 — unfold dot-keys
# ---------------------------------------------------------------------------

def unfold_dot_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert flat dot-notation keys into nested dicts.

    ``{"mailing_address.city": "Dallas"}`` → ``{"mailing_address": {"city": "Dallas"}}``.

    No-op on dicts without any dotted keys.
    """
    if not any("." in k for k in data):
        return data

    result: Dict[str, Any] = {}
    for key, value in data.items():
        if "." in key:
            parts = key.split(".")
            cur: Dict[str, Any] = result
            for part in parts[:-1]:
                if part not in cur or not isinstance(cur[part], dict):
                    cur[part] = {}
                cur = cur[part]
            cur[parts[-1]] = value
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# Step 2 — strip empty values
# ---------------------------------------------------------------------------

def strip_empty(data: Any) -> Any:
    """Recursively remove None / empty-string / empty-dict / empty-list values.

    For lists: items that themselves recursively-strip to empty are dropped.
    Whitespace-only strings count as empty.
    Numeric zero and bool ``False`` are NOT empty (they're real values).

    Returns a NEW dict at every dict level (lists are filtered, then a new
    list is returned). Top-level non-dict input is returned unchanged.
    """
    if not isinstance(data, dict):
        return data
    cleaned: Dict[str, Any] = {}
    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, dict):
            nested = strip_empty(value)
            if nested:
                cleaned[key] = nested
        elif isinstance(value, list):
            filtered: List[Any] = []
            for item in value:
                if isinstance(item, dict):
                    nested_item = strip_empty(item)
                    if nested_item:
                        filtered.append(nested_item)
                elif item is None:
                    continue
                elif isinstance(item, str) and not item.strip():
                    continue
                else:
                    filtered.append(item)
            if filtered:
                cleaned[key] = filtered
        else:
            cleaned[key] = value
    return cleaned


# ---------------------------------------------------------------------------
# Step 3 — drop phantom list items
# ---------------------------------------------------------------------------

# Per-list identity keys. A list-item must have a non-empty value at one of
# these keys to count as a real entity. Items without identity are LLM
# correction artifacts (e.g. "actually the year is 2022" → vehicle with
# only year set) — they create ghost entities downstream.
#
# v3 reference (accord_ai_v3/extraction/runner.py:123): vehicles need
# vin/make/model; drivers need full_name/license_number; loss_history
# needs description/occurrence_date; prior_insurance needs carrier_name.
# `year` deliberately NOT in vehicle identity — "actually it's 2022"
# is the canonical phantom-merge trigger and must not self-promote
# to a real entity.
_IDENTITY_KEYS_BY_PATH: Dict[str, List[str]] = {
    # v4 nests vehicles/drivers under lob_details.
    "lob_details.vehicles":   ["vin", "make", "model"],
    "lob_details.drivers":    ["license_number", "first_name", "last_name"],
    # Top-level lists that v3 had + v4 keeps at root.
    "loss_history":           ["description", "date_of_loss", "type_of_loss"],
    "prior_insurance":        ["carrier_name"],
    "additional_interests":   ["name"],
    "locations":              ["address", "description"],
}


def _walk(container: Any, path: str) -> Optional[Any]:
    """Walk a dotted path through a nested dict; return None on miss."""
    cur = container
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _set_path(container: Dict[str, Any], path: str, value: Any) -> None:
    """Set a dotted path into a nested dict, creating intermediates."""
    parts = path.split(".")
    cur = container
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _has_identity(item: Dict[str, Any], identity_keys: List[str]) -> bool:
    """An item has identity if at least one identity key is non-empty."""
    if not isinstance(item, dict):
        return False
    return any(item.get(k) not in (None, "", []) for k in identity_keys)


def drop_phantom_list_items(
    delta: Dict[str, Any],
    current_state: Dict[str, Any],
) -> None:
    """Mutate ``delta`` in place: drop list items lacking identity.

    If the current state has exactly one existing item with identity, the
    phantom item's fields are merged onto a copy of that item (identity
    inherited) so the correction lands on the right entity instead of
    being lost. Behavior matches v3's pattern.
    """
    for path, identity_keys in _IDENTITY_KEYS_BY_PATH.items():
        items = _walk(delta, path)
        if not isinstance(items, list):
            continue

        current_items = _walk(current_state, path)
        surviving: List[Any] = []
        dropped = 0
        merged_into_existing = 0

        for item in items:
            if not isinstance(item, dict):
                continue

            if _has_identity(item, identity_keys):
                surviving.append(item)
                continue

            # Phantom — try to inherit from a single existing item.
            #
            # Step 3A: we used to inherit only the identity_keys from
            # current_items[0], which meant a correction emitting
            # {year: 2023} would phantom-merge to {year, make, model,
            # vin} but LOSE garage_address / use_type / etc. The list
            # merge then replaces the whole list and those fields are
            # gone. Inherit the FULL current item and overlay the
            # correction fields so non-identity context (garage_address,
            # radius_of_operations, use_type) survives a correction turn.
            if (
                isinstance(current_items, list)
                and len(current_items) == 1
                and isinstance(current_items[0], dict)
                and _has_identity(current_items[0], identity_keys)
            ):
                surviving.append({**current_items[0], **item})
                merged_into_existing += 1
                continue

            dropped += 1

        if dropped or merged_into_existing:
            _logger.debug(
                "phantom-handling on %s: kept=%d merged=%d dropped=%d",
                path, len(surviving), merged_into_existing, dropped,
            )

        _set_path(delta, path, surviving)


# ---------------------------------------------------------------------------
# Step 4 — coerce list fields
# ---------------------------------------------------------------------------

# 2-letter US state codes. Used for normalizing string-form state lists.
_STATE_CODES = frozenset({
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN",
    "IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV",
    "NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN",
    "TX","UT","VT","VA","WA","WV","WI","WY","DC",
})

# Common state-name → 2-letter mapping for natural-language inputs.
_STATE_NAMES_TO_CODE = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE",
    "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ",
    "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR",
    "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
    "district of columbia": "DC", "d.c.": "DC", "dc": "DC",
}


def _normalize_state_list(value: Any) -> Optional[List[str]]:
    """Normalize a state-list value into a list of 2-letter USPS codes.

    Returns None if the input was already a list (no coercion needed) OR
    if the input is not a string at all. Returns the coerced list when the
    input was a string the function actually transformed.

    Two-pass tokenization handles both punctuated input ("Texas, Oklahoma,
    New Mexico" — comma-separated multi-word names must NOT split on
    internal whitespace) and pure-whitespace input ("TX OK NM"):

      Pass 1: split on punctuation + the literal word "and". Each chunk
              is matched as a state name first (multi-word safe), then as
              a 2-letter code.
      Pass 2: chunks that don't match as a whole are split on whitespace
              and each token is matched individually.

    Output preserves insertion order; duplicate codes are deduped while
    keeping the first occurrence.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    s = value.strip()
    # Punctuation + "and" only — preserve internal whitespace inside chunks
    # so "New Mexico" stays a single chunk on this pass.
    chunks = re.split(r"[,;/&]|\band\b", s, flags=re.IGNORECASE)
    out: List[str] = []

    def _add(code: str) -> None:
        if code not in out:
            out.append(code)

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        # Try multi-word state name first
        code = _STATE_NAMES_TO_CODE.get(chunk.lower())
        if code:
            _add(code)
            continue
        # Try whole chunk as 2-letter code
        upper = chunk.upper()
        if len(upper) == 2 and upper in _STATE_CODES:
            _add(upper)
            continue
        # Fallback: split this chunk on whitespace and match each token
        for token in chunk.split():
            t_upper = token.upper()
            if len(t_upper) == 2 and t_upper in _STATE_CODES:
                _add(t_upper)
                continue
            sub_code = _STATE_NAMES_TO_CODE.get(token.lower())
            if sub_code:
                _add(sub_code)
            # Unknown tokens are silently dropped — non-USPS values would
            # fail downstream pydantic validation anyway.
    return out or None


# v4 paths where state-list coercion applies.
_STATE_LIST_PATHS = (
    "lob_details.states_of_operation",
)


def coerce_list_fields(delta: Dict[str, Any]) -> None:
    """Coerce known list-shaped fields from delimited strings to real lists.

    Mutates ``delta`` in place. No-op on values that are already lists.
    """
    for path in _STATE_LIST_PATHS:
        value = _walk(delta, path)
        if value is None:
            continue
        coerced = _normalize_state_list(value)
        if coerced is not None:
            _set_path(delta, path, coerced)


# ---------------------------------------------------------------------------
# Step 5 — cap list entries
# ---------------------------------------------------------------------------

# Lists that cap inline; the upload flow handles overflow.
_CAPPED_LIST_PATHS = (
    "lob_details.vehicles",
    "lob_details.drivers",
    "loss_history",
)
_MAX_INLINE = 3


def cap_list_entries(delta: Dict[str, Any]) -> None:
    """Cap each tracked list to ``_MAX_INLINE`` entries. Mutates in place."""
    for path in _CAPPED_LIST_PATHS:
        lst = _walk(delta, path)
        if not isinstance(lst, list) or len(lst) <= _MAX_INLINE:
            continue
        dropped = len(lst) - _MAX_INLINE
        _logger.info(
            "Capping %s at %d entries (dropped %d inline; "
            "use upload for overflow)",
            path, _MAX_INLINE, dropped,
        )
        _set_path(delta, path, lst[:_MAX_INLINE])


# ---------------------------------------------------------------------------
# Step 6a — fix mis-slotted garage_address fields
# ---------------------------------------------------------------------------

# Two-letter USPS codes used to detect state codes in the wrong slot.
_US_STATE_CODES = frozenset({
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN",
    "IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV",
    "NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN",
    "TX","UT","VT","VA","WA","WV","WI","WY","DC",
})


def _looks_like_street_address(value: Optional[str]) -> bool:
    """Heuristic: strings starting with a digit (e.g. "123 Main St") look
    like a street address; pure-word strings ("Tucson") look like a city."""
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if not stripped:
        return False
    return stripped[0].isdigit()


def _fix_vehicle_garage_address(vehicle: Dict[str, Any]) -> None:
    """Correct a mis-slotted garage_address in place.

    The LLM sometimes parses "garaged Tucson AZ 85701" as
    line_one="Tucson", city="AZ", zip_code="85701" (city in
    line_one, state in city). This is a common Phase R SYSTEM_V2
    extraction error — without the harness's address-parsing prose,
    the model fills the slots left-to-right.

    Detection: city is a 2-letter USPS code AND line_one is non-numeric
    (not a street address) AND state is missing/None. When all three
    hold we shift: line_one→city (if empty), city→state, line_one=None.
    """
    ga = vehicle.get("garage_address")
    if not isinstance(ga, dict):
        return
    line_one = ga.get("line_one")
    city = ga.get("city")
    state = ga.get("state")

    if (
        isinstance(city, str)
        and city.upper() in _US_STATE_CODES
        and not state
        and isinstance(line_one, str)
        and not _looks_like_street_address(line_one)
    ):
        new_city = line_one.strip()
        new_state = city.upper()
        ga["line_one"] = None
        ga["city"] = new_city
        ga["state"] = new_state
        _logger.info(
            "fixed mis-slotted garage_address: line_one→city=%s, city→state=%s",
            new_city, new_state,
        )


def fix_garage_address_misparse(delta: Dict[str, Any]) -> None:
    """Walk lob_details.vehicles and fix garage_address mis-parses."""
    vehicles = _walk(delta, "lob_details.vehicles")
    if not isinstance(vehicles, list):
        return
    for v in vehicles:
        if isinstance(v, dict):
            _fix_vehicle_garage_address(v)


# ---------------------------------------------------------------------------
# Step 6 — inject missing LOB discriminator
# ---------------------------------------------------------------------------

# `lob_details` is a discriminated union on the `lob` field. Partial
# correction outputs (e.g. {"lob_details": {"vehicles": [{"year": 2023}]}})
# drop the discriminator and fail pydantic validation with
# "Unable to extract tag using discriminator 'lob'". When the delta carries
# lob_details but no `lob`, and current_state already has a resolved LOB,
# inherit it. Discovered on correction-vehicle-year turn 2: the year
# correction succeeded at the LLM but the whole extraction was discarded
# on a validation failure the postprocess can fix deterministically.
#
# Known lob values must match the Literal["..."] defaults on each
# *Details model — kept in sync by a one-off manual audit. A new LOB
# added to the schema must be added here too.
_KNOWN_LOBS = frozenset({
    "commercial_auto", "general_liability", "workers_comp",
})


def inject_lob_discriminator(
    delta: Dict[str, Any], current_state: Dict[str, Any],
) -> None:
    """Fill ``lob_details.lob`` from current_state when missing.

    Mutates ``delta`` in place. No-op if the delta has no lob_details,
    if lob is already set, or if current_state has no resolved lob.
    """
    lob_details = delta.get("lob_details")
    if not isinstance(lob_details, dict):
        return
    if isinstance(lob_details.get("lob"), str) and lob_details["lob"] in _KNOWN_LOBS:
        return
    current_lob = (
        current_state.get("lob_details", {}).get("lob")
        if isinstance(current_state.get("lob_details"), dict)
        else None
    )
    if isinstance(current_lob, str) and current_lob in _KNOWN_LOBS:
        lob_details["lob"] = current_lob
        _logger.info(
            "injected lob discriminator from current_state: lob=%s",
            current_lob,
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_postprocess(
    delta: Dict[str, Any],
    current_state: Dict[str, Any],
) -> Dict[str, Any]:
    """Run the full post-processing pipeline on a raw extraction delta.

    Returns the processed dict (steps 1-2 return new dicts; steps 3-6
    mutate in place after the unfolded/stripped dict is established).
    """
    promote_v3_flat_fields(delta)
    delta = unfold_dot_keys(delta)
    delta = strip_empty(delta)
    drop_phantom_list_items(delta, current_state)
    coerce_list_fields(delta)
    cap_list_entries(delta)
    fix_garage_address_misparse(delta)
    inject_lob_discriminator(delta, current_state)
    return delta
