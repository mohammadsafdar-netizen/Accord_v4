"""3-tier VIN-primary vehicle merge + license-primary driver merge.

Single authority for all entity-list dedup across:
  - Fleet upload (POST /upload-document)
  - Mid-conversation extraction (apply_diff → _merge_model)
  - Future OCR / registration-card paths

Vehicle tiers (applied in order):
  Tier 1: incoming VIN matches existing VIN — merge regardless of y/m/m
           (handles broker correction: "that 2022 is actually a 2023")
  Tier 2: full identity-tuple (vin_lower, year, make_lower, model_lower) matches
           an existing — merge; naturally handles both-no-VIN duplicate prevention
           since two no-VIN vehicles with the same y/m/m share the tuple
           ("", year, make, model).
  Tier 3: incoming has VIN + existing has no VIN + matching (year, make, model)
           — merge and populate the missing VIN on the existing record.

Driver: license_number-primary; falls back to first+last+date_of_birth.
"""
from __future__ import annotations

from accord_ai.schema import Driver, Vehicle


# ---------------------------------------------------------------------------
# Vehicle merge
# ---------------------------------------------------------------------------

def merge_vehicles(
    existing: list[Vehicle],
    incoming: list[Vehicle],
) -> list[Vehicle]:
    """Return a deduplicated, merged vehicle list.

    Existing vehicles are updated in-place (by position) when a match is
    found; new vehicles are appended. The VIN and identity indexes are
    refreshed after each merge so subsequent items see the current state.
    """
    result: list[Vehicle] = list(existing)
    vin_index: dict[str, int] = {}       # VIN.upper() → index in result
    ident_index: dict[tuple, int] = {}   # (vin_lower, yr, make, model) → index

    for i, v in enumerate(result):
        _index_vehicle(v, i, vin_index, ident_index)

    for inc in incoming:
        matched_idx = _find_vehicle_match(result, inc, vin_index, ident_index)

        if matched_idx is not None:
            result[matched_idx] = _merge_vehicle(result[matched_idx], inc)
            _index_vehicle(result[matched_idx], matched_idx, vin_index, ident_index)
        else:
            idx = len(result)
            result.append(inc)
            _index_vehicle(inc, idx, vin_index, ident_index)

    return result


def _index_vehicle(
    v: Vehicle,
    idx: int,
    vin_index: dict[str, int],
    ident_index: dict[tuple, int],
) -> None:
    vin = (v.vin or "").strip().upper()
    if vin:
        vin_index[vin] = idx
    ident = _v_ident(v)
    if any(ident):
        ident_index[ident] = idx


def _find_vehicle_match(
    result: list[Vehicle],
    inc: Vehicle,
    vin_index: dict[str, int],
    ident_index: dict[tuple, int],
) -> int | None:
    # Tier 1: VIN-primary match (both non-empty, same VIN)
    if inc.vin:
        vin_key = inc.vin.strip().upper()
        if vin_key in vin_index:
            return vin_index[vin_key]

    # Tier 2: full identity-tuple match (vin_lower + year + make + model)
    inc_ident = _v_ident(inc)
    if any(inc_ident) and inc_ident in ident_index:
        return ident_index[inc_ident]

    # Tier 3: incoming has VIN, existing has no VIN, matching (year, make, model)
    if inc.vin and _ymm_valid(inc):
        for i, ex in enumerate(result):
            if not ex.vin and _ymm_match(ex, inc):
                return i

    return None


def _v_ident(v: Vehicle) -> tuple:
    """Full identity tuple used for Tier 2 matching."""
    return (
        (v.vin or "").strip().lower(),
        str(v.year) if v.year else "",
        (v.make or "").strip().lower(),
        (v.model or "").strip().lower(),
    )


def _ymm_valid(v: Vehicle) -> bool:
    return bool(v.year and v.make and v.model)


def _ymm_match(a: Vehicle, b: Vehicle) -> bool:
    if not (_ymm_valid(a) and _ymm_valid(b)):
        return False
    return (
        a.year == b.year
        and (a.make or "").lower() == (b.make or "").lower()
        and (a.model or "").lower() == (b.model or "").lower()
    )


def _merge_vehicle(existing: Vehicle, incoming: Vehicle) -> Vehicle:
    """Return existing updated with non-None fields from incoming."""
    return existing.model_copy(update=incoming.model_dump(exclude_none=True))


# ---------------------------------------------------------------------------
# Driver merge
# ---------------------------------------------------------------------------

def merge_drivers(
    existing: list[Driver],
    incoming: list[Driver],
) -> list[Driver]:
    """Return a deduplicated, merged driver list.

    Primary key: license_number. Fallback: first+last+date_of_birth.
    """
    result: list[Driver] = list(existing)
    lic_index: dict[str, int] = {}  # license.upper() → index

    for i, d in enumerate(result):
        if d.license_number:
            lic_index[d.license_number.strip().upper()] = i

    for inc in incoming:
        matched_idx = _find_driver_match(result, inc, lic_index)

        if matched_idx is not None:
            result[matched_idx] = _merge_driver(result[matched_idx], inc)
            new_lic = (result[matched_idx].license_number or "").strip().upper()
            if new_lic:
                lic_index[new_lic] = matched_idx
        else:
            idx = len(result)
            result.append(inc)
            if inc.license_number:
                lic_index[inc.license_number.strip().upper()] = idx

    return result


def _find_driver_match(
    result: list[Driver],
    inc: Driver,
    lic_index: dict[str, int],
) -> int | None:
    if inc.license_number:
        lic_key = inc.license_number.strip().upper()
        if lic_key in lic_index:
            return lic_index[lic_key]

    first = (inc.first_name or "").strip().lower()
    last = (inc.last_name or "").strip().lower()
    dob = inc.date_of_birth
    if first and last and dob:
        for i, ex in enumerate(result):
            if (
                (ex.first_name or "").strip().lower() == first
                and (ex.last_name or "").strip().lower() == last
                and ex.date_of_birth == dob
            ):
                return i

    return None


def _merge_driver(existing: Driver, incoming: Driver) -> Driver:
    return existing.model_copy(update=incoming.model_dump(exclude_none=True))
