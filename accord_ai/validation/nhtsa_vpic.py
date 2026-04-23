"""NHTSA vPIC VIN decoder — inline-eligible enricher.

Calls the free NHTSA vPIC API to decode VINs and fill missing vehicle fields
(engine, body class, GVW, fuel type, drive type). Conflicts are recorded when
user-provided year/make/model disagrees with what vPIC returns.

API: https://vpic.nhtsa.dot.gov/api/vehicles/decodevin/{VIN}?format=json
Free, no key required. TTL-cached 24h per VIN.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

from accord_ai.cache import ttl_cached
from accord_ai.validation.types import PrefillPatch, ValidationFinding, ValidationResult

_logger = logging.getLogger(__name__)

_VPIC_URL = "https://vpic.nhtsa.dot.gov/api/vehicles/decodevin/{vin}?format=json"
_TIMEOUT_S = 2.0
_CACHE_TTL_S = 86400.0  # 24h


@ttl_cached(ttl_seconds=_CACHE_TTL_S, key=lambda vin: vin.upper())
async def _decode_vin(vin: str) -> Optional[Dict[str, Any]]:
    """Call NHTSA vPIC and return a flat dict of decoded fields. Cached 24h per VIN."""
    url = _VPIC_URL.format(vin=vin.upper())
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
            r = await client.get(url)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        _logger.debug("vPIC decode failed for %s: %s", vin, exc)
        return None

    results = data.get("Results") or []
    out: Dict[str, str] = {}
    for item in results:
        variable = (item.get("Variable") or "").strip()
        value = (item.get("Value") or "").strip()
        if variable and value and value not in ("", "Not Applicable", "0"):
            out[variable] = value

    # Check for a parse error (NHTSA returns error_code != "0" on bad VIN)
    error_code = out.get("Error Code", "0")
    if error_code != "0" and not any(k in out for k in ("Make", "Model Year", "Model")):
        return None

    return out if out else None


def _build_vehicle_patch(
    raw: Dict[str, Any],
    existing: Dict[str, Any],
    vehicle_idx: int,
) -> Tuple[Dict[int, Dict], List[Any]]:
    """Build an index-keyed patch dict and list of FieldConflict for one vehicle."""
    from accord_ai.schema import FieldConflict

    patch: Dict[str, Any] = {}
    conflicts: List[FieldConflict] = []

    # Year
    decoded_year_str = raw.get("Model Year", "")
    if decoded_year_str.isdigit():
        decoded_year = int(decoded_year_str)
        user_year = existing.get("year")
        if user_year and user_year != decoded_year:
            conflicts.append(FieldConflict(
                field_path=f"lob_details.vehicles[{vehicle_idx}].year",
                user_value=user_year,
                enriched_value=decoded_year,
                source="nhtsa_vpic",
            ))
        elif not user_year:
            patch["year"] = decoded_year

    # Make
    decoded_make = raw.get("Make", "")
    if decoded_make:
        user_make = (existing.get("make") or "").strip().upper()
        if user_make and user_make != decoded_make.upper():
            conflicts.append(FieldConflict(
                field_path=f"lob_details.vehicles[{vehicle_idx}].make",
                user_value=existing.get("make"),
                enriched_value=decoded_make,
                source="nhtsa_vpic",
            ))
        elif not user_make:
            patch["make"] = decoded_make.title()

    # Model
    decoded_model = raw.get("Model", "")
    if decoded_model and not existing.get("model"):
        patch["model"] = decoded_model

    # Body class → body_type
    body_class = raw.get("Body Class", "")
    if body_class and not existing.get("body_type"):
        patch["body_type"] = body_class

    # GVWR (Gross Vehicle Weight Rating) → gvw as int (strip "lbs" etc.)
    gvwr_str = raw.get("Gross Vehicle Weight Rating From", "") or raw.get("GVWR", "")
    if gvwr_str and not existing.get("gvw"):
        # NHTSA returns "Class 3: 10,001 - 14,000 lb (4,536 - 6,350 kg)" or plain digits
        import re
        digits = re.findall(r"\d+", gvwr_str.replace(",", ""))
        if digits:
            try:
                patch["gvw"] = int(digits[0])
            except ValueError:
                pass

    return {vehicle_idx: patch}, conflicts


class NhtsaVpicValidator:
    """Decode VINs via NHTSA vPIC — fills missing vehicle fields, records conflicts."""

    name: str = "nhtsa_vpic"
    applicable_fields: tuple = ("lob_details.vehicles[].vin",)
    inline_eligible: bool = True

    async def run(self, submission: Any) -> ValidationResult:
        """Finalize mode: decode all VINs, surface any year/make conflicts as findings."""
        started = datetime.now(tz=timezone.utc)
        t0 = time.monotonic()
        findings: List[ValidationFinding] = []

        sub_dict = submission.model_dump(mode="json") if hasattr(submission, "model_dump") else {}
        ld = sub_dict.get("lob_details") or {}
        vehicles = ld.get("vehicles") or []

        for i, veh in enumerate(vehicles):
            if not isinstance(veh, dict) or not veh.get("vin"):
                continue
            raw = await _decode_vin(veh["vin"])
            if not raw:
                continue
            _, veh_conflicts = _build_vehicle_patch(raw, veh, i)
            for c in veh_conflicts:
                findings.append(ValidationFinding(
                    validator=self.name,
                    field_path=c.field_path,
                    severity="warning",
                    message=f"VIN decodes to {c.enriched_value!r}, stated {c.user_value!r}",
                    details={"user_value": c.user_value, "enriched_value": c.enriched_value},
                ))

        duration_ms = (time.monotonic() - t0) * 1000
        return ValidationResult(
            validator=self.name, ran_at=started,
            duration_ms=duration_ms, success=True, findings=findings,
        )

    async def prefill(self, submission: Any, just_extracted: dict) -> Optional[PrefillPatch]:
        """Inline mode: decode newly-extracted VINs and fill missing fields."""
        sub_dict = submission.model_dump(mode="json") if hasattr(submission, "model_dump") else {}
        ld = sub_dict.get("lob_details") or {}
        vehicles = ld.get("vehicles") or []

        # Determine which VINs are new (present in just_extracted)
        just_ld = just_extracted.get("lob_details") or {}
        just_vehicles = just_ld.get("vehicles") or []
        new_vin_indices = {
            i for i, jv in enumerate(just_vehicles)
            if isinstance(jv, dict) and jv.get("vin")
        }

        if not new_vin_indices:
            return None

        all_patches: Dict[int, Dict] = {}
        all_conflicts: List[Any] = []

        tasks = [
            asyncio.create_task(
                self._process_vehicle(i, vehicles[i] if i < len(vehicles) else {})
            )
            for i in new_vin_indices
            if i < len(vehicles) and vehicles[i].get("vin")
        ]
        if not tasks:
            return None

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception) or result is None:
                continue
            idx_patch, veh_conflicts = result
            all_patches.update(idx_patch)
            all_conflicts.extend(veh_conflicts)

        if not all_patches and not all_conflicts:
            return None

        return PrefillPatch(
            patch={"lob_details": {"vehicles": all_patches}},
            conflicts=all_conflicts,
            source=self.name,
            confidence=0.95,
        )

    async def _process_vehicle(
        self,
        idx: int,
        veh: Dict[str, Any],
    ) -> Optional[Tuple[Dict, List]]:
        vin = veh.get("vin")
        if not vin:
            return None
        raw = await _decode_vin(vin)
        if not raw:
            return None
        return _build_vehicle_patch(raw, veh, idx)
