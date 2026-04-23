"""Zippopotam ZIP → city/state lookup — inline-eligible enricher.

Fills missing city and state when a ZIP code is provided. Records a conflict
when the user-supplied city/state doesn't match the ZIP's authoritative data.

API: https://api.zippopotam.us/us/{zip} (free, no key). TTL-cached 24h per ZIP.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from accord_ai.cache import ttl_cached
from accord_ai.validation.types import PrefillPatch, ValidationFinding, ValidationResult

_logger = logging.getLogger(__name__)

_ZIP_URL = "https://api.zippopotam.us/us/{zip}"
_TIMEOUT_S = 2.0
_CACHE_TTL_S = 86400.0


@ttl_cached(ttl_seconds=_CACHE_TTL_S, key=lambda zip_code: zip_code)
async def _lookup_zip(zip_code: str) -> Optional[Dict[str, Any]]:
    """Fetch ZIP data. Returns {"city": ..., "state": ..., "state_abbr": ...} or None."""
    url = _ZIP_URL.format(zip=zip_code)
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
            r = await client.get(url)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        _logger.debug("ZIP lookup failed for %s: %s", zip_code, exc)
        return None

    places = data.get("places") or []
    if not places:
        return None

    return {
        "city": places[0].get("place name", "").title(),
        "state": places[0].get("state abbreviation", "").upper(),
        "state_full": places[0].get("state", ""),
    }


def _patch_address(
    addr_key: str,
    addr_dict: Dict[str, Any],
    zip_data: Dict[str, Any],
) -> tuple[Dict, List]:
    """Build patch + conflicts for one address dict given ZIP lookup data."""
    from accord_ai.schema import FieldConflict

    patch: Dict[str, Any] = {}
    conflicts: List[FieldConflict] = []
    official_city = zip_data["city"]
    official_state = zip_data["state"]

    user_city = (addr_dict.get("city") or "").strip()
    user_state = (addr_dict.get("state") or "").strip().upper()

    if not user_city:
        patch["city"] = official_city
    elif user_city.upper() != official_city.upper():
        conflicts.append(FieldConflict(
            field_path=f"{addr_key}.city",
            user_value=user_city,
            enriched_value=official_city,
            source="zippopotam",
        ))

    if not user_state:
        patch["state"] = official_state
    elif user_state != official_state:
        conflicts.append(FieldConflict(
            field_path=f"{addr_key}.state",
            user_value=user_state,
            enriched_value=official_state,
            source="zippopotam",
        ))

    return patch, conflicts


class ZippopotamValidator:
    """Fill city/state from ZIP — inline-eligible, free, authoritative."""

    name: str = "zippopotam"
    applicable_fields: tuple = (
        "mailing_address.zip_code",
        "business_address.zip_code",
    )
    inline_eligible: bool = True

    async def run(self, submission: Any) -> ValidationResult:
        """Finalize mode: surface city/state mismatches as findings."""
        started = datetime.now(tz=timezone.utc)
        t0 = time.monotonic()
        findings: List[ValidationFinding] = []

        sub_dict = submission.model_dump(mode="json") if hasattr(submission, "model_dump") else {}
        for addr_key in ("mailing_address", "business_address"):
            addr = sub_dict.get(addr_key) or {}
            if not isinstance(addr, dict) or not addr.get("zip_code"):
                continue
            zip_data = await _lookup_zip(str(addr["zip_code"]).strip())
            if not zip_data:
                continue
            _, addr_conflicts = _patch_address(addr_key, addr, zip_data)
            for c in addr_conflicts:
                findings.append(ValidationFinding(
                    validator=self.name,
                    field_path=c.field_path,
                    severity="warning",
                    message=f"ZIP {addr['zip_code']} is in {zip_data['state']}, not {c.user_value!r}",
                    details={"official": zip_data["state"], "stated": c.user_value},
                ))

        duration_ms = (time.monotonic() - t0) * 1000
        return ValidationResult(
            validator=self.name, ran_at=started,
            duration_ms=duration_ms, success=True, findings=findings,
        )

    async def prefill(self, submission: Any, just_extracted: dict) -> Optional[PrefillPatch]:
        """Inline mode: fill city/state for newly-extracted ZIP codes."""
        patch: Dict[str, Any] = {}
        all_conflicts: List[Any] = []

        sub_dict = submission.model_dump(mode="json") if hasattr(submission, "model_dump") else {}

        for addr_key in ("mailing_address", "business_address"):
            just_addr = just_extracted.get(addr_key) or {}
            if not isinstance(just_addr, dict) or not just_addr.get("zip_code"):
                continue

            full_addr = sub_dict.get(addr_key) or {}
            zip_code = str(just_addr["zip_code"]).strip()
            zip_data = await _lookup_zip(zip_code)
            if not zip_data:
                continue

            addr_patch, addr_conflicts = _patch_address(addr_key, full_addr, zip_data)
            if addr_patch:
                existing = dict(full_addr)
                existing.update(addr_patch)
                patch[addr_key] = existing
            all_conflicts.extend(addr_conflicts)

        if not patch and not all_conflicts:
            return None

        return PrefillPatch(
            patch=patch,
            conflicts=all_conflicts,
            source=self.name,
            confidence=1.0,
        )
