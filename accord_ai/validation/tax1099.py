"""Tax1099 TIN matching validator — finalize-only.

POST https://api.tax1099.com/api/TinMatching with {tin, name}.
Match codes:
  0 → info  (name/TIN match)
  1 → warning (TIN not found)
  2 → warning (not issued)
  3 → error  (name mismatch)
  4 → error  (invalid TIN format)

Skips silently when TAX1099_API_KEY is not configured or when EIN is absent.
Caches responses per (ein, name) for 7 days.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

from accord_ai.validation.types import (
    PrefillPatch,
    ValidationFinding,
    ValidationResult,
)

_logger = logging.getLogger(__name__)

_API_URL = "https://api.tax1099.com/api/TinMatching"
_HTTP_TIMEOUT_S = 10.0
_CACHE_TTL_S = 7 * 24 * 3600.0  # 7 days

# Simple thread-safe TTL cache
_CACHE: Dict[Tuple[str, str], Tuple[Dict, float]] = {}
_CACHE_LOCK = threading.Lock()

_CODE_SEVERITY = {
    0: ("info", "Name and TIN combination matches IRS records"),
    1: ("warning", "TIN was not issued by IRS — verify the EIN"),
    2: ("warning", "TIN is listed as not issued — check with IRS"),
    3: ("error", "Name does not match TIN in IRS records"),
    4: ("error", "Invalid TIN format — EIN must be 9 digits"),
}


def _cache_get(key: Tuple[str, str]) -> Optional[Dict]:
    with _CACHE_LOCK:
        entry = _CACHE.get(key)
        if entry is None:
            return None
        result, exp = entry
        if time.monotonic() > exp:
            del _CACHE[key]
            return None
        return result


def _cache_set(key: Tuple[str, str], value: Dict) -> None:
    with _CACHE_LOCK:
        _CACHE[key] = (value, time.monotonic() + _CACHE_TTL_S)


async def _match_tin(api_key: str, ein: str, name: str) -> Optional[Dict]:
    """Call Tax1099 TIN matching API. Returns raw response dict or None on error."""
    cache_key = (ein.strip(), name.strip().lower())
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S) as client:
            r = await client.post(
                _API_URL,
                json={"tin": ein.strip(), "name": name.strip()},
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
        r.raise_for_status()
        result = r.json()
    except Exception as exc:
        _logger.debug("Tax1099 TIN match failed for EIN %s: %s", ein, exc)
        return None

    _cache_set(cache_key, result)
    return result


class Tax1099Validator:
    """Validate EIN/name TIN matching via Tax1099 API — finalize-only."""

    name: str = "tax1099"
    applicable_fields: tuple = ("ein", "business_name")
    inline_eligible: bool = False

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def run(self, submission: Any) -> ValidationResult:
        started = datetime.now(tz=timezone.utc)
        t0 = time.monotonic()
        findings: List[ValidationFinding] = []

        sub_dict = (
            submission.model_dump(mode="json")
            if hasattr(submission, "model_dump") else {}
        )

        ein = (sub_dict.get("ein") or "").strip()
        biz_name = (sub_dict.get("business_name") or "").strip()

        if not ein or not biz_name:
            duration_ms = (time.monotonic() - t0) * 1000
            return ValidationResult(
                validator=self.name, ran_at=started,
                duration_ms=duration_ms, success=True,
                findings=[],
            )

        result = await _match_tin(self._api_key, ein, biz_name)

        if result is not None:
            code = result.get("status_code", -1)
            severity, default_msg = _CODE_SEVERITY.get(
                code, ("warning", f"Unknown TIN match code: {code}")
            )
            msg = result.get("message") or default_msg
            findings.append(ValidationFinding(
                validator=self.name,
                field_path="ein",
                severity=severity,
                message=msg,
                details={"match_code": code, "ein": ein, "name": biz_name},
            ))

        duration_ms = (time.monotonic() - t0) * 1000
        return ValidationResult(
            validator=self.name, ran_at=started,
            duration_ms=duration_ms, success=True, findings=findings,
        )

    async def prefill(self, submission: Any, just_extracted: dict) -> Optional[PrefillPatch]:
        return None
