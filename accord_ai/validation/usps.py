"""USPS Address Validation v3 — finalize-only validator.

Uses the USPS Web Tools v3 OAuth2 API to standardize addresses and confirm
deliverability. Skipped silently when USPS_CONSUMER_KEY / USPS_CONSUMER_SECRET
are not configured.

OAuth2 token endpoint: POST https://api.usps.com/oauth2/v3/token
Address endpoint:      GET  https://api.usps.com/addresses/v3/address

TTL-cached OAuth token (1h). Retries once on 401.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from accord_ai.cache import ttl_cached
from accord_ai.validation.types import (
    PrefillPatch,
    ValidationFinding,
    ValidationResult,
)

_logger = logging.getLogger(__name__)

_TOKEN_URL = "https://api.usps.com/oauth2/v3/token"
_ADDRESS_URL = "https://api.usps.com/addresses/v3/address"
_HTTP_TIMEOUT_S = 5.0


# ---------------------------------------------------------------------------
# OAuth token (TTL-cached 1h)
# ---------------------------------------------------------------------------


def _make_token_fetcher(consumer_key: str, consumer_secret: str):
    @ttl_cached(ttl_seconds=3600.0, key=lambda: "usps_token")
    async def _fetch_token() -> Optional[str]:
        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S) as client:
                r = await client.post(
                    _TOKEN_URL,
                    data={
                        "grant_type": "client_credentials",
                        "client_id": consumer_key,
                        "client_secret": consumer_secret,
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
            r.raise_for_status()
            return r.json().get("access_token")
        except Exception as exc:
            _logger.debug("USPS token fetch failed: %s", exc)
            return None

    return _fetch_token


# ---------------------------------------------------------------------------
# Address lookup
# ---------------------------------------------------------------------------


_ADDR_OK = "OK"
_ADDR_EXPIRED = "EXPIRED_TOKEN"
_ADDR_FAILED = "FAILED"


async def _lookup_address(
    token: str,
    street: str,
    city: str,
    state: str,
    zip_code: str,
) -> tuple[str, Optional[Dict[str, Any]]]:
    """Return (status, address_dict). status is _ADDR_OK / _ADDR_EXPIRED / _ADDR_FAILED."""
    params = {
        "streetAddress": street,
        "city": city,
        "state": state,
        "ZIPCode": zip_code,
    }
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S) as client:
            r = await client.get(
                _ADDRESS_URL,
                params={k: v for k, v in params.items() if v},
                headers={"Authorization": f"Bearer {token}"},
            )
        if r.status_code == 200:
            return _ADDR_OK, r.json().get("address") or {}
        if r.status_code == 401:
            return _ADDR_EXPIRED, None
        return _ADDR_FAILED, None
    except Exception as exc:
        _logger.debug("USPS address lookup failed: %s", exc)
        return _ADDR_FAILED, None


def _addresses_from_submission(sub_dict: dict) -> List[tuple[str, dict]]:
    """Return [(field_path, address_dict)] for all addresses in submission."""
    results = []
    for field in ("mailing_address", "business_address"):
        addr = sub_dict.get(field)
        if addr and isinstance(addr, dict):
            results.append((field, addr))
    return results


def _addr_differs(entered: dict, standardized: dict) -> bool:
    """True if standardized meaningfully differs from entered."""
    def _norm(s: str) -> str:
        return (s or "").strip().upper()

    checks = [
        ("line_one", "streetAddress"),
        ("city", "city"),
        ("state", "state"),
        ("zip_code", "ZIPCode"),
    ]
    for entered_key, std_key in checks:
        ev = _norm(entered.get(entered_key) or "")
        sv = _norm(standardized.get(std_key) or "")
        if ev and sv and ev != sv:
            return True
    return False


class UspsValidator:
    """Validate addresses via USPS API v3 — finalize-only."""

    name: str = "usps"
    applicable_fields: tuple = ("mailing_address", "business_address")
    inline_eligible: bool = False

    def __init__(
        self,
        consumer_key: str,
        consumer_secret: str,
    ) -> None:
        self._consumer_key = consumer_key
        self._consumer_secret = consumer_secret
        self._fetch_token = _make_token_fetcher(consumer_key, consumer_secret)

    async def _get_token(self) -> Optional[str]:
        return await self._fetch_token()

    async def run(self, submission: Any) -> ValidationResult:
        started = datetime.now(tz=timezone.utc)
        t0 = time.monotonic()
        findings: List[ValidationFinding] = []

        sub_dict = (
            submission.model_dump(mode="json")
            if hasattr(submission, "model_dump") else {}
        )

        token = await self._get_token()
        if not token:
            duration_ms = (time.monotonic() - t0) * 1000
            return ValidationResult(
                validator=self.name, ran_at=started,
                duration_ms=duration_ms, success=False,
                error="Failed to obtain USPS OAuth token",
            )

        for field_path, addr in _addresses_from_submission(sub_dict):
            street = addr.get("line_one") or ""
            city = addr.get("city") or ""
            state = addr.get("state") or ""
            zip_code = addr.get("zip_code") or ""

            if not street:
                continue

            status, std = await _lookup_address(token, street, city, state, zip_code)
            if status == _ADDR_EXPIRED:
                # Real 401 → token expired mid-run; clear cache and retry once.
                if hasattr(self._fetch_token, "_ttl_cache"):
                    self._fetch_token._ttl_cache.clear()
                token = await self._get_token()
                if token:
                    status, std = await _lookup_address(token, street, city, state, zip_code)

            if status != _ADDR_OK or std is None:
                continue

            entered_display = f"{street}, {city}, {state} {zip_code}".strip(", ")
            std_display = ", ".join(filter(None, [
                std.get("streetAddress"),
                std.get("city"),
                std.get("state"),
                (std.get("ZIPCode") or "") + (
                    f"-{std.get('ZIPPlus4')}" if std.get("ZIPPlus4") else ""
                ),
            ]))

            if _addr_differs(addr, std):
                findings.append(ValidationFinding(
                    validator=self.name,
                    field_path=field_path,
                    severity="warning",
                    message=(
                        f"USPS standardized {field_path!r} differs from entered: "
                        f"{entered_display!r} → {std_display!r}"
                    ),
                    details={
                        "entered": entered_display,
                        "standardized": std_display,
                        "zip_plus_4": std.get("ZIPPlus4"),
                        "deliverable": True,
                    },
                ))
            else:
                findings.append(ValidationFinding(
                    validator=self.name,
                    field_path=field_path,
                    severity="info",
                    message=f"USPS confirmed {field_path!r} deliverable: {std_display!r}",
                    details={
                        "entered": entered_display,
                        "standardized": std_display,
                        "zip_plus_4": std.get("ZIPPlus4"),
                        "deliverable": True,
                    },
                ))

        duration_ms = (time.monotonic() - t0) * 1000
        return ValidationResult(
            validator=self.name, ran_at=started,
            duration_ms=duration_ms, success=True, findings=findings,
        )

    async def prefill(self, submission: Any, just_extracted: dict) -> Optional[PrefillPatch]:
        return None
