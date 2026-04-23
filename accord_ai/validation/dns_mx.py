"""DNS MX record validator — finalize-only.

Checks that every email address in the submission belongs to a domain that has
MX records, catching common typos (gmial.com, acme.co vs acme.com, etc.).

Uses dnspython for synchronous DNS resolution, wrapped in asyncio.to_thread()
to avoid blocking the event loop.

TTL-cached per domain for 1h — MX records rarely change and a broker's
corporate domain appears on every submission they file.
"""

from __future__ import annotations

import asyncio
import logging
import time
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from accord_ai.validation.types import (
    PrefillPatch,
    ValidationFinding,
    ValidationResult,
)

_logger = logging.getLogger(__name__)

_CACHE_TTL_S = 3600.0  # 1h
_DNS_LIFETIME_S = 3.0

_SENTINEL = object()  # distinct from None (which means NXDOMAIN cached)

# Thread-safe domain → (result, expires_at) cache
# result: None = NXDOMAIN, True = has MX, False = no MX records (NoAnswer)
_MX_CACHE: Dict[str, Tuple[Optional[bool], float]] = {}
_MX_LOCK = threading.Lock()


def _cache_get(domain: str) -> Optional[Optional[bool]]:
    with _MX_LOCK:
        entry = _MX_CACHE.get(domain)
        if entry is None:
            return _SENTINEL
        result, exp = entry
        if time.monotonic() > exp:
            del _MX_CACHE[domain]
            return _SENTINEL
        return result


def _cache_set(domain: str, has_mx: Optional[bool]) -> None:
    with _MX_LOCK:
        _MX_CACHE[domain] = (has_mx, time.monotonic() + _CACHE_TTL_S)


def _check_mx_sync(domain: str) -> Optional[bool]:
    """Synchronous DNS query. Returns True (has MX), False (no MX), None (NXDOMAIN)."""
    import dns.exception
    import dns.resolver

    try:
        dns.resolver.resolve(domain, "MX", lifetime=_DNS_LIFETIME_S)
        return True
    except dns.resolver.NXDOMAIN:
        return None
    except dns.resolver.NoAnswer:
        return False
    except (dns.exception.Timeout, dns.exception.DNSException):
        raise  # re-raise so caller can handle as transient


class DnsMxValidator:
    """Validate email domains have MX records — finalize-only."""

    name: str = "dns_mx"
    applicable_fields: tuple = ("contacts[].email",)
    inline_eligible: bool = False

    async def run(self, submission: Any) -> ValidationResult:
        started = datetime.now(tz=timezone.utc)
        t0 = time.monotonic()
        findings: List[ValidationFinding] = []

        sub_dict = (
            submission.model_dump(mode="json")
            if hasattr(submission, "model_dump") else {}
        )

        emails: List[Tuple[str, str]] = []  # (field_path, email)

        top_email = sub_dict.get("email")
        if top_email and "@" in top_email:
            emails.append(("email", top_email))

        for i, contact in enumerate((sub_dict.get("contacts") or [])):
            if not isinstance(contact, dict):
                continue
            em = contact.get("email")
            if em and "@" in em:
                emails.append((f"contacts[{i}].email", em))

        for field_path, email in emails:
            domain = email.split("@", 1)[1].strip().lower()
            has_mx = await self._lookup(domain, field_path, findings)

        duration_ms = (time.monotonic() - t0) * 1000
        return ValidationResult(
            validator=self.name, ran_at=started,
            duration_ms=duration_ms, success=True, findings=findings,
        )

    async def _lookup(
        self, domain: str, field_path: str, findings: List[ValidationFinding]
    ) -> None:
        cached = _cache_get(domain)
        if cached is not _SENTINEL:
            has_mx = cached
        else:
            try:
                has_mx = await asyncio.to_thread(_check_mx_sync, domain)
                _cache_set(domain, has_mx)
            except Exception as exc:
                import dns.exception
                if not isinstance(exc, dns.exception.DNSException):
                    _logger.debug("dns_mx unexpected error for %s: %s", domain, exc)
                else:
                    _logger.debug("dns_mx transient failure for %s: %s", domain, exc)
                return  # transient — don't penalize user

        if has_mx is None:
            findings.append(ValidationFinding(
                validator=self.name,
                field_path=field_path,
                severity="error",
                message=f"Email domain '{domain}' does not exist (NXDOMAIN)",
                details={"domain": domain},
            ))
        elif has_mx is False:
            findings.append(ValidationFinding(
                validator=self.name,
                field_path=field_path,
                severity="warning",
                message=f"Email domain '{domain}' has no MX records — may not receive email",
                details={"domain": domain},
            ))
        # has_mx is True → domain is fine, no finding

    async def prefill(self, submission: Any, just_extracted: dict) -> Optional[PrefillPatch]:
        return None
