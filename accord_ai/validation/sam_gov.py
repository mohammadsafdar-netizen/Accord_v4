"""SAM.gov entity registration validator — finalize-only.

Checks if the business has an active SAM.gov registration (federal contractor).
Useful for commercial underwriting — federal contractors often carry specific
insurance requirements. Free with a registered API key.

Endpoint: GET https://api.sam.gov/entity-information/v3/entities
          ?api_key={SAM_GOV_API_KEY}&taxIdentificationNumber={ein}
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List, Optional

import httpx

from accord_ai.cache import ttl_cached
from accord_ai.validation.types import (
    PrefillPatch,
    ValidationFinding,
    ValidationResult,
)

_logger = logging.getLogger(__name__)

_SAM_URL = "https://api.sam.gov/entity-information/v3/entities"
_HTTP_TIMEOUT_S = 10.0
_CACHE_TTL_S = 7 * 24 * 3600.0  # 7 days — SAM data changes slowly


@dataclass
class _SamEntity:
    uei: str
    cage_code: str
    registration_status: str
    exp_date: Optional[str]
    business_types: list


def _make_fetcher(api_key: str):
    @ttl_cached(ttl_seconds=_CACHE_TTL_S, key=lambda ein: ein.strip().replace("-", ""))
    async def _lookup_ein(ein: str) -> Optional[_SamEntity]:
        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S) as client:
                r = await client.get(
                    _SAM_URL,
                    params={
                        "api_key": api_key,
                        "taxIdentificationNumber": ein.strip().replace("-", ""),
                    },
                )
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            _logger.debug("SAM.gov lookup failed for EIN %s: %s", ein, exc)
            return None

        entities = (data.get("entityData") or [])
        if not entities:
            return None

        entity = entities[0]
        reg = entity.get("entityRegistration") or {}
        assertions = entity.get("assertions") or {}
        biz_types_raw = (assertions.get("businessTypes") or {}).get("businessTypeList") or []
        biz_types = [bt.get("businessTypeCode") or bt.get("businessType") or "" for bt in biz_types_raw]

        return _SamEntity(
            uei=reg.get("ueiSAM") or reg.get("uei") or "",
            cage_code=reg.get("cageCode") or "",
            registration_status=reg.get("registrationStatus") or "Unknown",
            exp_date=reg.get("registrationExpirationDate") or reg.get("expDate"),
            business_types=[bt for bt in biz_types if bt],
        )

    return _lookup_ein


class SamGovValidator:
    """Check SAM.gov registration status via EIN — finalize-only."""

    name: str = "sam_gov"
    applicable_fields: tuple = ("ein",)
    inline_eligible: bool = False

    def __init__(self, api_key: str) -> None:
        self._lookup = _make_fetcher(api_key)

    async def run(self, submission: Any) -> ValidationResult:
        started = datetime.now(tz=timezone.utc)
        t0 = time.monotonic()
        findings: List[ValidationFinding] = []

        ein = getattr(submission, "ein", None) or (
            submission.get("ein") if hasattr(submission, "get") else None
        )
        if not ein:
            duration_ms = (time.monotonic() - t0) * 1000
            return ValidationResult(
                validator=self.name, ran_at=started,
                duration_ms=duration_ms, success=True, findings=[],
            )

        entity = await self._lookup(ein)
        if entity is None:
            findings.append(ValidationFinding(
                validator=self.name,
                field_path="ein",
                severity="info",
                message="No SAM.gov registration found for EIN",
                details={"ein": ein},
            ))
        elif entity.registration_status == "Active":
            findings.append(ValidationFinding(
                validator=self.name,
                field_path="ein",
                severity="info",
                message="Active SAM.gov registration",
                details={
                    "uei": entity.uei,
                    "cage_code": entity.cage_code,
                    "registration_expiration": entity.exp_date,
                    "business_types": entity.business_types,
                },
            ))
        elif entity.registration_status in ("Expired", "Inactive"):
            findings.append(ValidationFinding(
                validator=self.name,
                field_path="ein",
                severity="warning",
                message=f"SAM.gov registration {entity.registration_status}",
                details={"uei": entity.uei, "exp_date": entity.exp_date},
            ))
        else:
            findings.append(ValidationFinding(
                validator=self.name,
                field_path="ein",
                severity="info",
                message=f"SAM.gov registration status: {entity.registration_status}",
                details={"uei": entity.uei},
            ))

        duration_ms = (time.monotonic() - t0) * 1000
        return ValidationResult(
            validator=self.name, ran_at=started,
            duration_ms=duration_ms, success=True, findings=findings,
        )

    async def prefill(self, submission: Any, just_extracted: dict) -> Optional[PrefillPatch]:
        return None
