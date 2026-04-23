"""FMCSA SAFER trucking carrier validator — finalize-only.

Applicable when the submission has a DOT number OR is commercial_auto with a
trucking NAICS (484xxx). Fetches operating authority, safety rating, and crash
history from the FMCSA SAFER web service.

Endpoint: GET https://mobile.fmcsa.dot.gov/qc/services/carriers/{dot}?webKey={key}
Free with a key from https://ai.fmcsa.dot.gov/SMS/Docs/DataQ/FMCSA_API.aspx
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
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

_SAFER_URL = "https://mobile.fmcsa.dot.gov/qc/services/carriers/{dot}?webKey={key}"
_HTTP_TIMEOUT_S = 8.0
_CACHE_TTL_S = 86400.0  # 24h — carrier status moves slowly

# FMCSA documents several "AUTHORIZED" variants — all are operating legally.
# Exact-string "AUTHORIZED" would flag "AUTHORIZED FOR HIRE" as an error.
_AUTHORIZED_STATUSES = frozenset({
    "AUTHORIZED",
    "AUTHORIZED FOR HIRE",
    "AUTHORIZED FOR PRIVATE MOTOR CARRIER OF PROPERTY",
    "AUTHORIZED FOR HOUSEHOLD GOODS",
})


@dataclass
class _CarrierData:
    legal_name: str
    operating_status: str
    safety_rating: str
    rated_date: Optional[str]
    out_of_service: bool
    power_units: int
    drivers: int
    crash_count_2yr: int
    inspection_count_2yr: int


def _parse_carrier(raw: dict) -> Optional[_CarrierData]:
    carrier = (raw.get("content") or {}).get("carrier") or {}
    if not carrier:
        return None
    return _CarrierData(
        legal_name=carrier.get("legalName") or carrier.get("name") or "",
        operating_status=(carrier.get("operatingStatus") or "UNKNOWN").upper(),
        safety_rating=(carrier.get("safetyRating") or "").upper(),
        rated_date=carrier.get("safetyRatingDate") or carrier.get("ratingDate"),
        out_of_service=bool(carrier.get("oosDate") or carrier.get("outOfService")),
        power_units=int(carrier.get("totalPowerUnits") or carrier.get("powerUnits") or 0),
        drivers=int(carrier.get("totalDrivers") or carrier.get("drivers") or 0),
        crash_count_2yr=int(
            carrier.get("crashTotal2Yr") or carrier.get("totalAccidents2Yr") or 0
        ),
        inspection_count_2yr=int(
            carrier.get("inspectionTotal2Yr") or carrier.get("totalInspections2Yr") or 0
        ),
    )


def _make_fetcher(web_key: str):
    @ttl_cached(ttl_seconds=_CACHE_TTL_S, key=lambda dot: dot.strip())
    async def _fetch_carrier(dot: str) -> Optional[_CarrierData]:
        url = _SAFER_URL.format(dot=dot.strip(), key=web_key)
        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S) as client:
                r = await client.get(url)
            r.raise_for_status()
            return _parse_carrier(r.json())
        except Exception as exc:
            _logger.debug("FMCSA SAFER fetch failed for DOT %s: %s", dot, exc)
            return None

    return _fetch_carrier


def _resolve_dot(submission: Any) -> Optional[str]:
    """Extract FMCSA DOT number from the submission, wherever it lives."""
    sub_dict = (
        submission.model_dump(mode="json") if hasattr(submission, "model_dump") else {}
    )
    ld = sub_dict.get("lob_details") or {}
    return ld.get("fmcsa_dot_number") or sub_dict.get("fmcsa_dot_number")


def _is_trucking_naics(naics: str) -> bool:
    return str(naics or "").startswith("484")


class FmcsaValidator:
    """Validate trucking carriers via FMCSA SAFER API — finalize-only."""

    name: str = "fmcsa_safer"
    applicable_fields: tuple = ("lob_details.fmcsa_dot_number",)
    inline_eligible: bool = False

    def __init__(self, web_key: str) -> None:
        self._key = web_key
        self._fetch = _make_fetcher(web_key)

    def _is_applicable(self, submission: Any) -> bool:
        dot = _resolve_dot(submission)
        if dot:
            return True
        lob_details = getattr(submission, "lob_details", None)
        is_ca = lob_details is not None and getattr(lob_details, "lob", None) == "commercial_auto"
        naics = getattr(submission, "naics_code", "") or ""
        return is_ca and _is_trucking_naics(naics)

    async def run(self, submission: Any) -> ValidationResult:
        started = datetime.now(tz=timezone.utc)
        t0 = time.monotonic()
        findings: List[ValidationFinding] = []

        if not self._is_applicable(submission):
            duration_ms = (time.monotonic() - t0) * 1000
            return ValidationResult(
                validator=self.name, ran_at=started,
                duration_ms=duration_ms, success=True, findings=[],
            )

        dot = _resolve_dot(submission)
        if not dot:
            findings.append(ValidationFinding(
                validator=self.name,
                field_path="lob_details.fmcsa_dot_number",
                severity="warning",
                message="Trucking NAICS detected but no DOT number provided",
                details={"naics_code": getattr(submission, "naics_code", None)},
            ))
            duration_ms = (time.monotonic() - t0) * 1000
            return ValidationResult(
                validator=self.name, ran_at=started,
                duration_ms=duration_ms, success=True, findings=findings,
            )

        data = await self._fetch(dot)
        if data is None:
            findings.append(ValidationFinding(
                validator=self.name,
                field_path="lob_details.fmcsa_dot_number",
                severity="warning",
                message=f"FMCSA SAFER lookup failed for DOT {dot!r}",
                details={"dot_number": dot},
            ))
            duration_ms = (time.monotonic() - t0) * 1000
            return ValidationResult(
                validator=self.name, ran_at=started,
                duration_ms=duration_ms, success=True, findings=findings,
            )

        if data.operating_status not in _AUTHORIZED_STATUSES:
            findings.append(ValidationFinding(
                validator=self.name,
                field_path="lob_details.fmcsa_dot_number",
                severity="error",
                message=f"FMCSA operating status: {data.operating_status}",
                details={
                    "status": data.operating_status,
                    "out_of_service": data.out_of_service,
                    "dot_number": dot,
                },
            ))

        if data.safety_rating in ("CONDITIONAL", "UNSATISFACTORY"):
            findings.append(ValidationFinding(
                validator=self.name,
                field_path="lob_details.fmcsa_dot_number",
                severity="warning",
                message=f"FMCSA safety rating: {data.safety_rating}",
                details={
                    "rating": data.safety_rating,
                    "rated_date": data.rated_date,
                    "dot_number": dot,
                },
            ))

        # Always emit info finding with full carrier picture for review screen
        findings.append(ValidationFinding(
            validator=self.name,
            field_path="lob_details.fmcsa_dot_number",
            severity="info",
            message="FMCSA carrier data retrieved",
            details={
                "legal_name": data.legal_name,
                "power_units": data.power_units,
                "drivers": data.drivers,
                "crashes_2yr": data.crash_count_2yr,
                "inspections_2yr": data.inspection_count_2yr,
                "dot_number": dot,
            },
        ))

        duration_ms = (time.monotonic() - t0) * 1000
        return ValidationResult(
            validator=self.name, ran_at=started,
            duration_ms=duration_ms, success=True, findings=findings,
        )

    async def prefill(self, submission: Any, just_extracted: dict) -> Optional[PrefillPatch]:
        return None
