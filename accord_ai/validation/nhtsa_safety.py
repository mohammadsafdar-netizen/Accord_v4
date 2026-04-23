"""NHTSA Safety Ratings validator — finalize-only, informational.

Returns NCAP crash test scores for each vehicle at finalize review.

API: https://api.nhtsa.gov/SafetyRatings/modelyear/{year}/make/{make}/model/{model}
Free, no key required.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, List, Optional

import httpx

from accord_ai.validation.types import PrefillPatch, ValidationFinding, ValidationResult

_logger = logging.getLogger(__name__)

_SAFETY_URL = (
    "https://api.nhtsa.gov/SafetyRatings/modelyear/{year}/make/{make}/model/{model}"
)
_TIMEOUT_S = 3.0


async def _fetch_safety(year: Any, make: str, model: str) -> List[dict]:
    url = _SAFETY_URL.format(year=year, make=make, model=model)
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
            r = await client.get(url)
        r.raise_for_status()
        data = r.json()
        return data.get("Results") or []
    except Exception as exc:
        _logger.debug("NHTSA safety fetch failed: %s", exc)
        return []


class NhtsaSafetyValidator:
    """Surface NHTSA NCAP safety ratings at finalize — informational only."""

    name: str = "nhtsa_safety"
    applicable_fields: tuple = (
        "lob_details.vehicles[].year",
        "lob_details.vehicles[].make",
        "lob_details.vehicles[].model",
    )
    inline_eligible: bool = False

    async def run(self, submission: Any) -> ValidationResult:
        started = datetime.now(tz=timezone.utc)
        t0 = time.monotonic()
        findings: List[ValidationFinding] = []

        sub_dict = submission.model_dump(mode="json") if hasattr(submission, "model_dump") else {}
        ld = sub_dict.get("lob_details") or {}
        vehicles = ld.get("vehicles") or []

        for i, veh in enumerate(vehicles):
            if not isinstance(veh, dict):
                continue
            year = veh.get("year")
            make = (veh.get("make") or "").strip()
            model = (veh.get("model") or "").strip()
            if not (year and make and model):
                continue

            ratings = await _fetch_safety(year, make, model)
            for rating in ratings[:1]:  # first variant is representative
                overall = rating.get("OverallRating") or rating.get("overallRating") or "N/A"
                vehicle_id = rating.get("VehicleId") or rating.get("vehicleId") or ""
                findings.append(ValidationFinding(
                    validator=self.name,
                    field_path=f"lob_details.vehicles[{i}]",
                    severity="info",
                    message=f"NHTSA safety rating: {overall} stars",
                    details={"overall_rating": overall, "vehicle_id": vehicle_id},
                ))

        duration_ms = (time.monotonic() - t0) * 1000
        return ValidationResult(
            validator=self.name, ran_at=started,
            duration_ms=duration_ms, success=True, findings=findings,
        )

    async def prefill(self, submission: Any, just_extracted: dict) -> Optional[PrefillPatch]:
        return None
