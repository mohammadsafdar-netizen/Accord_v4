"""NHTSA Recalls validator — finalize-only, informational.

Checks for open safety recalls on every vehicle in the submission.
Results appear on the finalize review screen as informational findings.

API: https://api.nhtsa.gov/recalls/recallsByVehicle?make={make}&model={model}&modelYear={year}
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

_RECALLS_URL = (
    "https://api.nhtsa.gov/recalls/recallsByVehicle"
    "?make={make}&model={model}&modelYear={year}"
)
_TIMEOUT_S = 3.0


async def _fetch_recalls(year: Any, make: str, model: str) -> List[dict]:
    url = _RECALLS_URL.format(year=year, make=make, model=model)
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
            r = await client.get(url)
        r.raise_for_status()
        data = r.json()
        return data.get("results") or []
    except Exception as exc:
        _logger.debug("NHTSA recalls fetch failed: %s", exc)
        return []


class NhtsaRecallsValidator:
    """Surface open safety recalls at finalize — informational only."""

    name: str = "nhtsa_recalls"
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

            recalls = await _fetch_recalls(year, make, model)
            for recall in recalls:
                subject = recall.get("Subject") or recall.get("subject") or "Unknown recall"
                campaign = recall.get("NHTSACampaignNumber") or recall.get("campaignNumber") or ""
                component = recall.get("Component") or recall.get("component") or ""
                findings.append(ValidationFinding(
                    validator=self.name,
                    field_path=f"lob_details.vehicles[{i}]",
                    severity="info",
                    message=f"Open recall: {subject}",
                    details={"campaign_number": campaign, "component": component},
                ))

        duration_ms = (time.monotonic() - t0) * 1000
        return ValidationResult(
            validator=self.name, ran_at=started,
            duration_ms=duration_ms, success=True, findings=findings,
        )

    async def prefill(self, submission: Any, just_extracted: dict) -> Optional[PrefillPatch]:
        return None
