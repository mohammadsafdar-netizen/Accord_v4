"""Phone area code validator — finalize-only.

Validates that phone numbers in the submission use real US/CA NANPA area codes.
Uses a committed local CSV (data/area_codes.csv) — no network required.

Catches common typos like 555 (fictional) or transposed digits.
"""

from __future__ import annotations

import csv
import logging
import pathlib
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from accord_ai.validation.types import (
    PrefillPatch,
    ValidationFinding,
    ValidationResult,
)

_logger = logging.getLogger(__name__)

_DEFAULT_CSV_PATH = pathlib.Path(__file__).parent.parent.parent / "data" / "area_codes.csv"
_DIGITS_ONLY = re.compile(r"\D")


@dataclass
class _AreaInfo:
    state: str
    region: str


def _load_csv(path: pathlib.Path) -> Dict[str, _AreaInfo]:
    areas: Dict[str, _AreaInfo] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            code = (row.get("area_code") or "").strip()
            state = (row.get("state") or "").strip()
            region = (row.get("region") or state).strip()
            if code and state:
                areas[code] = _AreaInfo(state=state, region=region)
    return areas


def _extract_area_code(phone: str) -> Optional[str]:
    """Return 3-digit area code string, or None if the number can't be parsed."""
    digits = _DIGITS_ONLY.sub("", phone)
    # Strip leading +1 country code
    if digits.startswith("1") and len(digits) == 11:
        digits = digits[1:]
    if len(digits) < 10:
        return None
    return digits[:3]


class PhoneAreaValidator:
    """Validate phone area codes against NANPA data — finalize-only."""

    name: str = "phone_area"
    applicable_fields: tuple = ("contacts[].phone",)
    inline_eligible: bool = False

    def __init__(self, csv_path: Optional[pathlib.Path] = None) -> None:
        path = csv_path or _DEFAULT_CSV_PATH
        self._areas = _load_csv(path)

    async def run(self, submission: Any) -> ValidationResult:
        started = datetime.now(tz=timezone.utc)
        t0 = time.monotonic()
        findings: List[ValidationFinding] = []

        sub_dict = (
            submission.model_dump(mode="json")
            if hasattr(submission, "model_dump") else {}
        )

        # Check submission-level phone
        top_phone = sub_dict.get("phone")
        if top_phone:
            area = _extract_area_code(top_phone)
            if area is None:
                findings.append(ValidationFinding(
                    validator=self.name,
                    field_path="phone",
                    severity="warning",
                    message=f"Phone number format unparseable: {top_phone!r}",
                    details={"phone": top_phone},
                ))
            elif area not in self._areas:
                findings.append(ValidationFinding(
                    validator=self.name,
                    field_path="phone",
                    severity="error",
                    message=f"Area code {area} is not a valid US/CA area code",
                    details={"area_code": area, "phone": top_phone},
                ))

        # Check contacts
        contacts = sub_dict.get("contacts") or []
        for i, contact in enumerate(contacts):
            if not isinstance(contact, dict):
                continue
            phone = contact.get("phone") or ""
            if not phone:
                continue
            area = _extract_area_code(phone)
            if area is None:
                findings.append(ValidationFinding(
                    validator=self.name,
                    field_path=f"contacts[{i}].phone",
                    severity="warning",
                    message=f"Phone number format unparseable: {phone!r}",
                    details={"phone": phone},
                ))
            elif area not in self._areas:
                findings.append(ValidationFinding(
                    validator=self.name,
                    field_path=f"contacts[{i}].phone",
                    severity="error",
                    message=f"Area code {area} is not a valid US/CA area code",
                    details={"area_code": area, "phone": phone},
                ))

        duration_ms = (time.monotonic() - t0) * 1000
        return ValidationResult(
            validator=self.name, ran_at=started,
            duration_ms=duration_ms, success=True, findings=findings,
        )

    async def prefill(self, submission: Any, just_extracted: dict) -> Optional[PrefillPatch]:
        return None
