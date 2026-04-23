"""Census NAICS 2022 validator — inline-eligible code lookup.

Loads data/naics_2022.csv (committed, ~5-year change cadence). On inline
prefill: if `naics_code` was just extracted and `naics_description` is
missing, fills the description from the CSV title. On finalize run: flags
unknown codes (warning) and description/title mismatches (warning).
"""

from __future__ import annotations

import csv
import logging
import pathlib
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from accord_ai.validation.types import (
    PrefillPatch,
    ValidationFinding,
    ValidationResult,
)

_logger = logging.getLogger(__name__)

_CSV_PATH = pathlib.Path(__file__).parent.parent.parent / "data" / "naics_2022.csv"

def _load_index(path: pathlib.Path = _CSV_PATH) -> Dict[str, str]:
    idx: Dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            code = (row.get("code") or "").strip()
            title = (row.get("title") or "").strip()
            if code and title:
                idx[code] = title
    return idx


# Eagerly loaded once at import time — eliminates lazy-load race and event-loop blocking.
_NAICS_INDEX: Dict[str, str] = _load_index()


def _get_index() -> Dict[str, str]:
    return _NAICS_INDEX


def lookup_naics(code: str) -> Optional[str]:
    """Return the NAICS title for a code, or None if not found."""
    return _get_index().get(code.strip())


class NaicsValidator:
    """NAICS 2022 code validator — inline-eligible (fills description from code)."""

    name: str = "census_naics"
    applicable_fields: tuple = ("naics_code",)
    inline_eligible: bool = True

    async def run(self, submission: Any) -> ValidationResult:
        started = datetime.now(tz=timezone.utc)
        t0 = time.monotonic()
        findings: list[ValidationFinding] = []
        idx = _get_index()

        sub_dict = submission.model_dump(mode="json") if hasattr(submission, "model_dump") else {}
        code = (sub_dict.get("naics_code") or "").strip()
        desc = (sub_dict.get("naics_description") or "").strip()

        if code:
            title = idx.get(code)
            if title is None:
                findings.append(ValidationFinding(
                    validator=self.name,
                    field_path="naics_code",
                    severity="warning",
                    message=f"NAICS code {code!r} is not in the 2022 code list",
                    details={"code": code},
                ))
            elif desc and desc.lower() != title.lower():
                findings.append(ValidationFinding(
                    validator=self.name,
                    field_path="naics_description",
                    severity="warning",
                    message=(
                        f"naics_description {desc!r} does not match "
                        f"the 2022 title for {code!r}: {title!r}"
                    ),
                    details={"code": code, "stated_description": desc, "official_title": title},
                ))

        duration_ms = (time.monotonic() - t0) * 1000
        return ValidationResult(
            validator=self.name, ran_at=started,
            duration_ms=duration_ms, success=True, findings=findings,
        )

    async def prefill(self, submission: Any, just_extracted: dict) -> Optional[PrefillPatch]:
        """Inline: if naics_code was just extracted, fill naics_description if missing."""
        if "naics_code" not in just_extracted:
            return None

        code = (just_extracted.get("naics_code") or "").strip()
        if not code:
            return None

        sub_dict = submission.model_dump(mode="json") if hasattr(submission, "model_dump") else {}
        existing_desc = (sub_dict.get("naics_description") or "").strip()
        if existing_desc:
            return None  # already populated — don't overwrite

        title = _get_index().get(code)
        if not title:
            return None  # unknown code — nothing to fill

        return PrefillPatch(
            patch={"naics_description": title},
            source=self.name,
            confidence=1.0,
        )
