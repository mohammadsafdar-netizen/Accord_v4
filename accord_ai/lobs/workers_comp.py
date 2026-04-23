"""Workers Compensation LOB plugin."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from accord_ai.lobs.common import COMMON_CRITICAL
from accord_ai.lobs.registry import register

# WC-specific fields beyond the common baseline.
# Designed from ACORD 130 minimum required widget set (no v3 reference).
_WC_SPECIFIC: List[Tuple[str, str]] = [
    ("lob_details.payroll_by_class",                           "at least one payroll_by_class entry required"),
    ("lob_details.coverage.employers_liability_per_accident",  "employers_liability_per_accident is required"),
]


@dataclass(frozen=True)
class WorkersCompPlugin:
    lob_key: str = "workers_comp"

    @property
    def critical_fields(self) -> List[Tuple[str, str]]:
        return [*COMMON_CRITICAL, *_WC_SPECIFIC]


register(WorkersCompPlugin())
