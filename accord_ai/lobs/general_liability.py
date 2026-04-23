"""General Liability LOB plugin."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from accord_ai.lobs.common import COMMON_CRITICAL
from accord_ai.lobs.registry import register

# GL-specific fields beyond the common baseline.
# Verbatim port from accord_ai_v3/lobs/general_liability/__init__.py.
_GL_SPECIFIC: List[Tuple[str, str]] = [
    ("nature_of_business",           "nature_of_business is required"),
    ("operations_description",       "operations_description is required"),
    ("annual_revenue",               "annual_revenue (annual_gross_receipts) is required"),
]


@dataclass(frozen=True)
class GeneralLiabilityPlugin:
    lob_key: str = "general_liability"

    @property
    def critical_fields(self) -> List[Tuple[str, str]]:
        return [*COMMON_CRITICAL, *_GL_SPECIFIC]


register(GeneralLiabilityPlugin())
