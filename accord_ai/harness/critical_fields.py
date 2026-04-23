"""Per-LOB critical field lists — v4 thin facade over the LOB plugin registry.

All LOB-specific constants now live in accord_ai/lobs/<lob>.py.  This
module exists for import compatibility (tests import _CA_CRITICAL etc.)
and the public get_critical_fields entrypoint used by SchemaJudge.
"""
from __future__ import annotations

from typing import List, Tuple

# Trigger registration of the three built-in plugins.
import accord_ai.lobs  # noqa: F401

from accord_ai.lobs.common import COMMON_CRITICAL as _COMMON_CRITICAL
from accord_ai.lobs.registry import get_critical_fields

CriticalField = Tuple[str, str]

# Backward-compat re-exports — test_judge.py imports these by name.
# Values are identical to the old module-level constants.
_CA_CRITICAL: List[CriticalField] = list(get_critical_fields("commercial_auto"))
_GL_CRITICAL: List[CriticalField] = list(get_critical_fields("general_liability"))
_WC_CRITICAL: List[CriticalField] = list(get_critical_fields("workers_comp"))

__all__ = [
    "CriticalField",
    "get_critical_fields",
    "_COMMON_CRITICAL",
    "_CA_CRITICAL",
    "_GL_CRITICAL",
    "_WC_CRITICAL",
]
