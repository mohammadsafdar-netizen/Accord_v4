"""Result types for the eval scorer (P10.S.11a)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple


@dataclass(frozen=True)
class FieldComparison:
    """One expected (path, value) pair evaluated against v4's submission."""
    v3_path:        str
    v4_path:        Optional[str]        # None if untranslatable
    expected_value: Any
    actual_value:   Any = None
    matched:        bool = False
    reason:         str = ""              # "ok" | "missing" | "mismatch" | "untranslatable"


@dataclass(frozen=True)
class ScoreResult:
    """L3 precision/recall/F1 over one scenario's expected dict."""
    scenario_id:          str
    total_expected:       int                    # count of v3 expectations
    translated:           int                    # count that mapped to v4 (pair-level)
    matched:              int                    # v4-pair matches (precision numerator)
    matched_v3_paths:     int = 0                # v3-path matches (recall numerator — a v3 path counts iff ALL its v4 pairs matched)
    precision:            float = 0.0            # matched / translated
    recall:               float = 0.0            # matched_v3_paths / total_expected
    f1:                   float = 0.0
    comparisons:          Tuple[FieldComparison, ...] = ()
    untranslatable_paths: Tuple[str, ...] = ()   # v3 paths we couldn't map

    def to_dict(self) -> dict:
        return {
            "scenario_id":          self.scenario_id,
            "total_expected":       self.total_expected,
            "translated":           self.translated,
            "matched":              self.matched,
            "matched_v3_paths":     self.matched_v3_paths,
            "precision":            round(self.precision, 4),
            "recall":               round(self.recall, 4),
            "f1":                   round(self.f1, 4),
            "untranslatable_paths": list(self.untranslatable_paths),
            "comparisons": [
                {
                    "v3_path":        c.v3_path,
                    "v4_path":        c.v4_path,
                    "expected_value": c.expected_value,
                    "actual_value":   c.actual_value,
                    "matched":        c.matched,
                    "reason":         c.reason,
                }
                for c in self.comparisons
            ],
        }
