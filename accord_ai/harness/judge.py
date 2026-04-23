"""Schema-based judge for CustomerSubmission — v3-aligned, declarative.

Rule source: per-LOB `critical` field lists in
:mod:`accord_ai.harness.critical_fields`. The judge walks each path,
checks for None/empty, and emits a `(reason, failed_path)` pair for
every miss. Plus one cross-field invariant:
``policy_dates.effective_date <= policy_dates.expiration_date`` when
both are set.

Rule data is declarative so adding a new critical field is a one-line
addition to ``critical_fields.py`` — no judge changes, no test changes
(the parametrized judge tests pick it up automatically).

Pure synchronous — no LLM, no I/O. A future LLMJudge can be added as a
separate implementation of a shared protocol.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Tuple

from accord_ai.harness.critical_fields import get_critical_fields
from accord_ai.schema import CustomerSubmission


@dataclass(frozen=True)
class JudgeVerdict:
    """Outcome of judging a submission.

    - passed: True iff no rule fired
    - reasons: human-readable failure lines (empty if passed); goes into
      the refiner prompt as "fix these specific problems"
    - failed_paths: dot-path identifiers of the offending fields; lets
      the refiner target changes rather than re-extracting everything
    """
    passed: bool
    reasons: Tuple[str, ...] = ()
    failed_paths: Tuple[str, ...] = ()


# Path walker. Duplicated from accord_ai.forms.mapper._resolve on
# purpose so the harness module doesn't depend on forms/. Returns None
# on any miss (missing attribute, out-of-range index, non-container
# intermediate).
_SEGMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\[(\d+)\])?$")


def _resolve(obj: Any, path: str) -> Any:
    current = obj
    for raw in path.split("."):
        if current is None:
            return None
        m = _SEGMENT.match(raw)
        if not m:
            return None
        attr, idx = m.group(1), m.group(2)
        current = getattr(current, attr, None)
        if idx is not None:
            if current is None:
                return None
            try:
                current = current[int(idx)]
            except (IndexError, TypeError, KeyError):
                return None
    return current


def _is_empty(v: Any) -> bool:
    """None / "" / whitespace-only / [] / {} all count as "not present".

    Zero and False are real values, not empty — v3 treats them as
    present and the judge must too (e.g. `hazmat: false` is a valid
    answer, not a missing field).
    """
    if v is None:
        return True
    if isinstance(v, str) and not v.strip():
        return True
    if isinstance(v, (list, dict)) and not v:
        return True
    return False


class SchemaJudge:
    """Rule-based evaluator. Deterministic, v3-aligned."""

    def evaluate(self, submission: CustomerSubmission) -> JudgeVerdict:
        reasons: List[str] = []
        failed_paths: List[str] = []

        # --- lob_details presence gate ---------------------------------
        # Forms dispatch on lob_details.lob; without it the pipeline has
        # nothing to fill. Emitted before the per-LOB loop so the LOB
        # discriminator is the first thing the refiner sees.
        lob = submission.lob_details.lob if submission.lob_details else ""
        if not lob:
            reasons.append(
                "lob_details is required "
                "(commercial_auto / general_liability / workers_comp)"
            )
            failed_paths.append("lob_details")

        # --- Per-LOB critical fields (declarative loop) ----------------
        # Unknown LOB falls back to the LOB-agnostic common list via
        # get_critical_fields(); the lob_details gate above already
        # flagged the missing discriminator.
        for path, reason in get_critical_fields(lob):
            value = _resolve(submission, path)
            if _is_empty(value):
                reasons.append(reason)
                failed_paths.append(path)

        # --- Cross-field invariant: policy date ordering ---------------
        # Distinct from the effective_date-required path above — this
        # fires only when BOTH dates are set and they're out of order.
        if submission.policy_dates is not None:
            eff = submission.policy_dates.effective_date
            exp = submission.policy_dates.expiration_date
            if eff is not None and exp is not None and eff > exp:
                reasons.append(
                    f"policy_dates.effective_date ({eff}) is after "
                    f"expiration_date ({exp})"
                )
                failed_paths.append("policy_dates.effective_date")
                failed_paths.append("policy_dates.expiration_date")

        return JudgeVerdict(
            passed=not reasons,
            reasons=tuple(reasons),
            failed_paths=tuple(failed_paths),
        )
