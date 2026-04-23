"""SFT eligibility filter for session-transcript capture (Phase 2.7).

Pure function — no I/O. Caller fetches correction_count and validation_results
and passes them in, keeping this layer unit-testable without DB mocking.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class EligibilityReason:
    eligible: bool
    reason: str


def is_session_sft_eligible(
    session_id: str,
    tenant: str,
    status: str,
    validation_results: Optional[List],
    correction_count: int,
) -> EligibilityReason:
    """Return whether a finalized session qualifies for SFT capture.

    Rules (all must pass):
      1. status is 'finalized'
      2. zero corrections logged against this session
      3. validation_results is not None (results were recorded at finalize time)
      4. no validator returned severity='error' findings (warnings and info are OK)
      5. all validators succeeded (no timeouts / crashes)
    """
    if status != "finalized":
        return EligibilityReason(False, f"status is {status!r}, not 'finalized'")

    if correction_count > 0:
        return EligibilityReason(
            False, f"session had {correction_count} correction(s)"
        )

    if validation_results is None:
        return EligibilityReason(
            False, "no validation results recorded at finalize"
        )

    for result in validation_results:
        if not result.success:
            return EligibilityReason(
                False,
                f"validator {result.validator} failed: {result.error}",
            )
        if any(f.severity == "error" for f in result.findings):
            return EligibilityReason(
                False,
                f"validator {result.validator} emitted error-severity findings",
            )

    return EligibilityReason(True, "clean")
