"""Harness manager — orchestrates judge + refine on extracted submissions.

Scope (pinned in 5.c):
  input:  CustomerSubmission (already-extracted) + user_message
  output: ManagerResult — submission, final verdict, refined flag
  NOT in scope: extraction (that's Phase 7).

Loop (5.d):
  - max_refines=0: refinement fully disabled — returns initial state + verdict
  - max_refines>=1: up to N refine passes, loop exits early on verdict.passed
  - RefinerOutputError mid-loop: stop. Keep any prior successful refinement.
  - Engine exceptions propagate — caller owns retry/fallback.
"""
from __future__ import annotations

from dataclasses import dataclass

from accord_ai.harness.judge import JudgeVerdict, SchemaJudge
from accord_ai.harness.refiner import Refiner, RefinerOutputError
from accord_ai.logging_config import get_logger
from accord_ai.schema import CustomerSubmission

_logger = get_logger("harness_manager")


@dataclass(frozen=True)
class ManagerResult:
    """Outcome of a judge→refine cycle.

    Fields:
      submission — best state produced (original OR any successful refinement)
      verdict    — final judge's verdict on `submission`
      refined    — True iff at least one refine pass produced a valid submission
    """
    submission: CustomerSubmission
    verdict: JudgeVerdict
    refined: bool


class HarnessManager:
    """Judge + refine orchestrator. Bounded loop per call."""

    def __init__(
        self,
        judge: SchemaJudge,
        refiner: Refiner,
        *,
        max_refines: int = 1,
    ) -> None:
        if max_refines < 0:
            raise ValueError(f"max_refines must be >= 0, got {max_refines}")
        self._judge = judge
        self._refiner = refiner
        self._max_refines = max_refines

    async def process(
        self,
        submission: CustomerSubmission,
        user_message: str,
    ) -> ManagerResult:
        """Judge, then refine up to `max_refines` times, returning the best state.

        Mid-loop exception semantics: if an engine exception (RateLimitError,
        APITimeoutError, etc.) propagates from the refiner during iteration N,
        the caller loses any partial refinement that succeeded on iterations
        < N — the exception surfaces instead of a ManagerResult. This is the
        intended trade-off: the caller owns retry/fallback and can re-invoke
        process() with the original submission. If partial-state recovery
        becomes important, introduce a PartialResultError(result, inner)
        wrapper rather than mutating this return shape.
        """
        current = submission
        verdict = self._judge.evaluate(current)
        # INFO-level on failure only: surfaces the refiner-trigger
        # condition + which fields the first-pass extractor missed, so
        # tuning/regression investigations have actionable data under a
        # production-default log level. Passing verdicts stay quiet at
        # DEBUG — no spam on the happy path.
        if verdict.passed:
            _logger.debug(
                "initial judge: passed=True reasons=0"
            )
        else:
            _logger.info(
                "initial judge: passed=False reasons=%d failed_paths=%s",
                len(verdict.reasons), list(verdict.failed_paths),
            )

        refined_any = False

        for attempt in range(self._max_refines):
            if verdict.passed:
                break

            try:
                refined = await self._refiner.refine(
                    original_user_message=user_message,
                    current_submission=current,
                    verdict=verdict,
                )
            except RefinerOutputError as e:
                _logger.warning(
                    "refiner output invalid — stopping refinement: %s", e
                )
                break
            # Engine exceptions propagate.

            refined_any = True
            current = refined
            verdict = self._judge.evaluate(current)
            if verdict.passed:
                _logger.info(
                    "post-refine judge: attempt=%d passed=True", attempt + 1,
                )
            else:
                _logger.info(
                    "post-refine judge: attempt=%d passed=False failed_paths=%s",
                    attempt + 1, list(verdict.failed_paths),
                )

        return ManagerResult(submission=current, verdict=verdict, refined=refined_any)
