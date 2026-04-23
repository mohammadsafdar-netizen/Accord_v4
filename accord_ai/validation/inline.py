"""Inline enrichment runner — fires during conversation turns for free/fast validators.

Only validators with inline_eligible=True are called. Results are merged back
into the submission before the turn result is returned. Conflicts are recorded
on the submission for review at the finalize screen.

Feature flag: ACCORD_INLINE_ENRICHMENT=true (default on).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

if TYPE_CHECKING:
    from accord_ai.schema import CustomerSubmission, FieldConflict
    from accord_ai.validation.types import PrefillPatch, Validator

_logger = logging.getLogger(__name__)


def _deep_merge(target: dict, source: dict) -> None:
    """Recursive dict merge. List values keyed by int → index-based update."""
    for key, value in source.items():
        if (
            isinstance(value, dict)
            and key in target
            and isinstance(target[key], dict)
        ):
            _deep_merge(target[key], value)
        elif (
            isinstance(value, dict)
            and all(isinstance(k, int) for k in value)
            and key in target
            and isinstance(target[key], list)
        ):
            # Index-keyed dict → update specific list items
            lst = list(target[key])
            for idx, partial in value.items():
                if idx < len(lst):
                    if isinstance(lst[idx], dict) and isinstance(partial, dict):
                        lst[idx] = {**lst[idx], **partial}
                    else:
                        lst[idx] = partial
            target[key] = lst
        else:
            target[key] = value


def _apply_patch(sub: "CustomerSubmission", patch: dict) -> "CustomerSubmission":
    from accord_ai.schema import CustomerSubmission
    base = sub.model_dump(mode="json")
    _deep_merge(base, patch)
    return CustomerSubmission.model_validate(base)


class InlineEnrichmentRunner:
    """Runs inline-eligible validators after each extraction turn."""

    def __init__(
        self,
        validators: List["Validator"],
        timeout_s: float = 2.0,
        enabled: bool = True,
    ) -> None:
        self._inline = [v for v in validators if v.inline_eligible]
        self._timeout_s = timeout_s
        self._enabled = enabled

    async def enrich(
        self,
        submission: "CustomerSubmission",
        just_extracted: dict,
    ) -> Tuple["CustomerSubmission", List["FieldConflict"]]:
        """Apply inline prefill validators and return (enriched_submission, conflicts)."""
        if not self._enabled or not self._inline:
            return submission, []

        async def _run_one(v: "Validator") -> Optional["PrefillPatch"]:
            try:
                return await asyncio.wait_for(
                    v.prefill(submission, just_extracted),
                    timeout=self._timeout_s,
                )
            except asyncio.TimeoutError:
                _logger.warning("inline validator=%s timed out — skipped", v.name)
                return None
            except Exception as exc:
                _logger.warning(
                    "inline validator=%s raised %s: %s", v.name, type(exc).__name__, exc
                )
                return None

        patches = await asyncio.gather(*(_run_one(v) for v in self._inline))

        merged = submission
        all_conflicts: List[Any] = []
        for patch in patches:
            if patch is None:
                continue
            try:
                merged = _apply_patch(merged, patch.patch)
                all_conflicts.extend(patch.conflicts)
            except Exception as exc:
                _logger.warning("inline patch apply failed: %s", exc)

        return merged, all_conflicts


def build_inline_runner(
    validators: Optional[List["Validator"]] = None,
    timeout_s: float = 2.0,
) -> InlineEnrichmentRunner:
    enabled = os.environ.get("ACCORD_INLINE_ENRICHMENT", "true").lower() not in (
        "0", "false", "no"
    )
    return InlineEnrichmentRunner(
        validators=validators or [],
        timeout_s=timeout_s,
        enabled=enabled,
    )
