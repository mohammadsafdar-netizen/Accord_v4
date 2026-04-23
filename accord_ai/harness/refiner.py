"""Refiner — re-prompts an LLM with a judge's verdict, returns a corrected submission.

The engine is injected so a caller (typically HarnessManager in 5.c) can
point the refiner at a different provider from the extractor — memory says
the refiner is the cold-path external-LLM exception to the localhost-only
privacy rule.

Error model:
  - Engine exceptions propagate (RateLimitError, APITimeoutError, etc.) —
    callers decide retry semantics via whichever engine wrapper they use.
  - RefinerOutputError is raised when the LLM returns text that isn't valid
    JSON or doesn't parse as CustomerSubmission. Retrying the same refiner
    call won't help in that case; caller decides to give up or re-prompt.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from accord_ai.config import Settings
    from accord_ai.harness.refiner_cascade import CascadingRefiner

from accord_ai.harness.judge import JudgeVerdict
from accord_ai.llm.engine import Engine, Message
from accord_ai.llm.prompts import refiner as refiner_prompts
from accord_ai.llm.prompts import render
from accord_ai.llm.prompts.parsing import parse_submission_output
from accord_ai.logging_config import get_logger
from accord_ai.schema import CustomerSubmission


def _refiner_system_prompt() -> str:
    """Pick the refiner's system prompt based on the harness feature flag.

    ACCORD_REFINER_HARNESS=1 (default): SYSTEM_V3 with the ported v3
    living-harness content. The refiner's job (take failed_paths,
    return a corrected full submission) is rule-heavy; harness rules
    like "no claims → leave loss_history empty" and "never invent
    dates" directly shape refiner output quality.

    ACCORD_REFINER_HARNESS=0: falls back to SYSTEM_V2 (anti-hallucination
    + LOB/enum guardrails, no harness rules). Flip to 0 if a future
    eval shows refiner-path regression traceable to harness content.
    """
    if os.environ.get("ACCORD_REFINER_HARNESS", "1") == "1":
        return refiner_prompts.SYSTEM_V3
    return refiner_prompts.SYSTEM_V2

_logger = get_logger("refiner")

# Cached schema for the guided_json kwarg. Same surface as the extractor
# — the refiner emits a full CustomerSubmission, not a diff, so structured
# output buys the same malformed-JSON immunity there.
_SCHEMA_DICT = CustomerSubmission.model_json_schema()


class RefinerOutputError(ValueError):
    """Refiner produced output that isn't a valid CustomerSubmission.

    Wraps the underlying json.JSONDecodeError or pydantic.ValidationError.
    Non-retryable at this layer — the LLM emitted garbage for the prompt.
    """


class Refiner:
    """Takes a judge's verdict, asks the LLM to fix the flagged fields."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    async def refine(
        self,
        *,
        original_user_message: str,
        current_submission: CustomerSubmission,
        verdict: JudgeVerdict,
    ) -> CustomerSubmission:
        reasons_text = (
            "\n".join(f"- {r}" for r in verdict.reasons) or "(none)"
        )
        paths_text = (
            "\n".join(f"- {p}" for p in verdict.failed_paths) or "(none)"
        )

        user_content = render(
            refiner_prompts.USER_TEMPLATE_V1,
            user_message=original_user_message,
            # exclude_none: the refiner only needs to see what's set plus
            # the verdict's failed_paths — null fields just waste tokens.
            current_submission_json=current_submission.model_dump_json(
                indent=2, exclude_none=True,
            ),
            verdict_reasons=reasons_text,
            failed_paths=paths_text,
        )
        messages: List[Message] = [
            # Harness-gated: ACCORD_REFINER_HARNESS=1 (default) →
            # SYSTEM_V3 (SYSTEM_V2 + HARNESS_RULES). =0 → SYSTEM_V2.
            # Postmortem 1A quarantined harness to the refiner path
            # after it regressed the extractor — keep the opt-out
            # here so a refiner-eval regression is a flag flip, not
            # a code change.
            {"role": "system", "content": _refiner_system_prompt()},
            {"role": "user", "content": user_content},
        ]

        _logger.debug(
            "refiner call: reasons=%d failed_paths=%d",
            len(verdict.reasons), len(verdict.failed_paths),
        )

        # Budget + guided_json: refiner emits a FULL CustomerSubmission
        # (not a diff), so a realistic multi-driver / multi-vehicle refine
        # serializes to ~1200-1800 JSON tokens. 1024 clips mid-string and
        # guided_json's xgrammar can't rescue a budget-truncated grammar
        # state — observed as "Unterminated string" parse errors from the
        # real vLLM runs. Match the extractor's 2048 cap.
        response = await self._engine.generate(
            messages,
            max_tokens=2048,
            json_schema=_SCHEMA_DICT,
        )
        return parse_submission_output(response.text, error_cls=RefinerOutputError)


def build_refiner(settings: "Settings") -> "Optional[CascadingRefiner]":
    """Return a CascadingRefiner if enabled; None when the cascade is disabled.

    ACCORD_DISABLE_REFINEMENT=true (default in production) → returns None.
    Caller falls back to the plain Refiner(engine) path.

    Client construction is conditional on the relevant API key being present
    in the environment.  LocalRefiner is always added as the final fallback
    when the cascade is active.
    """
    if os.environ.get("ACCORD_DISABLE_REFINEMENT", "true").lower() in ("1", "true", "yes"):
        return None

    # Deferred imports — avoid loading httpx / cascade at import time on
    # the hot path where ACCORD_DISABLE_REFINEMENT=true.
    from accord_ai.harness.refiner_cascade import CascadingRefiner
    from accord_ai.harness.refiner_clients.claude import ClaudeRefiner
    from accord_ai.harness.refiner_clients.gemini import GeminiRefiner
    from accord_ai.harness.refiner_clients.local import LocalRefiner
    from accord_ai.llm import build_refiner_engine

    clients = []

    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        clients.append(GeminiRefiner(api_key=gemini_key))

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        clients.append(ClaudeRefiner(api_key=anthropic_key))

    # Local is always last — uses the already-configured refiner engine.
    local_engine = build_refiner_engine(settings)
    clients.append(LocalRefiner(Refiner(local_engine)))

    return CascadingRefiner(clients)
