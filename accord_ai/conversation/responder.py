"""Responder — submission + judge verdict → next user-facing message.

LLM-generated (per decision #5) because LOB-conditional phrasing doesn't
fit a template system well: the natural next question depends on LOB +
what's present + what's missing, which combine combinatorially.

Input shape is deliberately narrow: just (submission, verdict). No
conversation history — MVP. If continuity phrasing becomes a requirement,
add a `history=` kwarg at that time; Responder has exactly one call site
(ConversationController.process_turn), so the refactor is localized.

Output is plain text — ready to hand straight to
store.append_message(role="assistant", ...). No JSON parsing; just strip.
"""
from __future__ import annotations

from typing import List, Optional

from accord_ai.harness.judge import JudgeVerdict
from accord_ai.llm.engine import Engine, Message
from accord_ai.llm.prompts import render
from accord_ai.llm.prompts import responder as responder_prompts
from accord_ai.logging_config import get_logger
from accord_ai.schema import CustomerSubmission

_logger = get_logger("responder")


class Responder:
    """Generates the next assistant message for a user turn."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    async def respond(
        self,
        *,
        submission: CustomerSubmission,
        verdict: JudgeVerdict,
        next_question: Optional[str] = None,
    ) -> str:
        """Return the next message to send the user. Plain text, stripped.

        next_question: when provided by the flow engine, appended as a hint
        so the LLM asks exactly this question rather than choosing freely.
        """
        reasons_text = (
            "\n".join(f"- {r}" for r in verdict.reasons) or "(none)"
        )
        paths_text = (
            "\n".join(f"- {p}" for p in verdict.failed_paths) or "(none)"
        )

        # exclude_none drops a 6 KB tree of nulls on empty submissions —
        # big prompt-cost savings on /start-session, and a clearer signal
        # of "what's actually known" to the responder LLM.
        user_content = render(
            responder_prompts.USER_TEMPLATE_V1,
            current_submission_json=submission.model_dump_json(
                indent=2, exclude_none=True,
            ),
            verdict_status="complete" if verdict.passed else "needs-info",
            verdict_reasons=reasons_text,
            failed_paths=paths_text,
        )
        if next_question:
            user_content += f"\n\nNext question to ask the user: {next_question}"
        messages: List[Message] = [
            {"role": "system", "content": responder_prompts.SYSTEM_V1},
            {"role": "user", "content": user_content},
        ]

        _logger.debug(
            "responder call: verdict.passed=%s reasons=%d",
            verdict.passed, len(verdict.reasons),
        )
        # 2-3 sentence conversational reply — ~200 tokens max. Keeping the
        # budget tight shortens latency and caps cost on hot sessions.
        response = await self._engine.generate(messages, max_tokens=400)
        return response.text.strip()
