"""ConversationController — stateless turn orchestrator (9.e: full flow).

Flow per call:
  1. Load session (KeyError on missing / wrong-tenant)
  2. Append user message (KeyError on finalized / expired)
  3. Extract diff from user message
     - ExtractionOutputError -> graceful degrade (keep current state)
  4. Apply diff via store.apply_submission_diff
     - LobTransitionError -> graceful degrade (keep pre-diff state)
     - Other store errors (concurrent finalize, tenant drift) -> propagate
  5. Run HarnessManager.process -> judge + optional refine
     - If refinement produced new state, persist via update_submission
  6. Generate response via Responder
  7. Append assistant message
  8. Return TurnResult

Error-surface policy:
  - Engine exceptions (RateLimitError, etc.) propagate — RetryingEngine
    handled transients; what reaches here is caller's problem.
  - ExtractionOutputError -> degrade. User turn already recorded; judge
    runs on unchanged state; responder asks a follow-up.
  - LobTransitionError -> degrade. LOB switch silently discarded; other
    fields in the (attempted) diff are discarded too (atomic transaction).
  - KeyError surface is uniform — missing / wrong-tenant / terminal state
    all look the same to the caller (no tenant-existence info leak).

Turn atomicity: each store operation commits independently. A crash
between steps leaves partial state (user message stored, submission
unchanged, no assistant response). Recovery is "caller re-invokes the
turn" — the next extraction sees the full history. True turn-level
atomicity would require a turn_state table; deferred.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from accord_ai.conversation.flow_engine import FlowEngine, FlowState
from accord_ai.extraction.context import EMPTY_CONTEXT, ExtractionContext
from accord_ai.conversation.responder import Responder
from accord_ai.core.diff import LobTransitionError
from accord_ai.core.store import ConcurrencyError, SessionStore
from accord_ai.extraction.extractor import ExtractionOutputError, Extractor
from accord_ai.harness.judge import JudgeVerdict
from accord_ai.harness.manager import HarnessManager
from accord_ai.logging_config import get_logger
from accord_ai.request_context import set_context
from accord_ai.schema import CustomerSubmission
from accord_ai.validation.inline import InlineEnrichmentRunner

_logger = get_logger("conversation_controller")


@dataclass(frozen=True)
class TurnResult:
    submission: CustomerSubmission
    verdict: JudgeVerdict
    assistant_message: str
    is_complete: bool
    # True iff the harness's refiner actually ran on this turn. Lets eval
    # tooling measure the first-pass-extract-pass vs. refiner-rescue split
    # without reaching into the harness internals. Default False keeps the
    # TurnResult(...) constructor backward-compatible for existing tests.
    refined: bool = False


class ConversationController:
    def __init__(
        self,
        store: SessionStore,
        extractor: Extractor,
        harness: HarnessManager,
        responder: Responder,
        inline_runner: Optional[InlineEnrichmentRunner] = None,
        flow_engine: Optional[FlowEngine] = None,
        extraction_context_enabled: bool = True,
    ) -> None:
        self._store = store
        self._extractor = extractor
        self._harness = harness
        self._responder = responder
        self._inline_runner = inline_runner
        self._flow_engine = flow_engine
        self._extraction_context_enabled = extraction_context_enabled

    async def process_turn(
        self,
        *,
        session_id: str,
        user_message: str,
        tenant: Optional[str] = None,
    ) -> TurnResult:
        """Process one user turn.

        Known state-inconsistency windows:

          - Refinement's update_submission (step 5b) is unconditional-overwrite.
            A concurrent commit between steps 4 and 5b is silently lost.
            Acceptable for single-user sessions today. Phase 9 multi-connection
            UIs will want optimistic concurrency (if_updated_at= on
            update_submission).

          - If responder (step 6) raises, steps 2-5 already committed. Session
            state reflects the turn but the conversation log is missing the
            assistant message — not data loss, just an unusual transcript.
            RetryingEngine handles transient responder failures; permanent
            responder errors deserve a fallback canned response (Phase 9).
        """
        # Defensive context injection. Phase 9 middleware is authoritative for
        # API callers; set_context() here tags logs from non-API callers
        # (CLI, batch runners, tests). ContextVars are per-asyncio-task so
        # cross-task leakage is impossible.
        set_context(session_id=session_id, tenant=tenant)

        # 1. Load session
        session = self._store.get_session(session_id, tenant=tenant)
        if session is None:
            raise KeyError(f"session not found: {session_id}")

        _logger.debug(
            "process_turn start: session=%s tenant=%s status=%s",
            session_id, tenant, session.status,
        )

        # 2. Append user message (state-machine enforced at store layer)
        self._store.append_message(
            session_id=session_id,
            role="user",
            content=user_message,
            tenant=tenant,
        )

        # 3. Extract — build flow context from what was asked LAST turn
        extraction_ctx = self._build_extraction_context(session.flow_state_json)
        current = session.submission
        try:
            diff = await self._extractor.extract(
                user_message=user_message,
                current_submission=current,
                context=extraction_ctx,
            )
        except ExtractionOutputError as e:
            _logger.warning("extraction failed — keeping current state: %s", e)
            diff = None

        # 4. Apply diff — skip if empty (no DB round-trip for a no-op extract)
        just_extracted: dict = {}
        if diff is not None and diff.model_fields_set:
            just_extracted = diff.model_dump(mode="json", exclude_none=True)
            try:
                current = self._store.apply_submission_diff(
                    session_id, diff, tenant=tenant,
                )
            except LobTransitionError as e:
                _logger.warning("LOB transition rejected — keeping current state: %s", e)
                just_extracted = {}

        # 4.5 Inline enrichment — fill missing fields from free/fast APIs
        if self._inline_runner is not None and just_extracted:
            try:
                enriched, conflicts = await self._inline_runner.enrich(
                    current, just_extracted
                )
                if conflicts or enriched is not current:
                    if conflicts:
                        existing_conflicts = list(enriched.conflicts)
                        from accord_ai.schema import FieldConflict
                        new_fc = [
                            c if isinstance(c, FieldConflict) else FieldConflict(**c)
                            for c in conflicts
                        ]
                        enriched = enriched.model_copy(
                            update={"conflicts": existing_conflicts + new_fc}
                        )
                    self._store.update_submission(session_id, enriched, tenant=tenant)
                    current = enriched
            except Exception as exc:
                _logger.warning("inline enrichment failed — continuing: %s", exc)

        # Baseline for optimistic-concurrency check on any refinement write.
        # Captured after step 2 (append) + step 4 (apply_diff) so our own
        # commits don't false-positive as "concurrent modifications".
        refreshed = self._store.get_session(session_id, tenant=tenant)
        baseline_updated_at = (
            refreshed.updated_at if refreshed is not None else session.updated_at
        )

        # 5. Harness: judge + optional refine
        harness_result = await self._harness.process(current, user_message)

        # Persist refined state if refinement ran. Refinement is an intentional
        # overwrite (per 5.c contract) — use update_submission, NOT apply_diff.
        # Optimistic check: if a concurrent writer committed during steps 3-5,
        # ConcurrencyError propagates so the caller (API handler) can return 409.
        if harness_result.refined:
            try:
                self._store.update_submission(
                    session_id,
                    harness_result.submission,
                    tenant=tenant,
                    expected_updated_at=baseline_updated_at,
                )
            except ConcurrencyError:
                _logger.warning(
                    "concurrent modification during refinement — surfacing to caller",
                )
                raise

        # 6. Flow engine — determine next question deterministically
        next_question: Optional[str] = None
        if self._flow_engine is not None:
            raw_state = session.flow_state_json
            flow_state = (
                FlowState.from_json(raw_state)
                if raw_state is not None
                else self._flow_engine.initial_state()
            )
            action, new_state = self._flow_engine.next_action(
                flow_state, harness_result.submission
            )
            if action.kind == "ask":
                next_question = action.question
                self._store.update_flow_state(
                    session_id, new_state.to_json(), tenant=tenant
                )
                _logger.debug(
                    "flow_engine ask: session=%s flow=%s q=%s",
                    session_id, action.flow_id, action.question_id,
                )
            else:
                _logger.debug(
                    "flow_engine finalize: session=%s flow=%s",
                    session_id, action.flow_id,
                )

        # 7. Respond
        assistant_message = await self._responder.respond(
            submission=harness_result.submission,
            verdict=harness_result.verdict,
            next_question=next_question,
        )

        # 8. Append assistant message
        self._store.append_message(
            session_id=session_id,
            role="assistant",
            content=assistant_message,
            tenant=tenant,
        )

        _logger.debug(
            "process_turn done: session=%s passed=%s refined=%s",
            session_id, harness_result.verdict.passed, harness_result.refined,
        )

        # 9. Return
        return TurnResult(
            submission=harness_result.submission,
            verdict=harness_result.verdict,
            assistant_message=assistant_message,
            is_complete=harness_result.verdict.passed,
            refined=harness_result.refined,
        )

    def _build_extraction_context(
        self, flow_state_json: Optional[str]
    ) -> ExtractionContext:
        """Build ExtractionContext from the prior FlowState (what was asked last turn).

        Returns EMPTY_CONTEXT when the feature flag is off, no engine is
        wired, no prior state exists, or the last question can no longer be
        resolved (e.g. flows.yaml edited mid-session).
        """
        if not self._extraction_context_enabled or self._flow_engine is None:
            return EMPTY_CONTEXT
        if flow_state_json is None:
            return EMPTY_CONTEXT
        try:
            prior_state = FlowState.from_json(flow_state_json)
        except Exception:
            return EMPTY_CONTEXT
        result = self._flow_engine.last_asked_question(prior_state)
        if result is None:
            return EMPTY_CONTEXT
        flow, question = result
        return ExtractionContext(
            current_flow=flow.id,
            expected_fields=tuple(question.expected_fields),
            question_text=question.text,
        )
