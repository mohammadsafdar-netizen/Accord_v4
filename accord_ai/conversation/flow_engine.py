"""Deterministic flow engine (Phase 3.2).

Replaces LLM-driven question selection with a pure-Python state machine
over the flows YAML document. Given a FlowState and CustomerSubmission,
next_action() returns what to do next: ask a specific question or finalize.

Public API
----------
FlowEngine(doc)                    — build from a loaded FlowsDocument
engine.initial_state()             — FlowState for a new session
engine.next_action(state, sub)     — (NextAction, new FlowState)
evaluate_condition(cond, sub)      — exported for unit testing
FlowState.to_json() / from_json()  — persistence helpers (used by store)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import FrozenSet, Literal, Optional, Tuple

from accord_ai.conversation.flow_schema import (
    AllCondition,
    AnyCondition,
    Condition,
    FieldEqualsCondition,
    FieldSetCondition,
    Flow,
    FlowsDocument,
    Question,
)
from accord_ai.harness.judge import _is_empty, _resolve
from accord_ai.schema import CustomerSubmission


class FlowCycleError(RuntimeError):
    """Raised when transition resolution visits a flow twice (infinite loop)."""


@dataclass(frozen=True)
class FlowState:
    """Immutable per-session engine state — persisted in sessions.flow_state_json."""

    current_flow: str
    asked_questions: Tuple[str, ...] = ()

    def to_json(self) -> str:
        return json.dumps(
            {
                "current_flow": self.current_flow,
                "asked_questions": list(self.asked_questions),
            },
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, s: str) -> FlowState:
        d = json.loads(s)
        return cls(
            current_flow=d["current_flow"],
            asked_questions=tuple(d.get("asked_questions", [])),
        )


@dataclass(frozen=True)
class NextAction:
    """Result of one engine tick.

    kind == "ask"      → present `question` text to the user
    kind == "finalize" → all flows exhausted; caller triggers finalization
    """

    kind: Literal["ask", "finalize"]
    question: Optional[str] = None       # populated when kind == "ask"
    question_id: Optional[str] = None
    flow_id: Optional[str] = None
    missing_required: Tuple[str, ...] = ()


def evaluate_condition(cond: Condition, submission: CustomerSubmission) -> bool:
    """Evaluate one Condition node against the current submission."""
    if isinstance(cond, FieldSetCondition):
        return not _is_empty(_resolve(submission, cond.path))
    if isinstance(cond, FieldEqualsCondition):
        return _resolve(submission, cond.path) == cond.value
    if isinstance(cond, AllCondition):
        return all(evaluate_condition(c, submission) for c in cond.conditions)
    if isinstance(cond, AnyCondition):
        return any(evaluate_condition(c, submission) for c in cond.conditions)
    raise TypeError(f"Unknown condition type: {type(cond)}")  # pragma: no cover


class FlowEngine:
    """Deterministic conversation flow state machine.

    Thread-safe: the engine is stateless after construction; all session
    state lives in FlowState which the caller persists.
    """

    def __init__(self, doc: FlowsDocument) -> None:
        self._doc = doc
        self._by_id: dict[str, Flow] = {f.id: f for f in doc.flows}

    def initial_state(self) -> FlowState:
        return FlowState(current_flow=self._doc.initial_flow)

    def next_action(
        self,
        state: FlowState,
        submission: CustomerSubmission,
        *,
        _visited: Optional[FrozenSet[str]] = None,
    ) -> Tuple[NextAction, FlowState]:
        """Return (NextAction, updated FlowState) for one engine tick.

        Callers must persist the new FlowState when kind == "ask".
        _visited is for internal cycle detection — leave at default None.
        """
        visited: FrozenSet[str] = _visited if _visited is not None else frozenset()
        if state.current_flow in visited:
            raise FlowCycleError(
                f"cycle at flow {state.current_flow!r}; "
                f"path={sorted(visited | {state.current_flow})}"
            )
        visited = visited | {state.current_flow}

        flow = self._by_id.get(state.current_flow)
        if flow is None:
            raise KeyError(f"unknown flow: {state.current_flow!r}")

        missing = tuple(
            p for p in flow.required_fields
            if _is_empty(_resolve(submission, p))
        )

        # Find the first pending (not yet asked, not skippable) question.
        # asked_questions stores qualified "flow_id.question_id" keys so
        # last_asked_question() can resolve them across flow transitions.
        for q in flow.questions:
            qualified_key = f"{state.current_flow}.{q.id}"
            if qualified_key in state.asked_questions:
                continue
            if q.skip_when is not None and evaluate_condition(q.skip_when, submission):
                continue
            new_state = FlowState(
                current_flow=state.current_flow,
                asked_questions=state.asked_questions + (qualified_key,),
            )
            return (
                NextAction(
                    kind="ask",
                    question=q.text,
                    question_id=q.id,
                    flow_id=state.current_flow,
                    missing_required=missing,
                ),
                new_state,
            )

        # All questions exhausted — resolve the next flow transition
        next_id = self._pick_next_flow(flow, submission)
        if next_id is None:
            # Terminal (finalize node or no matching transition)
            return (
                NextAction(
                    kind="finalize",
                    flow_id=state.current_flow,
                    missing_required=missing,
                ),
                state,
            )

        new_state = FlowState(current_flow=next_id, asked_questions=())
        return self.next_action(new_state, submission, _visited=visited)

    def last_asked_question(
        self, state: FlowState
    ) -> Optional[tuple[Flow, Question]]:
        """Resolve the most recently asked question to its (Flow, Question) pair.

        Returns None when asked_questions is empty, the qualified key is
        malformed, the flow no longer exists, or the question was removed.
        Callers should fall back to EMPTY_CONTEXT on None.
        """
        if not state.asked_questions:
            return None
        qualified = state.asked_questions[-1]
        flow_id, sep, question_id = qualified.partition(".")
        if not sep or not question_id:
            return None
        try:
            flow = self._doc.by_id(flow_id)
        except KeyError:
            return None
        for q in flow.questions:
            if q.id == question_id:
                return flow, q
        return None

    def _pick_next_flow(
        self, flow: Flow, submission: CustomerSubmission
    ) -> Optional[str]:
        """Return the id of the first matching transition, or None."""
        for trans in flow.next:
            if trans.when is None or evaluate_condition(trans.when, submission):
                return trans.flow
        return None
