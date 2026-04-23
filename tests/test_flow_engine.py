"""Tests for flow_engine.py (Phase 3.2) — 18 tests."""
from __future__ import annotations

import pytest

from accord_ai.conversation.flow_engine import (
    FlowCycleError,
    FlowEngine,
    FlowState,
    NextAction,
    evaluate_condition,
)
from accord_ai.conversation.flow_schema import (
    AllCondition,
    AnyCondition,
    FieldEqualsCondition,
    FieldSetCondition,
    Flow,
    FlowTransition,
    FlowsDocument,
    Question,
)
from accord_ai.schema import CustomerSubmission


# ---------------------------------------------------------------------------
# Helpers — minimal synthetic flow documents so tests don't depend on
# flows.yaml content
# ---------------------------------------------------------------------------

def _single_flow_doc(flow: Flow, *, initial: str | None = None) -> FlowsDocument:
    return FlowsDocument(
        version="1",
        initial_flow=initial or flow.id,
        flows=[flow],
    )


def _doc_with_flows(*flows: Flow, initial: str) -> FlowsDocument:
    return FlowsDocument(version="1", initial_flow=initial, flows=list(flows))


def _empty_sub() -> CustomerSubmission:
    return CustomerSubmission()


def _sub_with(**kwargs) -> CustomerSubmission:
    return CustomerSubmission(**kwargs)


# ---------------------------------------------------------------------------
# evaluate_condition (4 tests)
# ---------------------------------------------------------------------------


def test_evaluate_field_set_true():
    cond = FieldSetCondition(path="business_name")
    sub = _sub_with(business_name="Acme")
    assert evaluate_condition(cond, sub) is True


def test_evaluate_field_set_false():
    cond = FieldSetCondition(path="business_name")
    assert evaluate_condition(cond, _empty_sub()) is False


def test_evaluate_field_equals_match():
    cond = FieldEqualsCondition(path="ein", value="12-3456789")
    sub = _sub_with(ein="12-3456789")
    assert evaluate_condition(cond, sub) is True


def test_evaluate_field_equals_no_match():
    cond = FieldEqualsCondition(path="ein", value="12-3456789")
    assert evaluate_condition(cond, _empty_sub()) is False


# ---------------------------------------------------------------------------
# FlowEngine unit tests (14 tests)
# ---------------------------------------------------------------------------


def test_initial_state_uses_document_initial_flow():
    flow = Flow(id="start", description="d", questions=[], next=[])
    engine = FlowEngine(_single_flow_doc(flow))
    state = engine.initial_state()
    assert state.current_flow == "start"
    assert state.asked_questions == ()


def test_first_turn_asks_first_question():
    flow = Flow(
        id="start",
        description="d",
        questions=[
            Question(id="q1", text="What is your name?"),
        ],
        next=[],
    )
    engine = FlowEngine(_single_flow_doc(flow))
    state = engine.initial_state()
    action, new_state = engine.next_action(state, _empty_sub())
    assert action.kind == "ask"
    assert action.question == "What is your name?"
    assert action.question_id == "q1"
    assert "start.q1" in new_state.asked_questions


def test_asked_questions_accumulate():
    flow = Flow(
        id="start",
        description="d",
        questions=[
            Question(id="q1", text="Q1"),
            Question(id="q2", text="Q2"),
        ],
        next=[],
    )
    engine = FlowEngine(_single_flow_doc(flow))
    state = engine.initial_state()
    _, state = engine.next_action(state, _empty_sub())  # asks q1
    _, state = engine.next_action(state, _empty_sub())  # asks q2
    # asked_questions stores qualified "flow_id.question_id" keys
    assert "start.q1" in state.asked_questions
    assert "start.q2" in state.asked_questions


def test_skip_when_satisfied_skips_question():
    flow = Flow(
        id="start",
        description="d",
        questions=[
            Question(
                id="q1",
                text="What is your EIN?",
                skip_when=FieldSetCondition(path="ein"),
            ),
            Question(id="q2", text="What is your name?"),
        ],
        next=[],
    )
    engine = FlowEngine(_single_flow_doc(flow))
    state = engine.initial_state()
    sub = _sub_with(ein="12-3456789")  # q1's skip_when is satisfied
    action, _ = engine.next_action(state, sub)
    assert action.question_id == "q2"  # q1 skipped


def test_all_questions_exhausted_transitions_unconditionally():
    flow_a = Flow(
        id="a",
        description="d",
        questions=[Question(id="q1", text="Q1")],
        next=[FlowTransition(when=None, flow="b")],
    )
    flow_b = Flow(
        id="b",
        description="d",
        questions=[Question(id="q2", text="Q2")],
        next=[],
    )
    engine = FlowEngine(_doc_with_flows(flow_a, flow_b, initial="a"))
    # Ask q1 → state has q1 in asked_questions, still in flow a
    state = engine.initial_state()
    _, state = engine.next_action(state, _empty_sub())
    # Now all of flow_a's questions are asked → should transition to b and ask q2
    action, new_state = engine.next_action(state, _empty_sub())
    assert action.question_id == "q2"
    assert new_state.current_flow == "b"


def test_conditional_transition_picks_matching_branch():
    from accord_ai.schema import CommercialAutoDetails

    flow_policy = Flow(
        id="policy",
        description="d",
        questions=[Question(id="q1", text="Q1")],
        next=[
            FlowTransition(
                when=FieldEqualsCondition(path="lob_details.lob", value="commercial_auto"),
                flow="ca",
            ),
            FlowTransition(when=None, flow="other"),
        ],
    )
    flow_ca = Flow(id="ca", description="d", questions=[], next=[])
    flow_other = Flow(id="other", description="d", questions=[], next=[])
    engine = FlowEngine(
        _doc_with_flows(flow_policy, flow_ca, flow_other, initial="policy")
    )
    # Ask q1 first
    state = engine.initial_state()
    _, state = engine.next_action(state, _empty_sub())
    # Now all questions asked; submission has CA lob → should go to "ca"
    sub = CustomerSubmission(lob_details=CommercialAutoDetails())
    action, new_state = engine.next_action(state, sub)
    assert new_state.current_flow == "ca"
    assert action.kind == "finalize"  # ca flow has no questions


def test_conditional_transition_falls_through_to_default():
    flow_a = Flow(
        id="a",
        description="d",
        questions=[Question(id="q1", text="Q1")],
        next=[
            FlowTransition(
                when=FieldEqualsCondition(path="ein", value="MATCH"),
                flow="match",
            ),
            FlowTransition(when=None, flow="default"),
        ],
    )
    flow_match = Flow(id="match", description="d", questions=[], next=[])
    flow_default = Flow(id="default", description="d", questions=[], next=[])
    engine = FlowEngine(
        _doc_with_flows(flow_a, flow_match, flow_default, initial="a")
    )
    state = engine.initial_state()
    _, state = engine.next_action(state, _empty_sub())  # ask q1
    # ein != "MATCH" → should fall through to default
    action, new_state = engine.next_action(state, _empty_sub())
    assert new_state.current_flow == "default"
    assert action.kind == "finalize"


def test_terminal_flow_returns_finalize():
    flow = Flow(id="done", description="d", questions=[], next=[])
    engine = FlowEngine(_single_flow_doc(flow))
    state = engine.initial_state()
    action, _ = engine.next_action(state, _empty_sub())
    assert action.kind == "finalize"
    assert action.flow_id == "done"


def test_cycle_detection_raises():
    flow_a = Flow(id="a", description="d", questions=[], next=[FlowTransition(flow="b")])
    flow_b = Flow(id="b", description="d", questions=[], next=[FlowTransition(flow="a")])
    engine = FlowEngine(_doc_with_flows(flow_a, flow_b, initial="a"))
    state = engine.initial_state()
    with pytest.raises(FlowCycleError, match="cycle"):
        engine.next_action(state, _empty_sub())


def test_missing_required_reported_when_fields_absent():
    flow = Flow(
        id="start",
        description="d",
        required_fields=["ein", "business_name"],
        questions=[Question(id="q1", text="Q1")],
        next=[],
    )
    engine = FlowEngine(_single_flow_doc(flow))
    state = engine.initial_state()
    action, _ = engine.next_action(state, _empty_sub())
    assert "ein" in action.missing_required
    assert "business_name" in action.missing_required


def test_required_fields_empty_when_all_satisfied():
    flow = Flow(
        id="start",
        description="d",
        required_fields=["ein"],
        questions=[Question(id="q1", text="Q1")],
        next=[],
    )
    engine = FlowEngine(_single_flow_doc(flow))
    state = engine.initial_state()
    sub = _sub_with(ein="12-3456789")
    action, _ = engine.next_action(state, sub)
    assert action.missing_required == ()


def test_flow_state_json_roundtrip():
    original = FlowState(
        current_flow="business_identity",
        asked_questions=("ask_ein", "ask_entity_type"),
    )
    restored = FlowState.from_json(original.to_json())
    assert restored == original


def test_any_of_condition_true_when_any_matches():
    cond = AnyCondition(
        conditions=[
            FieldSetCondition(path="ein"),
            FieldSetCondition(path="business_name"),
        ]
    )
    sub = _sub_with(business_name="Acme")  # ein absent, business_name present
    assert evaluate_condition(cond, sub) is True


def test_any_of_condition_false_when_none_match():
    cond = AnyCondition(
        conditions=[
            FieldSetCondition(path="ein"),
            FieldSetCondition(path="business_name"),
        ]
    )
    assert evaluate_condition(cond, _empty_sub()) is False


# ---------------------------------------------------------------------------
# last_asked_question helper (Phase 3.3) — 4 tests
# ---------------------------------------------------------------------------


def test_last_asked_question_returns_most_recent():
    flow = Flow(
        id="biz",
        description="d",
        questions=[
            Question(id="q1", text="Q1", expected_fields=["business_name"]),
            Question(id="q2", text="Q2", expected_fields=["ein"]),
        ],
        next=[],
    )
    engine = FlowEngine(_single_flow_doc(flow))
    state = FlowState(current_flow="biz", asked_questions=("biz.q1", "biz.q2"))
    result = engine.last_asked_question(state)
    assert result is not None
    flow_obj, question = result
    assert flow_obj.id == "biz"
    assert question.id == "q2"
    assert question.expected_fields == ["ein"]


def test_last_asked_question_none_when_never_asked():
    flow = Flow(id="start", description="d", questions=[], next=[])
    engine = FlowEngine(_single_flow_doc(flow))
    state = FlowState(current_flow="start", asked_questions=())
    assert engine.last_asked_question(state) is None


def test_last_asked_question_handles_malformed_state():
    flow = Flow(id="start", description="d", questions=[], next=[])
    engine = FlowEngine(_single_flow_doc(flow))
    # Unqualified (no dot) key → malformed
    state = FlowState(current_flow="start", asked_questions=("noDotHere",))
    assert engine.last_asked_question(state) is None


def test_last_asked_question_after_cross_flow_transition():
    flow_a = Flow(
        id="a",
        description="d",
        questions=[Question(id="qa", text="QA", expected_fields=["email"])],
        next=[FlowTransition(flow="b")],
    )
    flow_b = Flow(
        id="b",
        description="d",
        questions=[Question(id="qb", text="QB", expected_fields=["phone"])],
        next=[],
    )
    engine = FlowEngine(_doc_with_flows(flow_a, flow_b, initial="a"))
    # State after transitioning to b and asking qb
    state = FlowState(current_flow="b", asked_questions=("a.qa", "b.qb"))
    result = engine.last_asked_question(state)
    assert result is not None
    flow_obj, question = result
    assert flow_obj.id == "b"
    assert question.id == "qb"
    assert question.expected_fields == ["phone"]
