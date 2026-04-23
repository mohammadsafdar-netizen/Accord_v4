"""9.e — ConversationController full-flow tests. All LLM calls are FakeEngine-driven."""
from dataclasses import FrozenInstanceError

import pytest

from accord_ai.conversation.controller import ConversationController, TurnResult
from accord_ai.conversation.responder import Responder
from accord_ai.extraction.extractor import Extractor
from accord_ai.harness.judge import JudgeVerdict, SchemaJudge
from accord_ai.harness.manager import HarnessManager
from accord_ai.harness.refiner import Refiner
from accord_ai.llm.fake_engine import FakeEngine
from accord_ai.schema import CustomerSubmission
from tests._fixtures import valid_ca, valid_ca_dict


def _build_controller(
    store,
    *,
    extractor_responses=None,
    refiner_responses=None,
    responder_responses=None,
    max_refines=0,
):
    """Wire a Controller with three separate FakeEngines (one per LLM consumer)."""
    extractor_engine = FakeEngine(extractor_responses or [])
    refiner_engine = FakeEngine(refiner_responses or [])
    responder_engine = FakeEngine(responder_responses or [])
    return (
        ConversationController(
            store=store,
            extractor=Extractor(extractor_engine),
            harness=HarnessManager(
                judge=SchemaJudge(),
                refiner=Refiner(refiner_engine),
                max_refines=max_refines,
            ),
            responder=Responder(responder_engine),
        ),
        extractor_engine, refiner_engine, responder_engine,
    )


# --- TurnResult shape ---

def test_turn_result_is_frozen():
    r = TurnResult(
        submission=CustomerSubmission(),
        verdict=JudgeVerdict(passed=True),
        assistant_message="hi",
        is_complete=True,
    )
    with pytest.raises(FrozenInstanceError):
        r.is_complete = False


# --- Happy path, no refinement needed ---

@pytest.mark.asyncio
async def test_extract_apply_judge_pass_respond(store):
    sid = store.create_session()
    # Extractor returns a fully-valid submission so the expanded judge
    # passes after apply_diff — covers the no-refinement branch.
    controller, ext, _ref, resp = _build_controller(
        store,
        extractor_responses=[valid_ca_dict()],
        responder_responses=["Great — we have Acme. Ready to finalize."],
        max_refines=0,
    )

    result = await controller.process_turn(session_id=sid, user_message="we are Acme")

    assert result.submission.business_name == "Acme Trucking"
    assert result.verdict.passed is True
    assert result.is_complete is True
    assert "Acme" in result.assistant_message
    # The submission persisted to the store
    assert (
        store.get_session(sid).submission.business_name == "Acme Trucking"
    )


# --- Happy path with refinement ---

@pytest.mark.asyncio
async def test_extract_judge_fail_refine_judge_pass(store):
    """Extraction produces partial state; judge fails; refiner fixes; re-judge passes."""
    sid = store.create_session()
    # Extractor writes a partial (bad dates only). Judge fails.
    # Refiner returns a fully-valid submission with corrected dates.
    refined = valid_ca_dict() | {
        "policy_dates": {
            "effective_date": "2026-05-01",
            "expiration_date": "2027-05-01",
        },
    }
    controller, _, _, _ = _build_controller(
        store,
        extractor_responses=[{"policy_dates": {"effective_date": "2027-05-01",
                                                "expiration_date": "2026-05-01"}}],
        refiner_responses=[refined],
        responder_responses=["All set — dates corrected."],
        max_refines=1,
    )

    result = await controller.process_turn(
        session_id=sid, user_message="policy 2026 to 2027 for Acme",
    )

    assert result.verdict.passed is True
    persisted = store.get_session(sid).submission
    assert persisted.business_name == "Acme Trucking"
    from datetime import date
    assert persisted.policy_dates.effective_date == date(2026, 5, 1)


# --- Session errors propagate ---

@pytest.mark.asyncio
async def test_missing_session_raises_key_error(store):
    controller, *_ = _build_controller(store)
    with pytest.raises(KeyError):
        await controller.process_turn(session_id="nope", user_message="hi")


@pytest.mark.asyncio
async def test_finalized_session_raises_key_error(store):
    sid = store.create_session()
    store.finalize_session(sid)
    controller, *_ = _build_controller(store)
    with pytest.raises(KeyError):
        await controller.process_turn(session_id=sid, user_message="late")


@pytest.mark.asyncio
async def test_wrong_tenant_raises_key_error(store):
    sid = store.create_session(tenant="acme")
    controller, *_ = _build_controller(store)
    with pytest.raises(KeyError):
        await controller.process_turn(
            session_id=sid, user_message="hi", tenant="globex",
        )


# --- Extractor degradation ---

@pytest.mark.asyncio
async def test_extractor_output_error_degrades_gracefully(store):
    """Non-JSON from extractor → skip apply, continue with current state."""
    sid = store.create_session()
    store.update_submission(sid, CustomerSubmission(business_name="Acme"))
    controller, _, _, _ = _build_controller(
        store,
        extractor_responses=["this is not JSON"],
        responder_responses=["Got it. What else can you tell me?"],
    )

    result = await controller.process_turn(session_id=sid, user_message="garbage")

    # Current state preserved
    assert result.submission.business_name == "Acme"
    assert store.get_session(sid).submission.business_name == "Acme"
    # Assistant message still stored
    messages = store.get_messages(sid)
    assert messages[-1].role == "assistant"


# ---------------------------------------------------------------------------
# Flow engine integration (Phase 3.2) — 4 tests
# ---------------------------------------------------------------------------


def _make_minimal_doc() -> "FlowsDocument":
    from accord_ai.conversation.flow_schema import (
        Flow,
        FlowTransition,
        FlowsDocument,
        Question,
    )
    return FlowsDocument(
        version="1",
        initial_flow="greet",
        flows=[
            Flow(
                id="greet",
                description="greeting",
                questions=[Question(id="welcome", text="Welcome! What's your business?")],
                next=[FlowTransition(flow="done")],
            ),
            Flow(id="done", description="terminal", questions=[], next=[]),
        ],
    )


def _build_controller_with_engine(store, *, doc=None, **kwargs):
    from accord_ai.conversation.flow_engine import FlowEngine
    engine_doc = doc or _make_minimal_doc()
    flow_engine = FlowEngine(engine_doc)
    ctrl, ext, ref, resp = _build_controller(store, **kwargs)
    from accord_ai.conversation.controller import ConversationController
    from accord_ai.conversation.responder import Responder
    from accord_ai.extraction.extractor import Extractor
    from accord_ai.harness.manager import HarnessManager
    from accord_ai.harness.judge import SchemaJudge
    from accord_ai.harness.refiner import Refiner
    from accord_ai.llm.fake_engine import FakeEngine
    extractor_engine = FakeEngine(kwargs.get("extractor_responses") or [])
    responder_engine = FakeEngine(kwargs.get("responder_responses") or [])
    return (
        ConversationController(
            store=store,
            extractor=Extractor(extractor_engine),
            harness=HarnessManager(judge=SchemaJudge(), refiner=Refiner(FakeEngine([])), max_refines=0),
            responder=Responder(responder_engine),
            flow_engine=flow_engine,
        ),
    )


@pytest.mark.asyncio
async def test_flow_engine_writes_flow_state_to_store(store):
    """After first turn with engine, flow_state_json is persisted to the store."""
    sid = store.create_session()
    (controller,) = _build_controller_with_engine(
        store,
        extractor_responses=["{}"],
        responder_responses=["Welcome! What's your business?"],
    )
    await controller.process_turn(session_id=sid, user_message="hello")

    session = store.get_session(sid)
    assert session.flow_state_json is not None
    from accord_ai.conversation.flow_engine import FlowState
    state = FlowState.from_json(session.flow_state_json)
    assert state.current_flow == "greet"
    assert "greet.welcome" in state.asked_questions


@pytest.mark.asyncio
async def test_flow_engine_none_does_not_write_flow_state(store):
    """Without flow engine, flow_state_json stays None."""
    sid = store.create_session()
    controller, _, _, _ = _build_controller(
        store,
        extractor_responses=["{}"],
        responder_responses=["Hi there!"],
    )
    await controller.process_turn(session_id=sid, user_message="hello")

    session = store.get_session(sid)
    assert session.flow_state_json is None


@pytest.mark.asyncio
async def test_flow_engine_state_advances_between_turns(store):
    """Second turn picks up where the first left off — state advances."""
    from accord_ai.conversation.flow_schema import (
        Flow, FlowsDocument, FlowTransition, Question,
    )
    doc = FlowsDocument(
        version="1",
        initial_flow="a",
        flows=[
            Flow(
                id="a",
                description="d",
                questions=[
                    Question(id="q1", text="Q1?"),
                    Question(id="q2", text="Q2?"),
                ],
                next=[FlowTransition(flow="done")],
            ),
            Flow(id="done", description="d", questions=[], next=[]),
        ],
    )
    sid = store.create_session()
    (controller,) = _build_controller_with_engine(
        store,
        doc=doc,
        extractor_responses=["{}"],
        responder_responses=["Q1?", "Q2?"],
    )
    # Recreate with fresh FakeEngine queues for two turns
    from accord_ai.conversation.controller import ConversationController
    from accord_ai.conversation.flow_engine import FlowEngine, FlowState
    from accord_ai.conversation.responder import Responder
    from accord_ai.extraction.extractor import Extractor
    from accord_ai.harness.judge import SchemaJudge
    from accord_ai.harness.manager import HarnessManager
    from accord_ai.harness.refiner import Refiner
    from accord_ai.llm.fake_engine import FakeEngine

    ctrl = ConversationController(
        store=store,
        extractor=Extractor(FakeEngine(["{}", "{}"])),
        harness=HarnessManager(judge=SchemaJudge(), refiner=Refiner(FakeEngine([])), max_refines=0),
        responder=Responder(FakeEngine(["Q1?", "Q2?"])),
        flow_engine=FlowEngine(doc),
    )

    await ctrl.process_turn(session_id=sid, user_message="turn 1")
    state_after_1 = FlowState.from_json(store.get_session(sid).flow_state_json)
    assert "a.q1" in state_after_1.asked_questions
    assert "a.q2" not in state_after_1.asked_questions

    await ctrl.process_turn(session_id=sid, user_message="turn 2")
    state_after_2 = FlowState.from_json(store.get_session(sid).flow_state_json)
    assert "a.q2" in state_after_2.asked_questions


@pytest.mark.asyncio
async def test_flow_engine_finalize_action_does_not_update_state(store):
    """When engine says finalize (terminal flow), flow_state_json is not mutated."""
    from accord_ai.conversation.flow_schema import (
        Flow, FlowsDocument, FlowTransition, Question,
    )
    # Flow with one question already asked → transition to terminal → finalize
    doc = FlowsDocument(
        version="1",
        initial_flow="only",
        flows=[
            Flow(id="only", description="d", questions=[], next=[]),
        ],
    )
    sid = store.create_session()
    from accord_ai.conversation.controller import ConversationController
    from accord_ai.conversation.flow_engine import FlowEngine
    from accord_ai.conversation.responder import Responder
    from accord_ai.extraction.extractor import Extractor
    from accord_ai.harness.judge import SchemaJudge
    from accord_ai.harness.manager import HarnessManager
    from accord_ai.harness.refiner import Refiner
    from accord_ai.llm.fake_engine import FakeEngine

    ctrl = ConversationController(
        store=store,
        extractor=Extractor(FakeEngine(["{}"])),
        harness=HarnessManager(judge=SchemaJudge(), refiner=Refiner(FakeEngine([])), max_refines=0),
        responder=Responder(FakeEngine(["All done!"])),
        flow_engine=FlowEngine(doc),
    )
    # Initial state has no flow_state_json
    assert store.get_session(sid).flow_state_json is None
    await ctrl.process_turn(session_id=sid, user_message="hi")
    # Engine returned finalize (no questions in "only" flow) → state not written
    assert store.get_session(sid).flow_state_json is None


@pytest.mark.asyncio
async def test_lob_transition_attempt_is_rejected_gracefully(store):
    """Extractor tries to switch LOB → apply_diff raises → keep pre-diff state."""
    sid = store.create_session()
    store.update_submission(sid, CustomerSubmission(
        business_name="Acme",
        lob_details={"lob": "commercial_auto", "drivers": [{"first_name": "Alice"}]},
    ))
    controller, _, _, _ = _build_controller(
        store,
        extractor_responses=[{"lob_details": {"lob": "general_liability",
                                               "employee_count": 5}}],
        responder_responses=["Noted — continuing with commercial auto."],
    )

    result = await controller.process_turn(
        session_id=sid, user_message="actually we're GL now",
    )

    # LOB unchanged
    assert result.submission.lob_details.lob == "commercial_auto"
    assert result.submission.lob_details.drivers[0].first_name == "Alice"


# --- Engine exceptions propagate ---

@pytest.mark.asyncio
async def test_extractor_engine_exception_propagates(store):
    sid = store.create_session()

    class _Boom:
        async def generate(self, messages, *, temperature=0.0, max_tokens=4096, json_schema=None):
            raise RuntimeError("extractor down")

    controller = ConversationController(
        store=store,
        extractor=Extractor(_Boom()),
        harness=HarnessManager(SchemaJudge(), Refiner(FakeEngine()), max_refines=0),
        responder=Responder(FakeEngine()),
    )
    with pytest.raises(RuntimeError, match="extractor down"):
        await controller.process_turn(session_id=sid, user_message="hi")


@pytest.mark.asyncio
async def test_responder_engine_exception_propagates(store):
    sid = store.create_session()
    store.update_submission(sid, CustomerSubmission(business_name="Acme"))

    class _Boom:
        async def generate(self, messages, *, temperature=0.0, max_tokens=4096, json_schema=None):
            raise RuntimeError("responder down")

    controller = ConversationController(
        store=store,
        extractor=Extractor(FakeEngine([{}])),
        harness=HarnessManager(SchemaJudge(), Refiner(FakeEngine()), max_refines=0),
        responder=Responder(_Boom()),
    )
    with pytest.raises(RuntimeError, match="responder down"):
        await controller.process_turn(session_id=sid, user_message="hi")


# --- Message persistence ---

@pytest.mark.asyncio
async def test_user_and_assistant_messages_both_persisted_in_order(store):
    sid = store.create_session()
    controller, *_ = _build_controller(
        store,
        extractor_responses=[{"business_name": "Acme"}],
        responder_responses=["Great — what's your EIN?"],
    )
    await controller.process_turn(session_id=sid, user_message="we are Acme")

    messages = store.get_messages(sid)
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[0].content == "we are Acme"
    assert messages[1].content == "Great — what's your EIN?"


@pytest.mark.asyncio
async def test_user_message_persisted_even_if_downstream_fails(store):
    """Failure after append_message still leaves user turn in the log."""
    sid = store.create_session()

    class _Boom:
        async def generate(self, messages, *, temperature=0.0, max_tokens=4096, json_schema=None):
            raise RuntimeError("extractor down")

    controller = ConversationController(
        store=store,
        extractor=Extractor(_Boom()),
        harness=HarnessManager(SchemaJudge(), Refiner(FakeEngine()), max_refines=0),
        responder=Responder(FakeEngine()),
    )
    with pytest.raises(RuntimeError):
        await controller.process_turn(session_id=sid, user_message="turn one")

    messages = store.get_messages(sid)
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content == "turn one"


# --- Multi-turn composition ---

@pytest.mark.asyncio
async def test_multi_turn_accumulates_extracted_fields(store):
    """Two turns compose — turn 1 adds business_name, turn 2 adds ein."""
    sid = store.create_session()
    controller, *_ = _build_controller(
        store,
        extractor_responses=[
            {"business_name": "Acme"},       # turn 1
            {"ein": "12-3456789"},            # turn 2
        ],
        responder_responses=[
            "Got it — what's your EIN?",     # turn 1
            "Perfect.",                       # turn 2
        ],
    )

    await controller.process_turn(session_id=sid, user_message="Acme Trucking")
    await controller.process_turn(session_id=sid, user_message="EIN 12-3456789")

    final = store.get_session(sid).submission
    assert final.business_name == "Acme"
    assert final.ein == "12-3456789"
    # 4 messages total (2 user + 2 assistant)
    assert len(store.get_messages(sid)) == 4


# --- Empty extraction is a no-op write ---

@pytest.mark.asyncio
async def test_empty_extraction_does_not_overwrite_current_state(store):
    """Extractor returns {} — current state preserved, no update_submission called."""
    sid = store.create_session()
    store.update_submission(sid, CustomerSubmission(business_name="Acme"))
    controller, *_ = _build_controller(
        store,
        extractor_responses=[{}],
        responder_responses=["Could you tell me more?"],
    )

    await controller.process_turn(session_id=sid, user_message="uh")

    assert store.get_session(sid).submission.business_name == "Acme"


# --- Symmetric: responder-failure state inconsistency (P7.1) ---

@pytest.mark.asyncio
async def test_responder_failure_leaves_submission_updated_but_no_assistant_message(store):
    """If responder crashes AFTER submission update:
      - user message: persisted (step 2)
      - submission: updated (step 4)
      - assistant message: NOT persisted (step 7 skipped)
    No data loss; just an unusual transcript. Pins the invariant."""
    sid = store.create_session()

    class _Boom:
        async def generate(self, messages, *, temperature=0.0, max_tokens=4096, json_schema=None):
            raise RuntimeError("responder down")

    controller = ConversationController(
        store=store,
        extractor=Extractor(FakeEngine([{"business_name": "Acme"}])),
        harness=HarnessManager(SchemaJudge(), Refiner(FakeEngine()), max_refines=0),
        responder=Responder(_Boom()),
    )
    with pytest.raises(RuntimeError, match="responder down"):
        await controller.process_turn(session_id=sid, user_message="we are Acme")

    # User message captured; assistant message not (step 7 never reached)
    messages = store.get_messages(sid)
    assert [m.role for m in messages] == ["user"]
    assert messages[0].content == "we are Acme"
    # Submission WAS updated (step 4 ran before responder)
    assert store.get_session(sid).submission.business_name == "Acme"


# --- Context injection smoke test ---

# --- P9.1: optimistic concurrency on refinement ---

@pytest.mark.asyncio
async def test_controller_propagates_concurrency_error_on_refinement(store):
    """If update_submission raises ConcurrencyError during refinement,
    Controller logs + re-raises. Caller (API) decides retry / 409."""
    from accord_ai.core.store import ConcurrencyError

    sid = store.create_session()

    # Build controller with a refinement-triggering setup: extractor returns
    # bad dates (judge fails), refiner produces corrected submission.
    main = FakeEngine([
        {"policy_dates": {
            "effective_date": "2027-05-01",
            "expiration_date": "2026-05-01",
        }},
        "ok",
    ])
    refiner_engine = FakeEngine([{
        "business_name": "Acme",
        "policy_dates": {
            "effective_date": "2026-05-01",
            "expiration_date": "2027-05-01",
        },
    }])

    controller = ConversationController(
        store=store,
        extractor=Extractor(main),
        harness=HarnessManager(
            SchemaJudge(), Refiner(refiner_engine), max_refines=1,
        ),
        responder=Responder(main),
    )

    # Simulate concurrent writer: patch update_submission to raise when the
    # optimistic check kwarg is present (refinement path).
    original_update = store.update_submission

    def racing_update(session_id, submission, *, tenant=None, expected_updated_at=None):
        if expected_updated_at is not None:
            raise ConcurrencyError("simulated concurrent write")
        return original_update(session_id, submission, tenant=tenant)

    store.update_submission = racing_update  # type: ignore[assignment]

    with pytest.raises(ConcurrencyError):
        await controller.process_turn(
            session_id=sid, user_message="bad dates",
        )


# ---------------------------------------------------------------------------
# Phase 3.3 — ExtractionContext controller integration (4 tests)
# ---------------------------------------------------------------------------


def _build_controller_with_context(store, *, doc=None, extraction_context_enabled=True, **kwargs):
    """Controller wired with flow engine + extraction_context_enabled flag."""
    from accord_ai.conversation.controller import ConversationController
    from accord_ai.conversation.flow_engine import FlowEngine
    from accord_ai.conversation.responder import Responder
    from accord_ai.extraction.extractor import Extractor
    from accord_ai.harness.judge import SchemaJudge
    from accord_ai.harness.manager import HarnessManager
    from accord_ai.harness.refiner import Refiner
    from accord_ai.llm.fake_engine import FakeEngine

    flow_engine = FlowEngine(doc or _make_minimal_doc())
    return ConversationController(
        store=store,
        extractor=Extractor(FakeEngine(kwargs.get("extractor_responses") or [])),
        harness=HarnessManager(judge=SchemaJudge(), refiner=Refiner(FakeEngine([])), max_refines=0),
        responder=Responder(FakeEngine(kwargs.get("responder_responses") or [])),
        flow_engine=flow_engine,
        extraction_context_enabled=extraction_context_enabled,
    )


@pytest.mark.asyncio
async def test_controller_passes_context_on_second_turn(store):
    """Second turn: context built from first turn's persisted flow state."""
    from accord_ai.conversation.flow_schema import (
        Flow, FlowsDocument, FlowTransition, Question,
    )
    from accord_ai.conversation.flow_engine import FlowEngine
    from accord_ai.conversation.controller import ConversationController
    from accord_ai.conversation.responder import Responder
    from accord_ai.extraction.extractor import Extractor
    from accord_ai.harness.judge import SchemaJudge
    from accord_ai.harness.manager import HarnessManager
    from accord_ai.harness.refiner import Refiner
    from accord_ai.llm.fake_engine import FakeEngine

    doc = FlowsDocument(
        version="1",
        initial_flow="biz",
        flows=[
            Flow(
                id="biz",
                description="d",
                questions=[
                    Question(id="q1", text="What is your EIN?", expected_fields=["ein"]),
                    Question(id="q2", text="What is your name?", expected_fields=["business_name"]),
                ],
                next=[FlowTransition(flow="done")],
            ),
            Flow(id="done", description="d", questions=[], next=[]),
        ],
    )
    sid = store.create_session()
    extractor_engine = FakeEngine(["{}", "{}"])
    responder_engine = FakeEngine(["What is your EIN?", "What is your name?"])

    ctrl = ConversationController(
        store=store,
        extractor=Extractor(extractor_engine),
        harness=HarnessManager(judge=SchemaJudge(), refiner=Refiner(FakeEngine([])), max_refines=0),
        responder=Responder(responder_engine),
        flow_engine=FlowEngine(doc),
        extraction_context_enabled=True,
    )

    await ctrl.process_turn(session_id=sid, user_message="turn 1")
    # Second turn — the extractor should have received context from q1
    await ctrl.process_turn(session_id=sid, user_message="turn 2")

    second_user_content = extractor_engine.last_messages[1]["content"]
    assert "FLOW: biz" in second_user_content
    assert "FOCUS FIELDS: ein" in second_user_content
    assert "QUESTION ASKED: What is your EIN?" in second_user_content


@pytest.mark.asyncio
async def test_controller_empty_context_when_flag_off(store):
    """extraction_context_enabled=False → no FLOW/FOCUS/QUESTION in extraction prompt."""
    from accord_ai.conversation.flow_schema import (
        Flow, FlowsDocument, FlowTransition, Question,
    )
    from accord_ai.conversation.flow_engine import FlowEngine
    from accord_ai.conversation.controller import ConversationController
    from accord_ai.conversation.responder import Responder
    from accord_ai.extraction.extractor import Extractor
    from accord_ai.harness.judge import SchemaJudge
    from accord_ai.harness.manager import HarnessManager
    from accord_ai.harness.refiner import Refiner
    from accord_ai.llm.fake_engine import FakeEngine

    doc = FlowsDocument(
        version="1",
        initial_flow="biz",
        flows=[
            Flow(
                id="biz",
                description="d",
                questions=[
                    Question(id="q1", text="What is your EIN?", expected_fields=["ein"]),
                    Question(id="q2", text="Q2?", expected_fields=["business_name"]),
                ],
                next=[FlowTransition(flow="done")],
            ),
            Flow(id="done", description="d", questions=[], next=[]),
        ],
    )
    sid = store.create_session()
    extractor_engine = FakeEngine(["{}", "{}"])
    responder_engine = FakeEngine(["Q1?", "Q2?"])

    ctrl = ConversationController(
        store=store,
        extractor=Extractor(extractor_engine),
        harness=HarnessManager(judge=SchemaJudge(), refiner=Refiner(FakeEngine([])), max_refines=0),
        responder=Responder(responder_engine),
        flow_engine=FlowEngine(doc),
        extraction_context_enabled=False,
    )

    await ctrl.process_turn(session_id=sid, user_message="turn 1")
    await ctrl.process_turn(session_id=sid, user_message="turn 2")

    second_user_content = extractor_engine.last_messages[1]["content"]
    assert "FLOW:" not in second_user_content
    assert "FOCUS FIELDS:" not in second_user_content


@pytest.mark.asyncio
async def test_controller_empty_context_when_no_flow_engine(store):
    """No flow engine → extraction context always empty, no FLOW header."""
    from accord_ai.llm.fake_engine import FakeEngine

    sid = store.create_session()
    extractor_engine = FakeEngine(["{}", "{}"])
    controller, _, _, _ = _build_controller(
        store,
        extractor_responses=["{}", "{}"],
        responder_responses=["R1", "R2"],
    )
    # Re-build with separate extractor_engine so we can inspect calls
    from accord_ai.conversation.controller import ConversationController
    from accord_ai.conversation.responder import Responder
    from accord_ai.extraction.extractor import Extractor
    from accord_ai.harness.judge import SchemaJudge
    from accord_ai.harness.manager import HarnessManager
    from accord_ai.harness.refiner import Refiner

    ctrl = ConversationController(
        store=store,
        extractor=Extractor(extractor_engine),
        harness=HarnessManager(judge=SchemaJudge(), refiner=Refiner(FakeEngine([])), max_refines=0),
        responder=Responder(FakeEngine(["R1", "R2"])),
        flow_engine=None,
        extraction_context_enabled=True,
    )
    sid2 = store.create_session()
    await ctrl.process_turn(session_id=sid2, user_message="turn 1")
    await ctrl.process_turn(session_id=sid2, user_message="turn 2")

    second_user_content = extractor_engine.last_messages[1]["content"]
    assert "FLOW:" not in second_user_content


@pytest.mark.asyncio
async def test_controller_handles_asked_question_no_longer_in_flows_yaml(store):
    """If flow_state_json references a removed question, context is EMPTY (graceful)."""
    from accord_ai.conversation.flow_schema import (
        Flow, FlowsDocument, Question,
    )
    from accord_ai.conversation.flow_engine import FlowEngine, FlowState
    from accord_ai.conversation.controller import ConversationController
    from accord_ai.conversation.responder import Responder
    from accord_ai.extraction.extractor import Extractor
    from accord_ai.harness.judge import SchemaJudge
    from accord_ai.harness.manager import HarnessManager
    from accord_ai.harness.refiner import Refiner
    from accord_ai.llm.fake_engine import FakeEngine

    # Doc with only q2 — q1 was "removed"
    doc = FlowsDocument(
        version="1",
        initial_flow="biz",
        flows=[
            Flow(
                id="biz",
                description="d",
                questions=[Question(id="q2", text="Q2?", expected_fields=["ein"])],
                next=[],
            ),
        ],
    )
    sid = store.create_session()
    # Manually set stale flow_state_json that references removed q1
    stale_state = FlowState(current_flow="biz", asked_questions=("biz.q1",))
    store.update_flow_state(sid, stale_state.to_json())

    extractor_engine = FakeEngine(["{}"])
    ctrl = ConversationController(
        store=store,
        extractor=Extractor(extractor_engine),
        harness=HarnessManager(judge=SchemaJudge(), refiner=Refiner(FakeEngine([])), max_refines=0),
        responder=Responder(FakeEngine(["ok"])),
        flow_engine=FlowEngine(doc),
        extraction_context_enabled=True,
    )

    # Should not raise — falls back to EMPTY_CONTEXT
    await ctrl.process_turn(session_id=sid, user_message="some info")
    user_content = extractor_engine.last_messages[1]["content"]
    assert "FLOW:" not in user_content


@pytest.mark.asyncio
async def test_process_turn_injects_session_and_tenant_into_context(store):
    """Defensive set_context — CLI / batch callers get log attribution without
    depending on a Phase 9 middleware."""
    from accord_ai.request_context import (
        clear_context,
        get_session_id,
        get_tenant,
    )

    clear_context()
    sid = store.create_session(tenant="acme")
    controller, *_ = _build_controller(
        store,
        extractor_responses=[{"business_name": "Acme"}],
        responder_responses=["ok"],
    )
    await controller.process_turn(
        session_id=sid, user_message="we are Acme", tenant="acme",
    )
    # Context persists in the test's task after process_turn returns
    assert get_session_id() == sid
    assert get_tenant() == "acme"
    clear_context()
