"""9.c — Responder tests. FakeEngine-driven."""
import pytest

from accord_ai.conversation.responder import Responder
from accord_ai.harness.judge import JudgeVerdict, SchemaJudge
from accord_ai.llm.fake_engine import FakeEngine
from accord_ai.schema import CustomerSubmission


@pytest.mark.asyncio
async def test_respond_does_not_send_guided_json():
    """Responder emits conversational text, not JSON — must NOT constrain
    output via guided_json, otherwise vLLM would force the reply to JSON
    shape and the user gets garbage."""
    engine = FakeEngine(["Hi there!"])
    await Responder(engine).respond(
        submission=CustomerSubmission(business_name="Acme"),
        verdict=JudgeVerdict(passed=True),
    )
    assert engine.last_call.json_schema is None


# --- Happy path ---

@pytest.mark.asyncio
async def test_respond_returns_string():
    engine = FakeEngine(["Great, we have your business name. What's your EIN?"])
    text = await Responder(engine).respond(
        submission=CustomerSubmission(business_name="Acme"),
        verdict=JudgeVerdict(passed=False, reasons=("ein missing",), failed_paths=("ein",)),
    )
    assert isinstance(text, str)
    assert "EIN" in text


@pytest.mark.asyncio
async def test_respond_strips_surrounding_whitespace():
    """LLM often emits trailing newlines — caller shouldn't see them."""
    engine = FakeEngine(["  \n  Got it!  \n\n  "])
    text = await Responder(engine).respond(
        submission=CustomerSubmission(),
        verdict=JudgeVerdict(passed=True),
    )
    assert text == "Got it!"


# --- Prompt composition ---

@pytest.mark.asyncio
async def test_respond_sends_system_then_user():
    engine = FakeEngine(["ok"])
    await Responder(engine).respond(
        submission=CustomerSubmission(),
        verdict=JudgeVerdict(passed=True),
    )
    assert engine.last_messages[0]["role"] == "system"
    assert engine.last_messages[1]["role"] == "user"


@pytest.mark.asyncio
async def test_respond_prompt_reflects_passing_verdict():
    engine = FakeEngine(["ok"])
    await Responder(engine).respond(
        submission=CustomerSubmission(business_name="Acme"),
        verdict=JudgeVerdict(passed=True),
    )
    content = engine.last_messages[1]["content"]
    assert "complete" in content


@pytest.mark.asyncio
async def test_respond_prompt_reflects_failing_verdict():
    engine = FakeEngine(["ok"])
    verdict = SchemaJudge().evaluate(CustomerSubmission())
    await Responder(engine).respond(
        submission=CustomerSubmission(),
        verdict=verdict,
    )
    assert "needs-info" in engine.last_messages[1]["content"]


@pytest.mark.asyncio
async def test_respond_prompt_includes_verdict_reasons():
    engine = FakeEngine(["ok"])
    verdict = SchemaJudge().evaluate(CustomerSubmission())  # fails business_name
    await Responder(engine).respond(
        submission=CustomerSubmission(),
        verdict=verdict,
    )
    assert "business_name is required" in engine.last_messages[1]["content"]


@pytest.mark.asyncio
async def test_respond_prompt_includes_failed_paths():
    engine = FakeEngine(["ok"])
    await Responder(engine).respond(
        submission=CustomerSubmission(),
        verdict=JudgeVerdict(passed=False, reasons=("x",), failed_paths=("ein",)),
    )
    assert "ein" in engine.last_messages[1]["content"]


@pytest.mark.asyncio
async def test_respond_prompt_includes_current_submission():
    engine = FakeEngine(["ok"])
    await Responder(engine).respond(
        submission=CustomerSubmission(business_name="Distinctive-Marker-42"),
        verdict=JudgeVerdict(passed=True),
    )
    assert "Distinctive-Marker-42" in engine.last_messages[1]["content"]


@pytest.mark.asyncio
async def test_respond_handles_empty_reasons_and_paths_gracefully():
    """Passing verdict has no reasons — template renders '(none)' placeholder."""
    engine = FakeEngine(["All set."])
    await Responder(engine).respond(
        submission=CustomerSubmission(business_name="Acme"),
        verdict=JudgeVerdict(passed=True),
    )
    content = engine.last_messages[1]["content"]
    # Both reasons and failed_paths blocks show (none)
    assert content.count("(none)") == 2


# --- Engine exceptions ---

@pytest.mark.asyncio
async def test_respond_propagates_engine_exceptions_unwrapped():
    class _BoomEngine:
        async def generate(self, messages, *, temperature=0.0, max_tokens=4096, json_schema=None):
            raise RuntimeError("provider down")

    with pytest.raises(RuntimeError, match="provider down"):
        await Responder(_BoomEngine()).respond(
            submission=CustomerSubmission(),
            verdict=JudgeVerdict(passed=True),
        )
