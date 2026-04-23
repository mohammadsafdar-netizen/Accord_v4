"""5.b — Refiner tests. FakeEngine-driven, no network."""
import json
from datetime import date

import pytest

from accord_ai.harness.judge import JudgeVerdict, SchemaJudge
from accord_ai.harness.refiner import Refiner, RefinerOutputError
from accord_ai.llm.fake_engine import FakeEngine
from accord_ai.schema import CustomerSubmission, PolicyDates


@pytest.mark.asyncio
async def test_refine_sends_guided_json_customer_submission_schema():
    """The refiner emits a full CustomerSubmission; the schema must be
    passed to vLLM so structured-output guarantees a parseable result."""
    engine = FakeEngine([{"business_name": "Acme"}])
    refiner = Refiner(engine)
    await refiner.refine(
        original_user_message="msg",
        current_submission=CustomerSubmission(),
        verdict=SchemaJudge().evaluate(CustomerSubmission()),
    )
    schema = engine.last_call.json_schema
    assert schema is not None
    assert "business_name" in schema.get("properties", {})
    assert "lob_details" in schema.get("properties", {})


# --- Happy path ---

@pytest.mark.asyncio
async def test_refine_returns_parsed_submission():
    corrected = {
        "business_name": "Acme Trucking",
        "lob_details": {"lob": "commercial_auto", "drivers": [{"first_name": "Alice"}]},
    }
    engine = FakeEngine([corrected])
    refiner = Refiner(engine)

    current = CustomerSubmission()   # empty — judge will flag business_name
    verdict = SchemaJudge().evaluate(current)

    result = await refiner.refine(
        original_user_message="We're Acme Trucking, our driver is Alice.",
        current_submission=current,
        verdict=verdict,
    )
    assert isinstance(result, CustomerSubmission)
    assert result.business_name == "Acme Trucking"
    assert result.lob_details.drivers[0].first_name == "Alice"


# --- Prompt composition ---

@pytest.mark.asyncio
async def test_refine_passes_system_and_user_messages():
    engine = FakeEngine([{"business_name": "Acme"}])
    refiner = Refiner(engine)

    verdict = SchemaJudge().evaluate(CustomerSubmission())
    await refiner.refine(
        original_user_message="We're Acme.",
        current_submission=CustomerSubmission(),
        verdict=verdict,
    )

    messages = engine.last_messages
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


@pytest.mark.asyncio
async def test_refine_prompt_includes_original_user_message():
    engine = FakeEngine([{"business_name": "Acme"}])
    refiner = Refiner(engine)

    await refiner.refine(
        original_user_message="distinctive-input-marker-42",
        current_submission=CustomerSubmission(),
        verdict=SchemaJudge().evaluate(CustomerSubmission()),
    )
    user_content = engine.last_messages[1]["content"]
    assert "distinctive-input-marker-42" in user_content


@pytest.mark.asyncio
async def test_refine_prompt_includes_verdict_reasons():
    engine = FakeEngine([{"business_name": "Acme"}])
    refiner = Refiner(engine)

    verdict = SchemaJudge().evaluate(CustomerSubmission())   # fails business_name rule
    await refiner.refine(
        original_user_message="msg",
        current_submission=CustomerSubmission(),
        verdict=verdict,
    )
    user_content = engine.last_messages[1]["content"]
    assert "business_name is required" in user_content


@pytest.mark.asyncio
async def test_refine_prompt_includes_failed_paths():
    engine = FakeEngine([{"business_name": "Acme"}])
    refiner = Refiner(engine)

    verdict = SchemaJudge().evaluate(CustomerSubmission())
    await refiner.refine(
        original_user_message="msg",
        current_submission=CustomerSubmission(),
        verdict=verdict,
    )
    user_content = engine.last_messages[1]["content"]
    assert "business_name" in user_content


@pytest.mark.asyncio
async def test_refine_prompt_includes_current_submission_json():
    engine = FakeEngine([{"business_name": "Corrected"}])
    refiner = Refiner(engine)

    current = CustomerSubmission(business_name="Typo-Corp")
    verdict = JudgeVerdict(passed=False, reasons=("typo",), failed_paths=("business_name",))
    await refiner.refine(
        original_user_message="msg",
        current_submission=current,
        verdict=verdict,
    )
    user_content = engine.last_messages[1]["content"]
    assert "Typo-Corp" in user_content


@pytest.mark.asyncio
async def test_refine_handles_empty_reasons_and_paths_gracefully():
    """Caller passed a passing verdict — refiner still runs, prompt says '(none)'."""
    engine = FakeEngine([{"business_name": "Acme"}])
    refiner = Refiner(engine)
    verdict = JudgeVerdict(passed=True)
    await refiner.refine(
        original_user_message="msg",
        current_submission=CustomerSubmission(business_name="Acme"),
        verdict=verdict,
    )
    user_content = engine.last_messages[1]["content"]
    assert "(none)" in user_content


# --- Error handling ---

@pytest.mark.asyncio
async def test_refine_raises_on_non_json_output():
    engine = FakeEngine(["this is not JSON"])
    refiner = Refiner(engine)

    with pytest.raises(RefinerOutputError, match="non-JSON"):
        await refiner.refine(
            original_user_message="msg",
            current_submission=CustomerSubmission(),
            verdict=SchemaJudge().evaluate(CustomerSubmission()),
        )


@pytest.mark.asyncio
async def test_refine_raises_on_schema_invalid_json():
    """Valid JSON, but doesn't parse as CustomerSubmission.

    Schema is extra='ignore' so unknown keys are dropped silently. Use
    a primitive-type mismatch instead — lob_details must be an object
    (discriminated union), a bare string violates the shape."""
    engine = FakeEngine([{"lob_details": "commercial_auto"}])
    refiner = Refiner(engine)

    with pytest.raises(RefinerOutputError, match="schema validation"):
        await refiner.refine(
            original_user_message="msg",
            current_submission=CustomerSubmission(),
            verdict=SchemaJudge().evaluate(CustomerSubmission()),
        )


@pytest.mark.asyncio
async def test_refiner_output_error_is_valueerror():
    """Subclass of ValueError so a generic 'bad input' handler catches it."""
    assert issubclass(RefinerOutputError, ValueError)


@pytest.mark.asyncio
async def test_refine_propagates_engine_exceptions_unwrapped():
    """Engine failures (e.g., RateLimitError) propagate — caller owns retry."""
    class _BoomEngine:
        async def generate(self, messages, *, temperature=0.0, max_tokens=4096, json_schema=None):
            raise RuntimeError("provider down")

    refiner = Refiner(_BoomEngine())
    with pytest.raises(RuntimeError, match="provider down"):
        await refiner.refine(
            original_user_message="msg",
            current_submission=CustomerSubmission(),
            verdict=SchemaJudge().evaluate(CustomerSubmission()),
        )


# --- Round-trip on a realistic correction ---

@pytest.mark.asyncio
async def test_refine_accepts_markdown_fenced_json():
    """Wire-level regression guard: LLM wraps output in ```json...``` despite
    being told not to. strip_code_fences inside refine() unwraps before parsing."""
    # FakeEngine canonicalizes dict → json.dumps, so we supply the fenced text as str.
    fenced_output = '```json\n{"business_name": "Acme"}\n```'
    engine = FakeEngine([fenced_output])
    refiner = Refiner(engine)

    result = await refiner.refine(
        original_user_message="msg",
        current_submission=CustomerSubmission(),
        verdict=SchemaJudge().evaluate(CustomerSubmission()),
    )
    assert result.business_name == "Acme"


@pytest.mark.asyncio
async def test_refine_corrects_bad_date_ordering():
    """Scenario: judge flags effective > expiration. Refiner swaps them.

    Needs a fully-valid baseline so the ONLY judge failure is the date
    ordering — otherwise the expanded (v3-aligned) judge flags more
    than the refiner can reasonably be expected to fix in one pass.
    """
    from tests._fixtures import valid_ca, valid_ca_dict

    current = valid_ca().model_copy(update={
        "policy_dates": PolicyDates(
            effective_date=date(2027, 5, 1),
            expiration_date=date(2026, 5, 1),
        ),
    })
    verdict = SchemaJudge().evaluate(current)
    assert verdict.passed is False

    corrected = valid_ca_dict() | {
        "policy_dates": {
            "effective_date": "2026-05-01",
            "expiration_date": "2027-05-01",
        },
    }
    engine = FakeEngine([corrected])
    refiner = Refiner(engine)

    result = await refiner.refine(
        original_user_message="policy runs 5/1/2026 to 5/1/2027",
        current_submission=current,
        verdict=verdict,
    )
    # Judge now passes on the refined result
    assert SchemaJudge().evaluate(result).passed is True
