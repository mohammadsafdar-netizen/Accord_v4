"""5.c — HarnessManager tests."""
from dataclasses import FrozenInstanceError
from datetime import date

import pytest

from accord_ai.harness.judge import JudgeVerdict, SchemaJudge
from accord_ai.harness.manager import HarnessManager, ManagerResult
from accord_ai.harness.refiner import Refiner
from accord_ai.llm.fake_engine import FakeEngine
from accord_ai.schema import CustomerSubmission, PolicyDates
from tests._fixtures import valid_ca, valid_ca_dict


def _manager(engine):
    return HarnessManager(judge=SchemaJudge(), refiner=Refiner(engine))


# --- ManagerResult shape ---

def test_manager_result_is_frozen():
    r = ManagerResult(
        submission=CustomerSubmission(),
        verdict=JudgeVerdict(passed=True),
        refined=False,
    )
    with pytest.raises(FrozenInstanceError):
        r.refined = True


# --- Passing judge: refiner never called ---

@pytest.mark.asyncio
async def test_passing_submission_skips_refiner():
    """FakeEngine with empty queue — raises if called. Proves refiner didn't run."""
    engine = FakeEngine()
    manager = _manager(engine)

    # v3-aligned judge requires the full critical-field set; valid_ca()
    # is the smallest submission that passes without triggering refinement.
    sub = valid_ca()
    result = await manager.process(sub, "we're Acme")

    assert result.verdict.passed is True
    assert result.refined is False
    assert result.submission is sub


# --- Failing judge + successful refinement ---

@pytest.mark.asyncio
async def test_failing_judge_refined_to_passing():
    # Refiner must return a FULLY valid submission to satisfy the
    # expanded judge. FakeEngine queues serialize dicts to JSON for the
    # refiner to parse; valid_ca_dict() is that JSON-safe form.
    refined = valid_ca_dict() | {"business_name": "Acme Trucking"}
    engine = FakeEngine([refined])
    result = await _manager(engine).process(
        CustomerSubmission(), "we're Acme Trucking"
    )
    assert result.refined is True
    assert result.verdict.passed is True
    assert result.submission.business_name == "Acme Trucking"


@pytest.mark.asyncio
async def test_failing_judge_refined_but_still_failing():
    """Refiner returns valid JSON but new verdict still flags something.
    Not an error — legitimate 'ask user for more info' signal."""
    engine = FakeEngine([{
        "business_name": "Acme",
        "policy_dates": {
            "effective_date": "2027-05-01",
            "expiration_date": "2026-05-01",   # still bad
        },
    }])
    current = CustomerSubmission(
        policy_dates={
            "effective_date": "2027-05-01",
            "expiration_date": "2026-05-01",
        },
    )
    result = await _manager(engine).process(current, "bad dates")

    assert result.refined is True
    assert result.verdict.passed is False
    assert result.submission.business_name == "Acme"   # partial fix landed


# --- RefinerOutputError downgrades ---

@pytest.mark.asyncio
async def test_refiner_non_json_downgrades_to_unrefined():
    engine = FakeEngine(["this is not JSON"])
    sub = CustomerSubmission()
    result = await _manager(engine).process(sub, "msg")
    assert result.refined is False
    assert result.submission is sub
    assert result.verdict.passed is False


@pytest.mark.asyncio
async def test_refiner_schema_invalid_downgrades_to_unrefined():
    """Refiner output that can't parse as CustomerSubmission downgrades
    to `refined=False`. Schema is extra='ignore', so use a primitive-
    type mismatch (bare string for the discriminated-union lob_details)
    rather than an unknown key, which would now be silently dropped."""
    engine = FakeEngine([{"lob_details": "commercial_auto"}])
    sub = CustomerSubmission()
    result = await _manager(engine).process(sub, "msg")
    assert result.refined is False
    assert result.submission is sub


# --- Engine exceptions propagate ---

@pytest.mark.asyncio
async def test_engine_exception_propagates():
    class _BoomEngine:
        async def generate(self, messages, *, temperature=0.0, max_tokens=4096, json_schema=None):
            raise RuntimeError("provider down")
    manager = HarnessManager(judge=SchemaJudge(), refiner=Refiner(_BoomEngine()))
    with pytest.raises(RuntimeError, match="provider down"):
        await manager.process(CustomerSubmission(), "msg")


# --- Arguments pass through ---

@pytest.mark.asyncio
async def test_refiner_receives_message_and_verdict_intact():
    engine = FakeEngine([{"business_name": "Acme"}])
    await _manager(engine).process(CustomerSubmission(), "distinctive-marker-99")

    user_prompt = engine.last_messages[1]["content"]
    assert "distinctive-marker-99" in user_prompt
    assert "business_name" in user_prompt   # verdict reason propagated


# --- End-to-end scenarios ---

@pytest.mark.asyncio
async def test_end_to_end_empty_to_passing():
    engine = FakeEngine([valid_ca_dict()])
    result = await _manager(engine).process(CustomerSubmission(), "we're Acme")
    assert result.refined is True
    assert result.verdict.passed is True


@pytest.mark.asyncio
async def test_end_to_end_bad_date_ordering_fixed():
    # Refiner produces a fully-valid submission with dates in the right
    # order so the expanded judge passes. We keep the dates as the
    # observable assertion (preserves the original scenario intent).
    refined = valid_ca_dict() | {
        "policy_dates": {
            "effective_date": "2026-05-01",
            "expiration_date": "2027-05-01",
        },
    }
    engine = FakeEngine([refined])
    current = valid_ca().model_copy(update={
        "policy_dates": PolicyDates(
            effective_date=date(2027, 5, 1),
            expiration_date=date(2026, 5, 1),
        ),
    })
    result = await _manager(engine).process(current, "2026 to 2027")

    assert result.refined is True
    assert result.verdict.passed is True
    assert result.submission.policy_dates.effective_date == date(2026, 5, 1)
    assert result.submission.policy_dates.expiration_date == date(2027, 5, 1)


# --- Logging ---

@pytest.mark.asyncio
async def test_debug_logs_emitted_on_passing_path(tmp_path, monkeypatch):
    import logging
    from accord_ai.config import Settings
    from accord_ai.logging_config import configure_logging

    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    configure_logging(Settings())

    # valid_ca() passes the expanded judge; empty FakeEngine would raise
    # if the refiner ran, so this also verifies the skip-refiner path.
    await _manager(FakeEngine()).process(valid_ca(), "msg")
    for h in logging.getLogger("accord_ai").handlers:
        h.flush()
    content = (tmp_path / "logs" / "app.log").read_text()
    assert "initial judge" in content
    assert "passed=True" in content


@pytest.mark.asyncio
async def test_debug_logs_emitted_on_refinement_path(tmp_path, monkeypatch):
    import logging
    from accord_ai.config import Settings
    from accord_ai.logging_config import configure_logging

    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    configure_logging(Settings())

    engine = FakeEngine([{"business_name": "Acme"}])
    await _manager(engine).process(CustomerSubmission(), "msg")
    for h in logging.getLogger("accord_ai").handlers:
        h.flush()
    content = (tmp_path / "logs" / "app.log").read_text()
    assert "initial judge" in content
    assert "post-refine judge" in content


# ============================================================
# 5.d — max_refines loop behavior
# ============================================================

@pytest.mark.asyncio
async def test_max_refines_zero_skips_refinement_even_on_failing_judge():
    """max_refines=0 means 'never refine', regardless of verdict."""
    engine = FakeEngine()   # empty queue — would raise if called
    manager = HarnessManager(SchemaJudge(), Refiner(engine), max_refines=0)

    result = await manager.process(CustomerSubmission(), "msg")   # business_name missing

    assert result.verdict.passed is False
    assert result.refined is False
    assert engine.calls == []


@pytest.mark.asyncio
async def test_max_refines_loop_stops_early_on_pass():
    """max_refines=3 but first refine passes → only 1 engine call."""
    engine = FakeEngine([
        valid_ca_dict(),                      # first refine passes judge
        # These never get consumed
        {"business_name": "should-not-reach"},
        {"business_name": "nor-this"},
    ])
    manager = HarnessManager(SchemaJudge(), Refiner(engine), max_refines=3)
    result = await manager.process(CustomerSubmission(), "msg")

    assert result.verdict.passed is True
    assert result.refined is True
    assert len(engine.calls) == 1


@pytest.mark.asyncio
async def test_max_refines_runs_up_to_cap_and_gives_up_still_failing():
    """Every refinement valid JSON but still failing judge → 3 calls, refined=True, verdict failing."""
    engine = FakeEngine([
        {"policy_dates": {"effective_date": "2027-05-01", "expiration_date": "2026-05-01"}},
        {"policy_dates": {"effective_date": "2027-06-01", "expiration_date": "2026-06-01"}},
        {"policy_dates": {"effective_date": "2027-07-01", "expiration_date": "2026-07-01"}},
    ])
    manager = HarnessManager(SchemaJudge(), Refiner(engine), max_refines=3)
    current = CustomerSubmission()   # business_name missing too

    result = await manager.process(current, "msg")

    assert len(engine.calls) == 3
    assert result.refined is True
    assert result.verdict.passed is False   # still bad


@pytest.mark.asyncio
async def test_refiner_crash_mid_loop_keeps_successful_refine():
    """First refine succeeds (but fails new judge), second refine crashes.
    Must keep the first refine's output — it's a valid improvement."""
    engine = FakeEngine([
        # First refine: adds business_name but leaves bad dates
        {
            "business_name": "Acme",
            "policy_dates": {
                "effective_date": "2027-05-01",
                "expiration_date": "2026-05-01",
            },
        },
        # Second refine: returns non-JSON → RefinerOutputError
        "garbage-not-json",
    ])
    manager = HarnessManager(SchemaJudge(), Refiner(engine), max_refines=3)
    current = CustomerSubmission(
        policy_dates={
            "effective_date": "2027-05-01",
            "expiration_date": "2026-05-01",
        },
    )
    result = await manager.process(current, "msg")

    assert result.refined is True   # first refine DID land
    assert result.submission.business_name == "Acme"
    assert result.verdict.passed is False   # dates still bad


def test_negative_max_refines_rejected():
    with pytest.raises(ValueError):
        HarnessManager(SchemaJudge(), Refiner(FakeEngine()), max_refines=-1)
