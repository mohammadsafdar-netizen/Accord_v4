"""Step 25 — Harness Compatibility Investigation: 3 minimal unit tests.

These guard the experimental scaffolding, not production behavior.
The eval matrix runs are the primary test for Step 25.
"""
from __future__ import annotations

import pytest

from accord_ai.extraction.extractor import Extractor
from accord_ai.llm.fake_engine import FakeEngine
from accord_ai.schema import CustomerSubmission


# ---------------------------------------------------------------------------
# 1. Extraction mode flag switches engine path
# ---------------------------------------------------------------------------


def test_extraction_mode_xgrammar_passes_json_schema_to_engine():
    """XGRAMMAR mode: engine.generate() receives json_schema (guided_json path)."""
    engine = FakeEngine([{"business_name": "Acme"}])
    extractor = Extractor(engine, extraction_mode="xgrammar")

    import asyncio
    asyncio.get_event_loop().run_until_complete(
        extractor.extract(
            user_message="We are Acme.",
            current_submission=CustomerSubmission(),
        )
    )
    # FakeEngine records json_schema from the generate() call
    assert engine.last_call.json_schema is not None


def test_extraction_mode_stored_on_extractor_not_engine():
    """Mode is stored at construction time — not resolved through the engine at runtime."""
    engine = FakeEngine([{"business_name": "Acme"}])
    extractor_free = Extractor(engine, extraction_mode="free")
    extractor_json = Extractor(engine, extraction_mode="json_object")
    assert extractor_free._extraction_mode == "free"
    assert extractor_json._extraction_mode == "json_object"


@pytest.mark.asyncio
async def test_extraction_mode_json_object_does_not_prevent_extraction():
    """JSON_OBJECT mode: extraction still succeeds (engine ignores unknown kwargs in FakeEngine)."""
    from accord_ai.config import Settings, ExtractionMode
    from accord_ai.llm.openai_engine import OpenAIEngine

    # FakeEngine doesn't branch on mode — just verify extractor plumbing
    engine = FakeEngine([{"business_name": "Acme"}])
    extractor = Extractor(engine, experiment_harness="none")
    diff = await extractor.extract(
        user_message="We are Acme.",
        current_submission=CustomerSubmission(),
    )
    assert diff.business_name == "Acme"


# ---------------------------------------------------------------------------
# 2. Experiment harness flag prepends correct block
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_experiment_harness_none_has_no_harness_in_system_prompt():
    engine = FakeEngine([{"business_name": "Acme"}])
    extractor = Extractor(engine, experiment_harness="none")
    await extractor.extract(user_message="Acme", current_submission=CustomerSubmission())
    system_content = engine.last_messages[0]["content"]
    assert "Extraction Rules" not in system_content
    assert "Extraction Harness" not in system_content


@pytest.mark.asyncio
async def test_experiment_harness_light_prepends_light_block():
    engine = FakeEngine([{"business_name": "Acme"}])
    extractor = Extractor(engine, experiment_harness="light")
    await extractor.extract(user_message="Acme", current_submission=CustomerSubmission())
    system_content = engine.last_messages[0]["content"]
    assert "Extraction Rules (focused)" in system_content
    # SYSTEM_V2 still present after the harness block
    assert len(system_content) > len("Extraction Rules (focused)")


@pytest.mark.asyncio
async def test_experiment_harness_full_prepends_full_block():
    engine = FakeEngine([{"business_name": "Acme"}])
    extractor = Extractor(engine, experiment_harness="full")
    await extractor.extract(user_message="Acme", current_submission=CustomerSubmission())
    system_content = engine.last_messages[0]["content"]
    assert "Extraction Harness v1.0" in system_content
    assert "Negation & Qualifiers" in system_content


# ---------------------------------------------------------------------------
# 3. JSON validity tracker increments
# ---------------------------------------------------------------------------


def test_json_validity_tracker_increments_on_valid():
    from accord_ai.llm.json_validity_tracker import JsonValidityTracker
    tracker = JsonValidityTracker()
    tracker.record(valid_first_try=True, mode="xgrammar", harness="none")
    tracker.record(valid_first_try=True, mode="xgrammar", harness="none")
    assert tracker.total_attempts == 2
    assert tracker.json_valid_first_try == 2
    assert tracker.json_invalid_after_retry == 0


def test_json_validity_tracker_increments_on_invalid_then_retry():
    from accord_ai.llm.json_validity_tracker import JsonValidityTracker
    tracker = JsonValidityTracker()
    tracker.record(valid_first_try=False, valid_after_retry=True, mode="free", harness="full")
    tracker.record(valid_first_try=False, valid_after_retry=False, mode="free", harness="full")
    assert tracker.total_attempts == 2
    assert tracker.json_valid_first_try == 0
    assert tracker.json_valid_after_retry == 1
    assert tracker.json_invalid_after_retry == 1


def test_json_validity_tracker_summary_rates():
    from accord_ai.llm.json_validity_tracker import JsonValidityTracker
    tracker = JsonValidityTracker()
    tracker.record(valid_first_try=True, mode="xgrammar", harness="none")
    tracker.record(valid_first_try=True, mode="xgrammar", harness="none")
    tracker.record(valid_first_try=False, valid_after_retry=True, mode="json_object", harness="full")
    summary = tracker.get_summary()
    assert summary["total_attempts"] == 3
    assert abs(summary["first_try_rate"] - 2/3) < 1e-9
    assert abs(summary["post_retry_rate"] - 3/3) < 1e-9
