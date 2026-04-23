"""Verify the harness_position flag controls where harness sits in the prompt.

Research 2026-04-22 (QWEN35_HARNESS_INVESTIGATION.md) identified prompt
placement as the likely dominant factor in Step 25's harness regression:

- Qwen3.5-9B is known to weight first system content most heavily
  (vLLM issue #23404; IFScale paper arXiv:2507.11538).
- v3 places harness at the END of its single system message, AFTER the
  schema reminder — confirmed by reading v3 prompts.py:358-461.
- v4's Step 25 matrix prepended harness BEFORE SYSTEM_V2, which means the
  model attended to harness first and SYSTEM_V2 second, inverting v3's
  attention pattern.

harness_position="after" (new) matches v3's canonical placement.
harness_position="before" (default) preserves Step 25 behavior for
apples-to-apples A/B testing.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from accord_ai.extraction import extractor as extractor_module
from accord_ai.extraction.extractor import Extractor
from accord_ai.schema import CustomerSubmission


@pytest.fixture
def mock_engine():
    eng = MagicMock()
    eng.generate = AsyncMock(return_value=MagicMock(text='{}'))
    return eng


def _build_extractor(mock_engine, **kwargs):
    """Instantiate Extractor with sensible defaults; overrides via kwargs."""
    defaults = {
        "engine": mock_engine,
        "memory": None,
        "memory_enabled": False,
        "experiment_harness": "none",
        "extraction_mode": "xgrammar",
        "harness_position": "before",
    }
    defaults.update(kwargs)
    return Extractor(**defaults)


@pytest.mark.asyncio
async def test_harness_position_before_prepends(mock_engine):
    """When position='before', harness block appears at the start of the system message."""
    extractor = _build_extractor(
        mock_engine,
        experiment_harness="core",
        harness_position="before",
    )
    await extractor.extract(
        current_submission=CustomerSubmission(),
        user_message="I need commercial auto insurance",
    )
    call = mock_engine.generate.call_args
    messages = call.args[0]
    system_msg = next(m for m in messages if m["role"] == "system")["content"]
    # "Core Principles v6.1" is the harness block opening
    core_pos = system_msg.find("Core Principles v6.1")
    # "SYSTEM_V2" isn't literally in the text, but SYSTEM_V2's characteristic
    # content starts with routing phrases. We look for where harness ends vs
    # where the rest begins.
    assert core_pos == 0 or core_pos < 30, (
        f"harness should be at/near start under 'before'; got index {core_pos}"
    )


@pytest.mark.asyncio
async def test_harness_position_after_appends(mock_engine):
    """When position='after', harness block appears at the END of the system message."""
    extractor = _build_extractor(
        mock_engine,
        experiment_harness="core",
        harness_position="after",
    )
    await extractor.extract(
        current_submission=CustomerSubmission(),
        user_message="I need commercial auto insurance",
    )
    call = mock_engine.generate.call_args
    messages = call.args[0]
    system_msg = next(m for m in messages if m["role"] == "system")["content"]
    core_pos = system_msg.find("Core Principles v6.1")
    # harness should appear AFTER the first ~100 chars (SYSTEM_V2 runs first)
    assert core_pos > 100, (
        f"harness should be at the end under 'after'; got index {core_pos} "
        f"in system message of length {len(system_msg)}"
    )
    # And the SYSTEM_V2 text should appear BEFORE the harness
    # (checking for a SYSTEM_V2 characteristic phrase — "routing" or similar)
    from accord_ai.llm.prompts import extraction as extraction_prompts
    system_v2_first_words = extraction_prompts.SYSTEM_V2.strip().split("\n")[0][:50]
    assert system_v2_first_words in system_msg
    assert system_msg.find(system_v2_first_words) < core_pos


@pytest.mark.asyncio
async def test_harness_position_none_harness_no_op(mock_engine):
    """When experiment_harness='none', position has no effect."""
    ext_before = _build_extractor(
        mock_engine, experiment_harness="none", harness_position="before"
    )
    await ext_before.extract(
        current_submission=CustomerSubmission(),
        user_message="hi",
    )
    before_msg = next(
        m for m in mock_engine.generate.call_args.args[0] if m["role"] == "system"
    )["content"]

    ext_after = _build_extractor(
        mock_engine, experiment_harness="none", harness_position="after"
    )
    await ext_after.extract(
        current_submission=CustomerSubmission(),
        user_message="hi",
    )
    after_msg = next(
        m for m in mock_engine.generate.call_args.args[0] if m["role"] == "system"
    )["content"]

    # With no harness, the two positions produce identical system content
    assert before_msg == after_msg


def test_invalid_harness_position_raises(mock_engine):
    with pytest.raises(ValueError, match="harness_position"):
        Extractor(
            engine=mock_engine,
            memory=None,
            memory_enabled=False,
            experiment_harness="core",
            harness_position="middle",
        )


@pytest.mark.asyncio
async def test_single_system_message_regardless_of_position(mock_engine):
    """v3's pattern is ONE system message. Both 'before' and 'after' must
    produce exactly one system message, not split into two.
    """
    for position in ("before", "after"):
        mock_engine.generate.reset_mock()
        ext = _build_extractor(
            mock_engine, experiment_harness="core", harness_position=position
        )
        await ext.extract(
            current_submission=CustomerSubmission(),
            user_message="test",
        )
        messages = mock_engine.generate.call_args.args[0]
        system_msgs = [m for m in messages if m["role"] == "system"]
        assert len(system_msgs) == 1, (
            f"position={position!r} produced {len(system_msgs)} system messages, "
            f"expected exactly 1"
        )


def test_config_harness_position_default_is_before():
    """Default preserves Step 25 apples-to-apples behavior until eval validates 'after'."""
    from accord_ai.config import Settings

    s = Settings()
    assert s.harness_position == "before"


def test_config_harness_position_accepts_after():
    from accord_ai.config import Settings

    s = Settings(harness_position="after")
    assert s.harness_position == "after"


def test_config_harness_position_rejects_invalid():
    from accord_ai.config import Settings
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(harness_position="middle")
