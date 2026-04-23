"""P5.0 — protocol-stack smoke tests.

Catches drift between layers before Phase 5 starts composing them. Unit
tests in individual files validate each component in isolation; these
validate that they compose as advertised.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import APITimeoutError

from accord_ai.config import Settings
from accord_ai.llm import Engine, build_engine
from accord_ai.llm.fake_engine import FakeEngine
from accord_ai.llm.openai_engine import OpenAIEngine
from accord_ai.llm.retrying_engine import RetryingEngine


# --- build_engine factory ---

def test_build_engine_returns_retrying_over_openai():
    engine = build_engine(Settings())
    assert isinstance(engine, RetryingEngine)
    assert isinstance(engine._inner, OpenAIEngine)


def test_build_engine_result_is_engine():
    """Protocol conformance — structural check."""
    engine: Engine = build_engine(Settings())
    assert hasattr(engine, "generate")


# --- Composition happy paths ---

@pytest.mark.asyncio
async def test_retrying_plus_fake_engine_happy_path():
    """FakeEngine satisfies Engine → RetryingEngine composes with it."""
    engine = RetryingEngine(FakeEngine(["ok"]), Settings())
    r = await engine.generate([{"role": "user", "content": "q"}])
    assert r.text == "ok"


@pytest.mark.asyncio
async def test_retrying_plus_openai_engine_with_stub_client():
    """Production shape: RetryingEngine(OpenAIEngine(stub_client))."""
    msg = MagicMock(); msg.content = "from stack"
    choice = MagicMock(); choice.message = msg
    usage = MagicMock(); usage.prompt_tokens = 5; usage.completion_tokens = 3
    resp = MagicMock(); resp.choices = [choice]; resp.model = "stub"; resp.usage = usage

    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=resp)

    settings = Settings()
    stack = RetryingEngine(OpenAIEngine(settings, client=client), settings)
    r = await stack.generate([{"role": "user", "content": "q"}])
    assert r.text == "from stack"
    assert r.tokens_in == 5
    assert r.tokens_out == 3


# --- Retry coordination across layers (the P5.0 bug we're closing) ---

@pytest.mark.asyncio
async def test_stack_respects_single_retry_budget(monkeypatch):
    """SDK max_retries=0 + RetryingEngine max_retries=3 == 4 total calls.

    Regression guard against the SDK silently retrying on top of us.
    """
    sleeps = []
    async def fake_sleep(s): sleeps.append(s)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=APITimeoutError(request=MagicMock())
    )

    settings = Settings()
    stack = RetryingEngine(OpenAIEngine(settings, client=client), settings)
    with pytest.raises(APITimeoutError):
        await stack.generate([{"role": "user", "content": "q"}])

    # Exactly (llm_retries + 1) SDK calls, not 2 × (llm_retries + 1) = 8
    assert client.chat.completions.create.call_count == settings.llm_retries + 1
    assert len(sleeps) == settings.llm_retries
