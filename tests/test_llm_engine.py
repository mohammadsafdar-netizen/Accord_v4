"""7.a — Engine protocol + EngineResponse + FakeEngine."""
import json

import pytest
from dataclasses import FrozenInstanceError

from accord_ai.llm.engine import Engine, EngineResponse
from accord_ai.llm.fake_engine import FakeEngine


# --- EngineResponse ---

def test_engine_response_is_frozen():
    r = EngineResponse(text="x", model="m", tokens_in=1, tokens_out=1, latency_ms=0.0)
    with pytest.raises(FrozenInstanceError):
        r.text = "y"


# --- FakeEngine FIFO ---

@pytest.mark.asyncio
async def test_fake_engine_pops_strings_in_order():
    engine = FakeEngine(["first", "second"])
    r1 = await engine.generate([{"role": "user", "content": "q1"}])
    r2 = await engine.generate([{"role": "user", "content": "q2"}])
    assert r1.text == "first"
    assert r2.text == "second"


@pytest.mark.asyncio
async def test_fake_engine_serializes_dict_responses_to_json():
    """Dicts get json.dumps'd at enqueue — wire shape stays str."""
    engine = FakeEngine([{"business_name": "Acme"}])
    r = await engine.generate([{"role": "user", "content": "extract"}])
    assert isinstance(r.text, str)
    assert json.loads(r.text) == {"business_name": "Acme"}


@pytest.mark.asyncio
async def test_fake_engine_mixed_str_and_dict_responses():
    engine = FakeEngine([{"first": 1}, "plain text", {"last": True}])
    r1 = await engine.generate([{"role": "user", "content": "q"}])
    r2 = await engine.generate([{"role": "user", "content": "q"}])
    r3 = await engine.generate([{"role": "user", "content": "q"}])
    assert json.loads(r1.text) == {"first": 1}
    assert r2.text == "plain text"
    assert json.loads(r3.text) == {"last": True}


@pytest.mark.asyncio
async def test_fake_engine_raises_when_queue_exhausted():
    engine = FakeEngine(["only one"])
    await engine.generate([{"role": "user", "content": "q"}])
    with pytest.raises(RuntimeError):
        await engine.generate([{"role": "user", "content": "q2"}])


# --- Call history ---

@pytest.mark.asyncio
async def test_fake_engine_records_each_call():
    engine = FakeEngine(["r1", "r2"])
    msgs1 = [{"role": "user", "content": "first"}]
    msgs2 = [{"role": "system", "content": "sys"}, {"role": "user", "content": "second"}]
    await engine.generate(msgs1)
    await engine.generate(msgs2)
    assert engine.calls == [msgs1, msgs2]


@pytest.mark.asyncio
async def test_last_messages_returns_most_recent_call():
    engine = FakeEngine(["r1", "r2"])
    await engine.generate([{"role": "user", "content": "first"}])
    await engine.generate([{"role": "user", "content": "second"}])
    assert engine.last_messages[-1]["content"] == "second"


def test_last_messages_raises_before_any_call():
    """Refuses to return None — silent assertion pass is worse than loud fail."""
    engine = FakeEngine(["r1"])
    with pytest.raises(RuntimeError):
        engine.last_messages


# --- extend() ---

@pytest.mark.asyncio
async def test_extend_adds_more_responses_to_queue():
    engine = FakeEngine()
    engine.extend(["a", "b"])
    r1 = await engine.generate([{"role": "user", "content": "q"}])
    engine.extend(["c"])
    r2 = await engine.generate([{"role": "user", "content": "q"}])
    r3 = await engine.generate([{"role": "user", "content": "q"}])
    assert r1.text == "a"
    assert r2.text == "b"
    assert r3.text == "c"


# --- Observability fields ---

@pytest.mark.asyncio
async def test_response_carries_model_and_token_counts():
    engine = FakeEngine(["hello world from fake"], model="fake-v2")
    r = await engine.generate([{"role": "user", "content": "some prompt text"}])
    assert r.model == "fake-v2"
    assert r.tokens_in > 0
    assert r.tokens_out > 0
    assert r.latency_ms == 0.0


# --- Protocol conformance ---

@pytest.mark.asyncio
async def test_fake_engine_conforms_to_engine_protocol():
    """FakeEngine is structurally an Engine — no inheritance required."""
    engine: Engine = FakeEngine(["x"])
    r = await engine.generate([{"role": "user", "content": "q"}])
    assert isinstance(r, EngineResponse)
