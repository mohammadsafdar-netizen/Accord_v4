"""7.c — OpenAIEngine unit tests. No network; client is dependency-injected."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from accord_ai.config import Settings
from accord_ai.llm.engine import Engine, EngineResponse
from accord_ai.llm.openai_engine import OpenAIEngine


def _openai_style_response(text="hello", model="test-model",
                           tokens_in=10, tokens_out=5):
    """Mock shaped like openai.types.chat.ChatCompletion."""
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    usage = MagicMock()
    usage.prompt_tokens = tokens_in
    usage.completion_tokens = tokens_out
    resp = MagicMock()
    resp.choices = [choice]
    resp.model = model
    resp.usage = usage
    return resp


def _client_returning(response):
    c = MagicMock()
    c.chat.completions.create = AsyncMock(return_value=response)
    return c


@pytest.fixture
def settings():
    return Settings()


# --- Happy path ---

@pytest.mark.asyncio
async def test_generate_returns_engine_response(settings):
    engine = OpenAIEngine(settings, client=_client_returning(
        _openai_style_response(text="Hi there")))
    r = await engine.generate([{"role": "user", "content": "hello"}])
    assert isinstance(r, EngineResponse)
    assert r.text == "Hi there"


@pytest.mark.asyncio
async def test_generate_uses_configured_model(settings):
    client = _client_returning(_openai_style_response())
    engine = OpenAIEngine(settings, client=client)
    await engine.generate([{"role": "user", "content": "hi"}])
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "insurance-agent"


@pytest.mark.asyncio
async def test_generate_forwards_temperature_and_max_tokens(settings):
    client = _client_returning(_openai_style_response())
    engine = OpenAIEngine(settings, client=client)
    await engine.generate(
        [{"role": "user", "content": "hi"}],
        temperature=0.7,
        max_tokens=100,
    )
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["temperature"] == 0.7
    assert kwargs["max_tokens"] == 100


@pytest.mark.asyncio
async def test_generate_populates_token_counts(settings):
    engine = OpenAIEngine(settings, client=_client_returning(
        _openai_style_response(tokens_in=42, tokens_out=7)))
    r = await engine.generate([{"role": "user", "content": "q"}])
    assert r.tokens_in == 42
    assert r.tokens_out == 7


@pytest.mark.asyncio
async def test_generate_measures_latency(settings):
    engine = OpenAIEngine(settings, client=_client_returning(
        _openai_style_response()))
    r = await engine.generate([{"role": "user", "content": "q"}])
    assert r.latency_ms >= 0.0


@pytest.mark.asyncio
async def test_generate_none_content_becomes_empty_string(settings):
    engine = OpenAIEngine(settings, client=_client_returning(
        _openai_style_response(text=None)))
    r = await engine.generate([{"role": "user", "content": "q"}])
    assert r.text == ""


@pytest.mark.asyncio
async def test_generate_zero_usage_when_usage_missing(settings):
    """If the provider omits usage (rare), token counts default to 0."""
    resp = _openai_style_response()
    resp.usage = None
    engine = OpenAIEngine(settings, client=_client_returning(resp))
    r = await engine.generate([{"role": "user", "content": "q"}])
    assert r.tokens_in == 0
    assert r.tokens_out == 0


# --- Error policy ---

@pytest.mark.asyncio
async def test_client_exceptions_propagate_unwrapped(settings):
    """No EngineError wrapper — runner (7.d) sees provider-native types."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=RuntimeError("simulated provider outage")
    )
    engine = OpenAIEngine(settings, client=client)
    with pytest.raises(RuntimeError, match="simulated provider outage"):
        await engine.generate([{"role": "user", "content": "q"}])


# --- API key handling ---

@pytest.mark.asyncio
async def test_no_api_key_uses_sk_unused_sentinel(monkeypatch):
    """Local vLLM ignores api_key value but openai SDK requires a string."""
    s = Settings()
    assert s.llm_api_key is None
    # Construction must not raise; real AsyncOpenAI gets "sk-unused"
    engine = OpenAIEngine(s, client=_client_returning(_openai_style_response()))
    r = await engine.generate([{"role": "user", "content": "q"}])
    assert isinstance(r, EngineResponse)


# --- Protocol conformance ---

@pytest.mark.asyncio
async def test_conforms_to_engine_protocol(settings):
    engine: Engine = OpenAIEngine(
        settings, client=_client_returning(_openai_style_response())
    )
    r = await engine.generate([{"role": "user", "content": "q"}])
    assert isinstance(r, EngineResponse)


# --- Qwen3 hybrid-reasoning CoT suppression ---

@pytest.mark.asyncio
async def test_generate_passes_enable_thinking_false_extra_body(settings):
    """Every call must send chat_template_kwargs.enable_thinking=False via
    extra_body — otherwise Qwen3 emits a multi-paragraph 'Thinking Process:'
    dump as the user-facing response (observed live on /start-session)."""
    client = _client_returning(_openai_style_response())
    engine = OpenAIEngine(settings, client=client)
    await engine.generate([{"role": "user", "content": "q"}])
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False},
    }


@pytest.mark.asyncio
async def test_generate_strips_think_tag_blocks(settings):
    """Safety net: if a provider / variant emits <think>…</think> despite
    enable_thinking=False, strip it from the returned text."""
    resp = _openai_style_response(
        text="<think>reasoning chain here</think>Hello there!",
    )
    engine = OpenAIEngine(settings, client=_client_returning(resp))
    r = await engine.generate([{"role": "user", "content": "q"}])
    assert r.text == "Hello there!"


@pytest.mark.asyncio
async def test_generate_strips_unterminated_think_block(settings):
    """Some models start a <think> block and never close it — scrub to EOS."""
    resp = _openai_style_response(text="<think>raw reasoning never closed")
    engine = OpenAIEngine(settings, client=_client_returning(resp))
    r = await engine.generate([{"role": "user", "content": "q"}])
    assert r.text == ""


@pytest.mark.asyncio
async def test_generate_strips_thinking_process_prefix(settings):
    """Hybrid model sometimes narrates: 'Thinking Process: …\\nFinal Answer:\\n<ans>'."""
    raw = (
        "Thinking Process:\n"
        "1. Analyze.\n"
        "2. Compose.\n"
        "Final Answer:\n"
        "Hi — what's your business name?"
    )
    resp = _openai_style_response(text=raw)
    engine = OpenAIEngine(settings, client=_client_returning(resp))
    r = await engine.generate([{"role": "user", "content": "q"}])
    assert r.text == "Hi — what's your business name?"


@pytest.mark.asyncio
async def test_generate_leaves_clean_output_untouched(settings):
    """No reasoning markers = no stripping."""
    resp = _openai_style_response(text="Hi — what's your business name?")
    engine = OpenAIEngine(settings, client=_client_returning(resp))
    r = await engine.generate([{"role": "user", "content": "q"}])
    assert r.text == "Hi — what's your business name?"


# ---------------------------------------------------------------------------
# Schema sanitization for xgrammar (guided_json preprocessing)
# ---------------------------------------------------------------------------


def test_sanitize_flattens_optional_to_bare_type():
    """Pydantic emits Optional[str] as anyOf:[{type:string}, {type:null}].
    xgrammar handles the plain {type:string} schema cleanly, so flatten."""
    from accord_ai.llm.openai_engine import _sanitize_schema_for_xgrammar
    raw = {
        "properties": {
            "business_name": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
                "title": "Business Name",
            },
        },
    }
    out = _sanitize_schema_for_xgrammar(raw)
    bn = out["properties"]["business_name"]
    assert bn["type"] == "string"
    assert "anyOf" not in bn
    # Wrapper metadata preserved on the hoisted schema.
    assert bn.get("title") == "Business Name"
    assert bn.get("default") is None


def test_sanitize_flattens_optional_literal_enum():
    """Optional[Literal[a,b,c]] → anyOf:[{enum:[a,b,c], type:string}, null].
    After sanitize, the enum/type land at the top so xgrammar pins tokens."""
    from accord_ai.llm.openai_engine import _sanitize_schema_for_xgrammar
    raw = {
        "properties": {
            "entity_type": {
                "anyOf": [
                    {"enum": ["corporation", "llc", "individual"], "type": "string"},
                    {"type": "null"},
                ],
                "default": None,
                "title": "Entity Type",
            },
        },
    }
    out = _sanitize_schema_for_xgrammar(raw)
    et = out["properties"]["entity_type"]
    assert et["enum"] == ["corporation", "llc", "individual"]
    assert et["type"] == "string"
    assert "anyOf" not in et


def test_sanitize_flattens_discriminated_union():
    """Optional[Union[A,B,C] discriminator='lob'] → anyOf:[{oneOf:[...],
    discriminator:...}, null]. After sanitize, the oneOf+discriminator
    survives without the null alternative — xgrammar can enforce the
    object union, but struggles with object-or-null at the field level."""
    from accord_ai.llm.openai_engine import _sanitize_schema_for_xgrammar
    raw = {
        "anyOf": [
            {
                "discriminator": {"propertyName": "lob"},
                "oneOf": [
                    {"$ref": "#/$defs/CA"},
                    {"$ref": "#/$defs/GL"},
                ],
            },
            {"type": "null"},
        ],
        "default": None,
    }
    out = _sanitize_schema_for_xgrammar(raw)
    assert "anyOf" not in out
    assert "oneOf" in out
    assert out.get("discriminator", {}).get("propertyName") == "lob"


def test_sanitize_leaves_non_null_unions_alone():
    """True unions (anyOf: [A, B] with no null branch) stay intact —
    only the Optional idiom (anyOf: [T, null]) collapses."""
    from accord_ai.llm.openai_engine import _sanitize_schema_for_xgrammar
    raw = {
        "anyOf": [
            {"type": "string"},
            {"type": "integer"},
        ],
    }
    out = _sanitize_schema_for_xgrammar(raw)
    assert out == raw   # unchanged


def test_sanitize_recurses_into_nested_objects():
    """Optional fields inside nested objects get flattened too."""
    from accord_ai.llm.openai_engine import _sanitize_schema_for_xgrammar
    raw = {
        "type": "object",
        "properties": {
            "address": {
                "type": "object",
                "properties": {
                    "city": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                    },
                },
            },
        },
    }
    out = _sanitize_schema_for_xgrammar(raw)
    city = out["properties"]["address"]["properties"]["city"]
    assert city == {"type": "string"}


def test_sanitize_does_not_mutate_input():
    """Sanitizer returns a copy; caller's schema stays intact."""
    from accord_ai.llm.openai_engine import _sanitize_schema_for_xgrammar
    raw = {
        "properties": {
            "x": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
    }
    _ = _sanitize_schema_for_xgrammar(raw)
    assert raw["properties"]["x"]["anyOf"] == [
        {"type": "string"}, {"type": "null"},
    ]


@pytest.mark.asyncio
async def test_generate_sends_schema_verbatim_to_vllm(settings):
    """The schema that reaches vLLM's guided_json is identical to what
    the caller passed. Sanitization is shelved (see engine comment);
    this test pins "no covert rewriting" so any future re-enabling of
    the sanitizer breaks this intentionally."""
    raw = {
        "properties": {
            "entity_type": {
                "anyOf": [
                    {"enum": ["llc", "corporation"], "type": "string"},
                    {"type": "null"},
                ],
            },
        },
    }
    client = _client_returning(_openai_style_response())
    engine = OpenAIEngine(settings, client=client)
    await engine.generate(
        [{"role": "user", "content": "q"}], json_schema=raw,
    )
    sent = client.chat.completions.create.call_args.kwargs[
        "extra_body"
    ]["guided_json"]
    assert sent == raw


@pytest.mark.asyncio
async def test_generate_without_json_schema_omits_guided_json(settings):
    """Default (no schema) → extra_body has only chat_template_kwargs,
    no guided_json key. Keeps free-form text calls (responder) unaffected."""
    client = _client_returning(_openai_style_response())
    engine = OpenAIEngine(settings, client=client)
    await engine.generate([{"role": "user", "content": "q"}])
    kwargs = client.chat.completions.create.call_args.kwargs
    assert "guided_json" not in kwargs["extra_body"]


@pytest.mark.asyncio
async def test_generate_with_json_schema_adds_guided_json_to_extra_body(settings):
    """json_schema kwarg → vLLM structured-output directive goes through
    extra_body.guided_json alongside the thinking toggle (both survive)."""
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    client = _client_returning(_openai_style_response())
    engine = OpenAIEngine(settings, client=client)
    await engine.generate([{"role": "user", "content": "q"}], json_schema=schema)
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["extra_body"]["guided_json"] == schema
    # Thinking-toggle still present — guided_json must never drop the
    # enable_thinking=False directive.
    assert kwargs["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False,
    }


@pytest.mark.asyncio
async def test_generate_default_max_tokens_is_1024(settings):
    """Engine default dropped from 4096 → 1024 to bound cost/latency on
    call sites that don't override (responder's 2-3 sentence replies
    never need more than ~200)."""
    client = _client_returning(_openai_style_response())
    engine = OpenAIEngine(settings, client=client)
    await engine.generate([{"role": "user", "content": "q"}])
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["max_tokens"] == 1024


# --- SDK retry disabled (P5.0 regression guard) ---

def test_real_client_has_sdk_retries_disabled():
    """Must be 0 — RetryingEngine owns the retry budget. Prevents double-retry."""
    engine = OpenAIEngine(Settings())
    assert engine._client.max_retries == 0
