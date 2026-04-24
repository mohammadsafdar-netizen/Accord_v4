"""Verify vLLM cache_salt is populated from request_context.get_tenant().

Context: vLLM's prefix cache is shared across all requests. When multiple
tenants share a byte-identical preamble (harness + schema), attackers can
measure cache-hit timing to detect cross-tenant activity (NDSS 2025:
"Side-Channel Attacks on LLM Inference via Prefix Caches").

Fix: set `cache_salt=<tenant_slug>` on every vLLM request. Same-salt requests
share cache; different-salt requests are namespace-isolated.

Source:
- arxiv.org/abs/2501.15925 (NDSS 2025)
- docs.vllm.ai/en/latest/features/prefix_caching.html (vLLM 0.8+ supports cache_salt)
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from accord_ai.request_context import clear_context, set_context


def _make_engine_and_mock_client():
    """Build an OpenAIEngine with a mocked AsyncOpenAI client."""
    from accord_ai.config import ExtractionMode, Settings
    from accord_ai.llm.openai_engine import OpenAIEngine

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content='{"business_name": "X"}'))]
    mock_response.model = "test-model"
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    settings = Settings(
        llm_base_url="http://localhost:8000/v1",
        llm_model="test-model",
        extraction_mode=ExtractionMode.JSON_OBJECT,
    )
    engine = OpenAIEngine(settings=settings)
    engine._client = mock_client  # injection
    return engine, mock_client


@pytest.mark.asyncio
async def test_cache_salt_absent_when_no_tenant():
    """Without tenant context, no cache_salt is passed."""
    clear_context()
    engine, mock_client = _make_engine_and_mock_client()

    await engine.generate(
        messages=[{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
    )

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    extra_body = call_kwargs.get("extra_body", {})
    assert "cache_salt" not in extra_body


@pytest.mark.asyncio
async def test_cache_salt_set_from_tenant_context():
    """With tenant context, cache_salt=tenant_slug is injected."""
    clear_context()
    set_context(request_id="rid1", tenant="acme_trucking", session_id="s1")
    try:
        engine, mock_client = _make_engine_and_mock_client()

        await engine.generate(
            messages=[{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
        )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        extra_body = call_kwargs.get("extra_body", {})
        assert extra_body.get("cache_salt") == "acme_trucking"
    finally:
        clear_context()


@pytest.mark.asyncio
async def test_cache_salt_changes_with_tenant():
    """Different tenants produce different cache_salt values (isolation)."""
    engine, mock_client = _make_engine_and_mock_client()

    # Tenant A
    clear_context()
    set_context(request_id="r1", tenant="tenant_a", session_id="s1")
    try:
        await engine.generate(
            messages=[{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
        )
        salt_a = mock_client.chat.completions.create.call_args.kwargs.get("extra_body", {}).get("cache_salt")
    finally:
        clear_context()

    # Tenant B
    clear_context()
    set_context(request_id="r2", tenant="tenant_b", session_id="s2")
    try:
        await engine.generate(
            messages=[{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
        )
        salt_b = mock_client.chat.completions.create.call_args.kwargs.get("extra_body", {}).get("cache_salt")
    finally:
        clear_context()

    assert salt_a == "tenant_a"
    assert salt_b == "tenant_b"
    assert salt_a != salt_b, "different tenants must get different salts"


@pytest.mark.asyncio
async def test_cache_salt_identical_for_same_tenant_twice():
    """Two requests from the same tenant share cache namespace (same salt)."""
    clear_context()
    set_context(request_id="r1", tenant="acme", session_id="s1")
    try:
        engine, mock_client = _make_engine_and_mock_client()

        await engine.generate(
            messages=[{"role": "system", "content": "x"}, {"role": "user", "content": "y1"}],
        )
        salt1 = mock_client.chat.completions.create.call_args.kwargs.get("extra_body", {}).get("cache_salt")

        await engine.generate(
            messages=[{"role": "system", "content": "x"}, {"role": "user", "content": "y2"}],
        )
        salt2 = mock_client.chat.completions.create.call_args.kwargs.get("extra_body", {}).get("cache_salt")

        assert salt1 == salt2 == "acme"
    finally:
        clear_context()


@pytest.mark.asyncio
async def test_cache_salt_coexists_with_llm_seed():
    """When both LLM_SEED and tenant are set, both go into extra_body."""
    from accord_ai.config import ExtractionMode, Settings
    from accord_ai.llm.openai_engine import OpenAIEngine

    clear_context()
    set_context(request_id="r1", tenant="acme", session_id="s1")
    try:
        settings = Settings(
            llm_base_url="http://localhost:8000/v1",
            llm_model="test-model",
            extraction_mode=ExtractionMode.JSON_OBJECT,
            llm_seed=42,
        )
        engine = OpenAIEngine(settings=settings)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content='{}'))]
        mock_response.model = "test-model"
        mock_response.usage = MagicMock(prompt_tokens=5, completion_tokens=1)
        engine._client = MagicMock()
        engine._client.chat.completions.create = AsyncMock(return_value=mock_response)

        await engine.generate(
            messages=[{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
        )

        extra_body = engine._client.chat.completions.create.call_args.kwargs.get("extra_body", {})
        assert extra_body.get("cache_salt") == "acme"
        assert extra_body.get("seed") == 42
    finally:
        clear_context()
