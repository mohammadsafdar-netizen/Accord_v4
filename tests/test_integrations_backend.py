"""Tests for the Insurance Backend client — tenant URL + service token (P10.C.1)."""
from __future__ import annotations

import json

import httpx
import pytest

from accord_ai.config import Settings
from accord_ai.core.store import SessionStore
from accord_ai.integrations.backend import (
    BackendClient,
    BackendSSRFError,
    backend_base_url,
    build_backend_client,
)


def _settings(**overrides) -> Settings:
    defaults = dict(
        backend_enabled=True,
        backend_host_suffix="copilot.inevo.ai",
        backend_client_id="intake-agent",
        backend_client_secret="test-secret",
        backend_timeout_s=5.0,
        backend_tls_verify=True,
        backend_token_ttl_s=600,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def _client(handler, *, audit_store=None, settings=None) -> BackendClient:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport)
    return BackendClient(
        settings=settings or _settings(),
        http=http,
        audit_store=audit_store,
    )


# ---------------------------------------------------------------------------
# URL derivation + SSRF guards
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tenant_ref,expected", [
    ("acme",                         "https://acme.copilot.inevo.ai"),
    ("Acme",                         "https://acme.copilot.inevo.ai"),
    ("acme.copilot.inevo.ai",        "https://acme.copilot.inevo.ai"),
    ("acme.brocopilot.com",          "https://acme.copilot.inevo.ai"),
])
def test_backend_base_url(tenant_ref, expected):
    assert backend_base_url(tenant_ref, _settings()) == expected


@pytest.mark.parametrize("bad_ref", [
    "",
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "::1",
    "10.0.0.5",
    "192.168.1.5",
    "169.254.169.254",       # AWS metadata
    "bad slug with spaces",
    "-leading-hyphen",
    "under_score.example",
    ".leading-dot",
])
def test_backend_base_url_rejects_ssrf_inputs(bad_ref):
    with pytest.raises(BackendSSRFError):
        backend_base_url(bad_ref, _settings())


# ---------------------------------------------------------------------------
# Service token — happy path + caching
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_service_token_success():
    calls = []
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.headers["x-forwarded-host"] == "acme.brocopilot.com"
        assert str(request.url) == (
            "https://acme.copilot.inevo.ai/api/v1/auth/oauth/service-token"
        )
        return httpx.Response(200, json={"access_token": "tok-abc"})

    client = _client(handler)
    try:
        token = await client.get_service_token("acme", "acme.brocopilot.com")
        assert token == "tok-abc"
        assert len(calls) == 1

        # Body payload — parse JSON rather than string-match so we're not
        # coupled to httpx's default separator formatting.
        body = json.loads(calls[0].read())
        assert body["client_id"] == "intake-agent"
        assert body["client_secret"] == "test-secret"
        assert body["tenant_slug"] == "acme"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_get_service_token_cached_within_ttl():
    call_count = 0
    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(200, json={"access_token": "tok-cached"})

    client = _client(handler, settings=_settings(backend_token_ttl_s=600))
    try:
        a = await client.get_service_token("acme", "acme.brocopilot.com")
        b = await client.get_service_token("acme", "acme.brocopilot.com")
        c = await client.get_service_token("acme", "acme.brocopilot.com")
        assert a == b == c == "tok-cached"
        assert call_count == 1                    # cache hit x2
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_get_service_token_cache_expires():
    """ttl=0 → every call is a fresh roundtrip (cache immediately stale)."""
    call_count = 0
    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(200, json={"access_token": f"tok-{call_count}"})

    client = _client(handler, settings=_settings(backend_token_ttl_s=0))
    try:
        await client.get_service_token("acme", "acme.brocopilot.com")
        await client.get_service_token("acme", "acme.brocopilot.com")
        assert call_count == 2
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# Service token — failure modes (all must return None, never raise)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_service_token_http_error_returns_none():
    def handler(request):
        return httpx.Response(401, json={"detail": "nope"})
    client = _client(handler)
    try:
        assert await client.get_service_token(
            "acme", "acme.brocopilot.com",
        ) is None
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_get_service_token_network_error_returns_none():
    def handler(request):
        raise httpx.ConnectError("dns failed", request=request)
    client = _client(handler)
    try:
        assert await client.get_service_token(
            "acme", "acme.brocopilot.com",
        ) is None
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_get_service_token_missing_access_token_returns_none():
    def handler(request):
        return httpx.Response(200, json={"no_token_here": True})
    client = _client(handler)
    try:
        assert await client.get_service_token(
            "acme", "acme.brocopilot.com",
        ) is None
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_get_service_token_non_json_returns_none():
    def handler(request):
        return httpx.Response(200, content=b"<html>oops</html>")
    client = _client(handler)
    try:
        assert await client.get_service_token(
            "acme", "acme.brocopilot.com",
        ) is None
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_get_service_token_ssrf_returns_none():
    """localhost tenant must be swallowed — no exception surfacing."""
    def handler(request):  # pragma: no cover — must not be called
        raise AssertionError("SSRF-blocked request reached the transport")
    client = _client(handler)
    try:
        assert await client.get_service_token(
            "localhost", "localhost",
        ) is None
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_get_service_token_disabled_returns_none():
    def handler(request):  # pragma: no cover
        raise AssertionError("disabled client made a network call")
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = BackendClient(
        settings=_settings(backend_enabled=False), http=http,
    )
    try:
        assert await client.get_service_token(
            "acme", "acme.brocopilot.com",
        ) is None
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_get_service_token_missing_secret_returns_none():
    def handler(request):  # pragma: no cover
        raise AssertionError("unconfigured client made a network call")
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = BackendClient(
        settings=_settings(backend_client_secret=None),
        http=http,
    )
    try:
        assert await client.get_service_token(
            "acme", "acme.brocopilot.com",
        ) is None
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# Audit events
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_service_token_records_audit_on_success(tmp_path):
    store = SessionStore(str(tmp_path / "audit.db"))
    try:
        def handler(request):
            return httpx.Response(200, json={"access_token": "t"})
        client = _client(handler, audit_store=store)
        try:
            await client.get_service_token("acme", "acme.brocopilot.com")
        finally:
            await client.aclose()

        events = store.list_audit_events(
            event_type="backend.token_issued",
        )
        assert len(events) == 1
        assert events[0].tenant == "acme"
        assert events[0].payload["status"] == "ok"
        assert events[0].payload["ttl_s"] == 600
    finally:
        store.close()


@pytest.mark.asyncio
async def test_get_service_token_records_audit_on_ssrf(tmp_path):
    store = SessionStore(str(tmp_path / "audit.db"))
    try:
        def handler(request):
            return httpx.Response(200)
        client = _client(handler, audit_store=store)
        try:
            await client.get_service_token("localhost", "localhost")
        finally:
            await client.aclose()

        events = store.list_audit_events(
            event_type="backend.token_issued",
        )
        assert len(events) == 1
        assert events[0].payload["status"] == "ssrf_blocked"
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def test_build_backend_client_disabled_returns_none():
    assert build_backend_client(_settings(backend_enabled=False)) is None


def test_build_backend_client_enabled_returns_client():
    client = build_backend_client(_settings())
    assert isinstance(client, BackendClient)
