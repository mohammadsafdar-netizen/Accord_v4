"""Tests for BackendClient.push_fields / read_fields (P10.C.2)."""
from __future__ import annotations

import json

import httpx
import pytest

from accord_ai.config import Settings
from accord_ai.core.store import SessionStore
from accord_ai.integrations.backend import BackendClient


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


SUBMISSION_ID = "sub-abc-123"
TENANT_DOMAIN = "acme.brocopilot.com"
TOKEN = "fake-jwt-token"


# ---------------------------------------------------------------------------
# push_fields — happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_push_fields_success_200():
    calls = []
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.method == "POST"
        assert str(request.url) == (
            f"https://acme.copilot.inevo.ai/api/v1/submissions/"
            f"{SUBMISSION_ID}/fields"
        )
        assert request.headers["authorization"] == f"Bearer {TOKEN}"
        assert request.headers["x-forwarded-host"] == TENANT_DOMAIN
        return httpx.Response(200, json={"ok": True})

    client = _client(handler)
    try:
        result = await client.push_fields(
            TOKEN, SUBMISSION_ID, TENANT_DOMAIN,
            fields_data={"form_125": {"NamedInsured_FullName_A": "Acme"}},
            completion_percentage=42.5,
        )
        assert result is True
        assert len(calls) == 1
        body = json.loads(calls[0].read())
        assert body["fields_data"] == {
            "form_125": {"NamedInsured_FullName_A": "Acme"},
        }
        assert body["completion_percentage"] == 42.5
        assert body["source"] == "chat_backend"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_push_fields_success_201_also_accepted():
    def handler(request):
        return httpx.Response(201, json={})
    client = _client(handler)
    try:
        assert await client.push_fields(
            TOKEN, SUBMISSION_ID, TENANT_DOMAIN, {"x": 1},
        ) is True
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_push_fields_custom_source():
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200)

    client = _client(handler)
    try:
        await client.push_fields(
            TOKEN, SUBMISSION_ID, TENANT_DOMAIN, {"x": 1}, source="finalize",
        )
        assert json.loads(calls[0].read())["source"] == "finalize"
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# push_fields — failure modes (all return False, never raise)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_push_fields_http_error_returns_false():
    def handler(request):
        return httpx.Response(500, json={"err": "boom"})
    client = _client(handler)
    try:
        assert await client.push_fields(
            TOKEN, SUBMISSION_ID, TENANT_DOMAIN, {},
        ) is False
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_push_fields_network_error_returns_false():
    def handler(request):
        raise httpx.ConnectError("dns failed", request=request)
    client = _client(handler)
    try:
        assert await client.push_fields(
            TOKEN, SUBMISSION_ID, TENANT_DOMAIN, {},
        ) is False
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_push_fields_empty_token_returns_false():
    def handler(request):  # pragma: no cover — must not be called
        raise AssertionError("pushed with empty token")
    client = _client(handler)
    try:
        assert await client.push_fields(
            "", SUBMISSION_ID, TENANT_DOMAIN, {},
        ) is False
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_push_fields_ssrf_returns_false():
    def handler(request):  # pragma: no cover
        raise AssertionError("SSRF-blocked request reached transport")
    client = _client(handler)
    try:
        assert await client.push_fields(
            TOKEN, SUBMISSION_ID, "localhost", {},
        ) is False
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_push_fields_disabled_returns_false():
    def handler(request):  # pragma: no cover
        raise AssertionError("disabled client made a network call")
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = BackendClient(
        settings=_settings(backend_enabled=False), http=http,
    )
    try:
        assert await client.push_fields(
            TOKEN, SUBMISSION_ID, TENANT_DOMAIN, {},
        ) is False
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# push_fields — audit events
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_push_fields_audit_on_success(tmp_path):
    store = SessionStore(str(tmp_path / "audit.db"))
    try:
        def handler(request):
            return httpx.Response(200)
        client = _client(handler, audit_store=store)
        try:
            await client.push_fields(
                TOKEN, SUBMISSION_ID, TENANT_DOMAIN,
                {"form_125": {"a": 1, "b": 2}},
                completion_percentage=75.0,
                source="chat_backend",
            )
        finally:
            await client.aclose()

        events = store.list_audit_events(event_type="backend.push_fields")
        assert len(events) == 1
        p = events[0].payload
        assert p["status"] == "ok"
        assert p["submission_id"] == SUBMISSION_ID
        assert p["fields_count"] == 1            # one form in fields_data
        assert p["completion_pct"] == 75.0
        assert p["source"] == "chat_backend"
        assert events[0].tenant == "acme"
    finally:
        store.close()


@pytest.mark.asyncio
async def test_push_fields_audit_on_http_error(tmp_path):
    store = SessionStore(str(tmp_path / "audit.db"))
    try:
        def handler(request):
            return httpx.Response(503)
        client = _client(handler, audit_store=store)
        try:
            await client.push_fields(
                TOKEN, SUBMISSION_ID, TENANT_DOMAIN, {},
            )
        finally:
            await client.aclose()
        events = store.list_audit_events(event_type="backend.push_fields")
        assert len(events) == 1
        assert events[0].payload["status"] == "http_error"
        assert events[0].payload["http_status"] == 503
    finally:
        store.close()


# ---------------------------------------------------------------------------
# read_fields — happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_read_fields_success():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert str(request.url) == (
            f"https://acme.copilot.inevo.ai/api/v1/submissions/"
            f"{SUBMISSION_ID}/fields"
        )
        assert request.headers["authorization"] == f"Bearer {TOKEN}"
        return httpx.Response(200, json={
            "submission_id": SUBMISSION_ID,
            "fields_data": {"form_125": {"NamedInsured_FullName_A": "Acme"}},
            "sla_status":  "green",
        })

    client = _client(handler)
    try:
        result = await client.read_fields(
            TOKEN, SUBMISSION_ID, TENANT_DOMAIN,
        )
        assert result == {"form_125": {"NamedInsured_FullName_A": "Acme"}}
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# read_fields — failure modes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_read_fields_404_returns_none():
    def handler(request):
        return httpx.Response(404)
    client = _client(handler)
    try:
        assert await client.read_fields(
            TOKEN, SUBMISSION_ID, TENANT_DOMAIN,
        ) is None
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_read_fields_missing_fields_data_key_returns_none():
    """Backend returned 200 but no fields_data → treat as 'nothing there'."""
    def handler(request):
        return httpx.Response(
            200,
            json={"submission_id": SUBMISSION_ID, "sla_status": "green"},
        )
    client = _client(handler)
    try:
        assert await client.read_fields(
            TOKEN, SUBMISSION_ID, TENANT_DOMAIN,
        ) is None
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_read_fields_fields_data_wrong_type_returns_none():
    """fields_data present but not a dict → None, not a shape error."""
    def handler(request):
        return httpx.Response(200, json={"fields_data": "oops not a dict"})
    client = _client(handler)
    try:
        assert await client.read_fields(
            TOKEN, SUBMISSION_ID, TENANT_DOMAIN,
        ) is None
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_read_fields_network_error_returns_none():
    def handler(request):
        raise httpx.ConnectError("net down", request=request)
    client = _client(handler)
    try:
        assert await client.read_fields(
            TOKEN, SUBMISSION_ID, TENANT_DOMAIN,
        ) is None
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_read_fields_non_json_returns_none():
    def handler(request):
        return httpx.Response(200, content=b"<html>oops</html>")
    client = _client(handler)
    try:
        assert await client.read_fields(
            TOKEN, SUBMISSION_ID, TENANT_DOMAIN,
        ) is None
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_read_fields_empty_token_returns_none():
    def handler(request):  # pragma: no cover
        raise AssertionError("read with empty token")
    client = _client(handler)
    try:
        assert await client.read_fields(
            "", SUBMISSION_ID, TENANT_DOMAIN,
        ) is None
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_read_fields_disabled_returns_none():
    def handler(request):  # pragma: no cover
        raise AssertionError("disabled client made a network call")
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = BackendClient(
        settings=_settings(backend_enabled=False), http=http,
    )
    try:
        assert await client.read_fields(
            TOKEN, SUBMISSION_ID, TENANT_DOMAIN,
        ) is None
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_read_fields_ssrf_returns_none():
    def handler(request):  # pragma: no cover
        raise AssertionError("SSRF-blocked request reached transport")
    client = _client(handler)
    try:
        assert await client.read_fields(
            TOKEN, SUBMISSION_ID, "localhost",
        ) is None
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# read_fields — audit events
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_read_fields_audit_on_success(tmp_path):
    store = SessionStore(str(tmp_path / "audit.db"))
    try:
        def handler(request):
            return httpx.Response(200, json={
                "fields_data": {"form_125": {}, "form_130": {}},
            })
        client = _client(handler, audit_store=store)
        try:
            await client.read_fields(TOKEN, SUBMISSION_ID, TENANT_DOMAIN)
        finally:
            await client.aclose()
        events = store.list_audit_events(event_type="backend.read_fields")
        assert len(events) == 1
        assert events[0].payload["status"] == "ok"
        assert events[0].payload["fields_count"] == 2
    finally:
        store.close()


@pytest.mark.asyncio
async def test_read_fields_audit_on_http_error(tmp_path):
    store = SessionStore(str(tmp_path / "audit.db"))
    try:
        def handler(request):
            return httpx.Response(500)
        client = _client(handler, audit_store=store)
        try:
            await client.read_fields(TOKEN, SUBMISSION_ID, TENANT_DOMAIN)
        finally:
            await client.aclose()
        events = store.list_audit_events(event_type="backend.read_fields")
        assert len(events) == 1
        assert events[0].payload["status"] == "http_error"
        assert events[0].payload["http_status"] == 500
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Integration: round-trip push → read
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_push_then_read_roundtrip():
    """Ensure push and read methods coexist on the same client instance
    without state leakage (e.g. a stateful mock receiver)."""
    stored = {}
    def handler(request):
        if request.method == "POST":
            body = json.loads(request.read())
            stored["data"] = body["fields_data"]
            return httpx.Response(200)
        elif request.method == "GET":
            return httpx.Response(
                200, json={"fields_data": stored.get("data", {})},
            )

    client = _client(handler)
    try:
        pushed = {"form_125": {"NamedInsured_FullName_A": "Acme"}}
        assert await client.push_fields(
            TOKEN, SUBMISSION_ID, TENANT_DOMAIN, pushed,
        ) is True
        result = await client.read_fields(
            TOKEN, SUBMISSION_ID, TENANT_DOMAIN,
        )
        assert result == pushed
    finally:
        await client.aclose()
