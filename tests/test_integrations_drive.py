"""Tests for BackendClient Drive methods + DriveClient (P10.C.3)."""
from __future__ import annotations

import httpx
import pytest

from accord_ai.config import Settings
from accord_ai.core.store import SessionStore
from accord_ai.integrations.backend import BackendClient
from accord_ai.integrations.drive import (
    DriveClient,
    _escape_q_value,
    build_drive_client,
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
        drive_enabled=True,
        drive_api_base="https://www.googleapis.com",
        drive_timeout_s=5.0,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def _backend_client(handler, *, audit_store=None, settings=None) -> BackendClient:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport)
    return BackendClient(
        settings=settings or _settings(),
        http=http,
        audit_store=audit_store,
    )


def _drive_client(handler, *, audit_store=None, settings=None) -> DriveClient:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport)
    return DriveClient(
        settings=settings or _settings(),
        http=http,
        audit_store=audit_store,
    )


SID = "sub-abc"
TD  = "acme.brocopilot.com"


# ===========================================================================
# BackendClient.get_drive_token
# ===========================================================================

@pytest.mark.asyncio
async def test_drive_token_success():
    def handler(req):
        assert req.method == "GET"
        assert str(req.url) == (
            f"https://acme.copilot.inevo.ai/api/v1/drive/service-token"
            f"?submission_id={SID}"
        )
        assert req.headers["authorization"] == "Bearer svc-tok"
        assert req.headers["x-forwarded-host"] == TD
        return httpx.Response(200, json={"access_token": "drive-tok-xyz"})

    client = _backend_client(handler)
    try:
        assert await client.get_drive_token("svc-tok", SID, TD) == "drive-tok-xyz"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_drive_token_http_error_returns_none():
    def handler(req):
        return httpx.Response(403)
    client = _backend_client(handler)
    try:
        assert await client.get_drive_token("svc-tok", SID, TD) is None
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_drive_token_missing_access_token_returns_none():
    def handler(req):
        return httpx.Response(200, json={"unexpected": "shape"})
    client = _backend_client(handler)
    try:
        assert await client.get_drive_token("svc-tok", SID, TD) is None
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_drive_token_empty_service_token_returns_none():
    def handler(req):  # pragma: no cover
        raise AssertionError("called with empty service token")
    client = _backend_client(handler)
    try:
        assert await client.get_drive_token("", SID, TD) is None
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_drive_token_ssrf_returns_none():
    def handler(req):  # pragma: no cover
        raise AssertionError("SSRF reached transport")
    client = _backend_client(handler)
    try:
        assert await client.get_drive_token(
            "svc-tok", SID, "localhost",
        ) is None
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_drive_token_disabled_returns_none():
    def handler(req):  # pragma: no cover
        raise AssertionError("disabled backend made network call")
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = BackendClient(
        settings=_settings(backend_enabled=False), http=http,
    )
    try:
        assert await client.get_drive_token("svc-tok", SID, TD) is None
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_drive_token_audit(tmp_path):
    store = SessionStore(str(tmp_path / "a.db"))
    try:
        def handler(req):
            return httpx.Response(200, json={"access_token": "t"})
        client = _backend_client(handler, audit_store=store)
        try:
            await client.get_drive_token("svc-tok", SID, TD)
        finally:
            await client.aclose()
        events = store.list_audit_events(event_type="backend.drive_token")
        assert len(events) == 1
        assert events[0].tenant == "acme"
        assert events[0].payload["status"] == "ok"
    finally:
        store.close()


# ===========================================================================
# BackendClient.get_agent_folder
# ===========================================================================

@pytest.mark.asyncio
async def test_agent_folder_success():
    expected = {
        "id": "folder-123",
        "name": "Acme / CA / 2026-Q2",
        "webViewLink": "https://...",
    }
    def handler(req):
        assert str(req.url) == (
            f"https://acme.copilot.inevo.ai/api/v1/drive/agent-folder"
            f"?submission_id={SID}"
        )
        return httpx.Response(200, json=expected)

    client = _backend_client(handler)
    try:
        assert await client.get_agent_folder("svc-tok", SID, TD) == expected
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_agent_folder_missing_id_returns_none():
    def handler(req):
        return httpx.Response(200, json={"name": "no id"})
    client = _backend_client(handler)
    try:
        assert await client.get_agent_folder("svc-tok", SID, TD) is None
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_agent_folder_http_error_returns_none():
    def handler(req):
        return httpx.Response(500)
    client = _backend_client(handler)
    try:
        assert await client.get_agent_folder("svc-tok", SID, TD) is None
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_agent_folder_audit(tmp_path):
    store = SessionStore(str(tmp_path / "a.db"))
    try:
        def handler(req):
            return httpx.Response(200, json={"id": "folder-xyz"})
        client = _backend_client(handler, audit_store=store)
        try:
            await client.get_agent_folder("svc-tok", SID, TD)
        finally:
            await client.aclose()
        events = store.list_audit_events(event_type="backend.agent_folder")
        assert len(events) == 1
        assert events[0].payload["folder_id"] == "folder-xyz"
    finally:
        store.close()


# ===========================================================================
# DriveClient — q-escape helper
# ===========================================================================

def test_escape_q_value_single_quote():
    assert _escape_q_value("O'Brien's Trucking") == "O\\'Brien\\'s Trucking"


def test_escape_q_value_backslash():
    # Backslash escapes first — then single quotes would see a doubled backslash.
    assert _escape_q_value("a\\b") == "a\\\\b"


def test_escape_q_value_plain():
    assert _escape_q_value("plain_id_123") == "plain_id_123"


# ===========================================================================
# DriveClient.find_or_create_submission_folder
# ===========================================================================

@pytest.mark.asyncio
async def test_find_existing_submission_folder():
    def handler(req):
        assert req.method == "GET"
        assert "files" in str(req.url)
        assert req.headers["authorization"] == "Bearer drive-tok"
        return httpx.Response(200, json={
            "files": [{"id": "existing-folder-id", "name": SID}],
        })

    client = _drive_client(handler)
    try:
        result = await client.find_or_create_submission_folder(
            "drive-tok", SID, "parent-lob-folder-id",
        )
        assert result == "existing-folder-id"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_create_submission_folder_when_missing():
    calls = []
    def handler(req):
        calls.append((req.method, str(req.url)))
        if req.method == "GET":
            return httpx.Response(200, json={"files": []})   # not found
        if req.method == "POST":
            return httpx.Response(
                200, json={"id": "new-folder-id", "name": SID},
            )

    client = _drive_client(handler)
    try:
        result = await client.find_or_create_submission_folder(
            "drive-tok", SID, "parent-lob-folder-id",
        )
        assert result == "new-folder-id"
        assert len(calls) == 2
        assert calls[0][0] == "GET"
        assert calls[1][0] == "POST"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_find_or_create_401_returns_none():
    """Auth failures must NOT fall through to the create path; that would
    just fail again and double-audit."""
    calls = []
    def handler(req):
        calls.append(req.method)
        return httpx.Response(401, json={"error": {"code": 401}})

    client = _drive_client(handler)
    try:
        result = await client.find_or_create_submission_folder(
            "drive-tok", SID, "parent-id",
        )
        assert result is None
        assert calls == ["GET"]      # no POST attempted after 401
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_find_query_500_does_not_blindly_create():
    """Query-side 500 must NOT trigger create — could duplicate folders."""
    calls = []
    def handler(req):
        calls.append(req.method)
        return httpx.Response(500)
    client = _drive_client(handler)
    try:
        result = await client.find_or_create_submission_folder(
            "drive-tok", SID, "parent-id",
        )
        assert result is None
        assert calls == ["GET"]
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_create_501_after_empty_query_returns_none():
    """Query returns 200 empty, POST fails → None (not a dangling folder)."""
    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json={"files": []})
        return httpx.Response(501)
    client = _drive_client(handler)
    try:
        assert await client.find_or_create_submission_folder(
            "drive-tok", SID, "parent-id",
        ) is None
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_network_error_returns_none():
    def handler(req):
        raise httpx.ConnectError("dns", request=req)
    client = _drive_client(handler)
    try:
        assert await client.find_or_create_submission_folder(
            "drive-tok", SID, "parent-id",
        ) is None
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_drive_client_disabled_returns_none():
    def handler(req):  # pragma: no cover
        raise AssertionError("disabled drive client made a call")
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = DriveClient(settings=_settings(drive_enabled=False), http=http)
    try:
        assert await client.find_or_create_submission_folder(
            "drive-tok", SID, "p",
        ) is None
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_empty_drive_token_returns_none():
    def handler(req):  # pragma: no cover
        raise AssertionError("called with empty drive token")
    client = _drive_client(handler)
    try:
        assert await client.find_or_create_submission_folder(
            "", SID, "parent-id",
        ) is None
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_q_escape_applied_to_submission_id():
    """Submission IDs with single quotes must not break the Drive query."""
    captured = {}
    def handler(req):
        # Only capture the GET — the follow-up POST doesn't carry the q= param
        # and would overwrite the captured URL.
        if req.method == "GET":
            captured["url"] = str(req.url)
            return httpx.Response(200, json={"files": []})
        return httpx.Response(200, json={"id": "new-fid"})

    client = _drive_client(handler)
    try:
        await client.find_or_create_submission_folder(
            "drive-tok", "sub's-id", "parent-id",
        )
        # Escaped form appears in the q= parameter — httpx may URL-encode
        # the backslash + apostrophe, so accept either literal or percent form.
        url = captured["url"]
        assert "sub\\'s-id" in url or "sub%5C%27s-id" in url, (
            f"expected escaped sub-id in q-param: {url}"
        )
    finally:
        await client.aclose()


# ===========================================================================
# DriveClient audit events
# ===========================================================================

@pytest.mark.asyncio
async def test_audit_on_found_existing(tmp_path):
    store = SessionStore(str(tmp_path / "a.db"))
    try:
        def handler(req):
            return httpx.Response(200, json={"files": [{"id": "fid"}]})
        client = _drive_client(handler, audit_store=store)
        try:
            await client.find_or_create_submission_folder(
                "drive-tok", SID, "p", tenant="acme",
            )
        finally:
            await client.aclose()
        events = store.list_audit_events(
            event_type="drive.submission_folder_resolved",
        )
        assert len(events) == 1
        assert events[0].tenant == "acme"
        assert events[0].payload["status"] == "found"
        assert events[0].payload["folder_id"] == "fid"
    finally:
        store.close()


@pytest.mark.asyncio
async def test_audit_on_created(tmp_path):
    store = SessionStore(str(tmp_path / "a.db"))
    try:
        def handler(req):
            if req.method == "GET":
                return httpx.Response(200, json={"files": []})
            return httpx.Response(200, json={"id": "new-fid"})
        client = _drive_client(handler, audit_store=store)
        try:
            await client.find_or_create_submission_folder(
                "drive-tok", SID, "p", tenant="acme",
            )
        finally:
            await client.aclose()
        events = store.list_audit_events(
            event_type="drive.submission_folder_resolved",
        )
        assert len(events) == 1
        assert events[0].payload["status"] == "created"
        assert events[0].payload["folder_id"] == "new-fid"
    finally:
        store.close()


@pytest.mark.asyncio
async def test_audit_on_auth_failed(tmp_path):
    store = SessionStore(str(tmp_path / "a.db"))
    try:
        def handler(req):
            return httpx.Response(401)
        client = _drive_client(handler, audit_store=store)
        try:
            await client.find_or_create_submission_folder(
                "drive-tok", SID, "p", tenant="acme",
            )
        finally:
            await client.aclose()
        events = store.list_audit_events(
            event_type="drive.submission_folder_resolved",
        )
        assert len(events) == 1
        assert events[0].payload["status"] == "auth_failed"
    finally:
        store.close()


# ===========================================================================
# Factory
# ===========================================================================

def test_build_drive_client_disabled_returns_none():
    assert build_drive_client(_settings(drive_enabled=False)) is None


def test_build_drive_client_enabled_returns_client():
    client = build_drive_client(_settings())
    assert isinstance(client, DriveClient)
