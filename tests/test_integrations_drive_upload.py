"""Tests for DriveClient.upload_filled_pdf (P10.C.4)."""
from __future__ import annotations

import httpx
import pytest

from accord_ai.config import Settings
from accord_ai.core.store import SessionStore
from accord_ai.integrations.drive import (
    DriveAuthError,
    DriveClient,
    UploadResult,
)


PDF_BYTES = b"%PDF-1.4\n%fake-pdf-content\n%%EOF\n"
FOLDER_ID = "folder-abc"
FILE_NAME = "ACORD_125_filled.pdf"


def _settings(**overrides) -> Settings:
    defaults = dict(
        backend_enabled=True,
        backend_host_suffix="copilot.inevo.ai",
        backend_client_id="intake-agent",
        backend_client_secret="test-secret",
        drive_enabled=True,
        drive_api_base="https://www.googleapis.com",
        drive_timeout_s=5.0,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def _client(handler, *, audit_store=None, settings=None) -> DriveClient:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport)
    return DriveClient(
        settings=settings or _settings(),
        http=http,
        audit_store=audit_store,
    )


# ---------------------------------------------------------------------------
# UploadResult helper
# ---------------------------------------------------------------------------

def test_upload_result_from_id_builds_urls():
    r = UploadResult.from_id("fid-xyz")
    assert r.file_id == "fid-xyz"
    assert r.view_url == "https://drive.google.com/file/d/fid-xyz/view"
    assert "id=fid-xyz" in r.web_content_link


def test_upload_result_is_frozen():
    r = UploadResult.from_id("fid")
    with pytest.raises((AttributeError, TypeError)):
        r.file_id = "other"  # type: ignore


# ---------------------------------------------------------------------------
# POST fresh — no existing_file_id, probe empty
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upload_post_when_no_existing_file():
    calls = []
    def handler(req):
        calls.append((req.method, str(req.url)))
        if req.method == "GET":
            return httpx.Response(200, json={"files": []})
        if req.method == "POST":
            assert "uploadType=multipart" in str(req.url)
            return httpx.Response(200, json={"id": "new-fid-1"})

    client = _client(handler)
    try:
        result = await client.upload_filled_pdf(
            "drive-tok", FOLDER_ID, FILE_NAME, PDF_BYTES,
        )
        assert isinstance(result, UploadResult)
        assert result.file_id == "new-fid-1"
        assert len(calls) == 2
        assert calls[0][0] == "GET"
        assert calls[1][0] == "POST"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_upload_post_metadata_includes_parents():
    captured = {}
    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json={"files": []})
        captured["body"] = req.read()
        return httpx.Response(200, json={"id": "fid"})
    client = _client(handler)
    try:
        await client.upload_filled_pdf(
            "drive-tok", FOLDER_ID, FILE_NAME, PDF_BYTES,
        )
        body_str = captured["body"].decode(errors="ignore")
        assert FILE_NAME in body_str
        assert FOLDER_ID in body_str          # parents[0]
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# PATCH when existing_file_id provided — skips probe
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upload_patch_with_existing_file_id_skips_probe():
    calls = []
    def handler(req):
        calls.append((req.method, str(req.url)))
        if req.method == "PATCH":
            return httpx.Response(200, json={"id": "existing-fid"})

    client = _client(handler)
    try:
        result = await client.upload_filled_pdf(
            "drive-tok", FOLDER_ID, FILE_NAME, PDF_BYTES,
            existing_file_id="existing-fid",
        )
        assert result is not None
        assert result.file_id == "existing-fid"
        assert len(calls) == 1
        assert calls[0][0] == "PATCH"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_upload_patch_metadata_excludes_parents():
    captured = {}
    def handler(req):
        if req.method == "PATCH":
            captured["body"] = req.read()
            return httpx.Response(200, json={"id": "fid"})
    client = _client(handler)
    try:
        await client.upload_filled_pdf(
            "drive-tok", FOLDER_ID, FILE_NAME, PDF_BYTES,
            existing_file_id="fid",
        )
        body_str = captured["body"].decode(errors="ignore")
        # PATCH metadata must NOT carry parents (immutable on update)
        assert '"parents":' not in body_str
        assert '"parents" :' not in body_str
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# PATCH → 404 → POST fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upload_patch_404_falls_back_to_post():
    calls = []
    def handler(req):
        calls.append(req.method)
        if req.method == "PATCH":
            return httpx.Response(404)
        if req.method == "POST":
            return httpx.Response(200, json={"id": "recreated-fid"})

    client = _client(handler)
    try:
        result = await client.upload_filled_pdf(
            "drive-tok", FOLDER_ID, FILE_NAME, PDF_BYTES,
            existing_file_id="stale-fid",
        )
        assert result is not None
        assert result.file_id == "recreated-fid"
        assert calls == ["PATCH", "POST"]
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# Probe finds existing → PATCH with discovered id
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upload_probe_finds_existing_then_patches():
    calls = []
    def handler(req):
        calls.append((req.method, str(req.url)))
        if req.method == "GET":
            return httpx.Response(
                200, json={"files": [{"id": "found-fid"}]},
            )
        if req.method == "PATCH":
            assert "found-fid" in str(req.url)
            return httpx.Response(200, json={"id": "found-fid"})

    client = _client(handler)
    try:
        result = await client.upload_filled_pdf(
            "drive-tok", FOLDER_ID, FILE_NAME, PDF_BYTES,
        )
        assert result.file_id == "found-fid"
        assert calls[0][0] == "GET"
        assert calls[1][0] == "PATCH"
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# 401 at any stage — raises DriveAuthError
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upload_401_on_probe_raises():
    def handler(req):
        return httpx.Response(401)
    client = _client(handler)
    try:
        with pytest.raises(DriveAuthError):
            await client.upload_filled_pdf(
                "drive-tok", FOLDER_ID, FILE_NAME, PDF_BYTES,
            )
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_upload_401_on_post_raises():
    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json={"files": []})
        return httpx.Response(401)
    client = _client(handler)
    try:
        with pytest.raises(DriveAuthError):
            await client.upload_filled_pdf(
                "drive-tok", FOLDER_ID, FILE_NAME, PDF_BYTES,
            )
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_upload_401_on_patch_raises():
    def handler(req):
        return httpx.Response(401)
    client = _client(handler)
    try:
        with pytest.raises(DriveAuthError):
            await client.upload_filled_pdf(
                "drive-tok", FOLDER_ID, FILE_NAME, PDF_BYTES,
                existing_file_id="fid",
            )
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# Non-auth failures — None (no raise)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upload_500_returns_none():
    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json={"files": []})
        return httpx.Response(500)
    client = _client(handler)
    try:
        assert await client.upload_filled_pdf(
            "drive-tok", FOLDER_ID, FILE_NAME, PDF_BYTES,
        ) is None
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_upload_network_error_returns_none():
    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json={"files": []})
        raise httpx.ConnectError("dns", request=req)
    client = _client(handler)
    try:
        assert await client.upload_filled_pdf(
            "drive-tok", FOLDER_ID, FILE_NAME, PDF_BYTES,
        ) is None
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_upload_missing_id_in_response_returns_none():
    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json={"files": []})
        return httpx.Response(200, json={"name": "no id"})
    client = _client(handler)
    try:
        assert await client.upload_filled_pdf(
            "drive-tok", FOLDER_ID, FILE_NAME, PDF_BYTES,
        ) is None
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# Disabled / empty token
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upload_empty_token_returns_none():
    def handler(req):  # pragma: no cover
        raise AssertionError("called with empty drive_token")
    client = _client(handler)
    try:
        assert await client.upload_filled_pdf(
            "", FOLDER_ID, FILE_NAME, PDF_BYTES,
        ) is None
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_upload_disabled_returns_none():
    def handler(req):  # pragma: no cover
        raise AssertionError("disabled drive client uploaded")
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = DriveClient(
        settings=_settings(drive_enabled=False), http=http,
    )
    try:
        assert await client.upload_filled_pdf(
            "drive-tok", FOLDER_ID, FILE_NAME, PDF_BYTES,
        ) is None
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# Audit events
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upload_audit_on_ok_post(tmp_path):
    store = SessionStore(str(tmp_path / "a.db"))
    try:
        def handler(req):
            if req.method == "GET":
                return httpx.Response(200, json={"files": []})
            return httpx.Response(200, json={"id": "new-fid"})
        client = _client(handler, audit_store=store)
        try:
            await client.upload_filled_pdf(
                "drive-tok", FOLDER_ID, FILE_NAME, PDF_BYTES, tenant="acme",
            )
        finally:
            await client.aclose()
        events = store.list_audit_events(event_type="drive.upload")
        assert len(events) == 1
        assert events[0].tenant == "acme"
        p = events[0].payload
        assert p["status"] == "ok"                # POST path
        assert p["file_id"] == "new-fid"
        assert p["file_name"] == FILE_NAME
        assert p["byte_size"] == len(PDF_BYTES)
    finally:
        store.close()


@pytest.mark.asyncio
async def test_upload_audit_on_overwritten_patch(tmp_path):
    store = SessionStore(str(tmp_path / "a.db"))
    try:
        def handler(req):
            return httpx.Response(200, json={"id": "fid"})
        client = _client(handler, audit_store=store)
        try:
            await client.upload_filled_pdf(
                "drive-tok", FOLDER_ID, FILE_NAME, PDF_BYTES,
                existing_file_id="fid", tenant="acme",
            )
        finally:
            await client.aclose()
        events = store.list_audit_events(event_type="drive.upload")
        assert len(events) == 1
        assert events[0].payload["status"] == "overwritten"
    finally:
        store.close()


@pytest.mark.asyncio
async def test_upload_audit_on_auth_failed_probe(tmp_path):
    store = SessionStore(str(tmp_path / "a.db"))
    try:
        def handler(req):
            return httpx.Response(401)
        client = _client(handler, audit_store=store)
        try:
            with pytest.raises(DriveAuthError):
                await client.upload_filled_pdf(
                    "drive-tok", FOLDER_ID, FILE_NAME, PDF_BYTES,
                    tenant="acme",
                )
        finally:
            await client.aclose()
        events = store.list_audit_events(event_type="drive.upload")
        assert len(events) == 1
        assert events[0].payload["status"] == "auth_failed"
        assert events[0].payload["stage"] == "probe"
    finally:
        store.close()


@pytest.mark.asyncio
async def test_upload_audit_on_http_error(tmp_path):
    store = SessionStore(str(tmp_path / "a.db"))
    try:
        def handler(req):
            if req.method == "GET":
                return httpx.Response(200, json={"files": []})
            return httpx.Response(503)
        client = _client(handler, audit_store=store)
        try:
            await client.upload_filled_pdf(
                "drive-tok", FOLDER_ID, FILE_NAME, PDF_BYTES,
                tenant="acme",
            )
        finally:
            await client.aclose()
        events = store.list_audit_events(event_type="drive.upload")
        assert len(events) == 1
        assert events[0].payload["status"] == "http_error"
        assert events[0].payload["http_status"] == 503
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Multipart body structure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_multipart_body_contains_both_parts():
    captured = {}
    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json={"files": []})
        captured["body"] = req.read()
        captured["ct"] = req.headers.get("content-type", "")
        return httpx.Response(200, json={"id": "fid"})
    client = _client(handler)
    try:
        await client.upload_filled_pdf(
            "drive-tok", FOLDER_ID, FILE_NAME, PDF_BYTES,
        )
        body = captured["body"]
        assert b"application/json" in body
        assert b"application/pdf" in body
        assert PDF_BYTES in body                  # raw PDF bytes embedded
        assert captured["ct"].startswith("multipart/related; boundary=")
    finally:
        await client.aclose()
