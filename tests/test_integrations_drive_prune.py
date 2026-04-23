"""Tests for DriveClient.list_folder_children + prune_stale_pdfs (P10.C.5)."""
from __future__ import annotations

import httpx
import pytest

from accord_ai.config import Settings
from accord_ai.core.store import SessionStore
from accord_ai.integrations.drive import DriveAuthError, DriveClient


FOLDER_ID = "folder-sub-1"


def _settings(**overrides) -> Settings:
    defaults = dict(
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
# list_folder_children
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_returns_name_to_id_map():
    def handler(req):
        return httpx.Response(200, json={"files": [
            {"id": "fid-125", "name": "ACORD_125_filled.pdf"},
            {"id": "fid-127", "name": "ACORD_127_filled.pdf"},
            {"id": "fid-129", "name": "ACORD_129_filled.pdf"},
        ]})
    client = _client(handler)
    try:
        result = await client.list_folder_children("drive-tok", FOLDER_ID)
        assert result == {
            "ACORD_125_filled.pdf": "fid-125",
            "ACORD_127_filled.pdf": "fid-127",
            "ACORD_129_filled.pdf": "fid-129",
        }
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_list_q_excludes_folders():
    captured = {}
    def handler(req):
        captured["url"] = str(req.url)
        return httpx.Response(200, json={"files": []})
    client = _client(handler)
    try:
        await client.list_folder_children("drive-tok", FOLDER_ID)
        # q= should contain the mimeType-exclusion predicate. httpx encodes
        # `/` as %2F in query params, so accept either encoded or raw form.
        url = captured["url"]
        assert "mimeType" in url
        assert (
            "application/vnd.google-apps.folder" in url
            or "application%2Fvnd.google-apps.folder" in url
        )
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_list_skips_malformed_rows():
    """Defensive: if Drive returns a row missing name or id, skip silently."""
    def handler(req):
        return httpx.Response(200, json={"files": [
            {"id": "fid-1", "name": "good.pdf"},
            {"id": "fid-2"},                   # missing name
            {"name": "orphan.pdf"},            # missing id
            {"id": "", "name": "empty.pdf"},   # empty id
            "string-instead-of-dict",          # wrong type
        ]})
    client = _client(handler)
    try:
        result = await client.list_folder_children("drive-tok", FOLDER_ID)
        assert result == {"good.pdf": "fid-1"}
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_list_401_raises():
    def handler(req):
        return httpx.Response(401)
    client = _client(handler)
    try:
        with pytest.raises(DriveAuthError):
            await client.list_folder_children("drive-tok", FOLDER_ID)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_list_500_returns_empty():
    def handler(req):
        return httpx.Response(500)
    client = _client(handler)
    try:
        assert await client.list_folder_children(
            "drive-tok", FOLDER_ID,
        ) == {}
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_list_network_error_returns_empty():
    def handler(req):
        raise httpx.ConnectError("down", request=req)
    client = _client(handler)
    try:
        assert await client.list_folder_children(
            "drive-tok", FOLDER_ID,
        ) == {}
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_list_disabled_returns_empty():
    def handler(req):  # pragma: no cover
        raise AssertionError("disabled client listed")
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = DriveClient(settings=_settings(drive_enabled=False), http=http)
    try:
        assert await client.list_folder_children(
            "drive-tok", FOLDER_ID,
        ) == {}
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# prune_stale_pdfs — core behaviors
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_prune_deletes_only_not_in_keep_set():
    deletes_seen = []
    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json={"files": [
                {"id": "fid-125", "name": "ACORD_125_filled.pdf"},
                {"id": "fid-127", "name": "ACORD_127_filled.pdf"},   # stale
                {"id": "fid-129", "name": "ACORD_129_filled.pdf"},   # stale
                {"id": "fid-126", "name": "ACORD_126_filled.pdf"},
            ]})
        if req.method == "DELETE":
            fid = str(req.url).rsplit("/", 1)[-1]
            deletes_seen.append(fid)
            return httpx.Response(204)

    client = _client(handler)
    try:
        keep = {"ACORD_125_filled.pdf", "ACORD_126_filled.pdf"}
        deleted = await client.prune_stale_pdfs(
            "drive-tok", FOLDER_ID, keep,
        )
        assert set(deleted) == {"fid-127", "fid-129"}
        assert set(deletes_seen) == {"fid-127", "fid-129"}
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_prune_no_stale_files_no_deletes():
    calls = []
    def handler(req):
        calls.append(req.method)
        if req.method == "GET":
            return httpx.Response(200, json={"files": [
                {"id": "fid-125", "name": "ACORD_125_filled.pdf"},
            ]})

    client = _client(handler)
    try:
        keep = {"ACORD_125_filled.pdf"}
        deleted = await client.prune_stale_pdfs(
            "drive-tok", FOLDER_ID, keep,
        )
        assert deleted == []
        assert calls == ["GET"]    # no DELETE calls
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_prune_empty_keep_set_deletes_all():
    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json={"files": [
                {"id": "fid-1", "name": "a.pdf"},
                {"id": "fid-2", "name": "b.pdf"},
            ]})
        if req.method == "DELETE":
            return httpx.Response(204)

    client = _client(handler)
    try:
        deleted = await client.prune_stale_pdfs(
            "drive-tok", FOLDER_ID, set(),
        )
        assert set(deleted) == {"fid-1", "fid-2"}
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_prune_accepts_folder_snapshot_skips_list():
    """Caller already listed children — skip the extra API call."""
    calls = []
    def handler(req):
        calls.append(req.method)
        if req.method == "DELETE":
            return httpx.Response(204)

    client = _client(handler)
    try:
        snapshot = {
            "ACORD_125_filled.pdf": "fid-125",
            "stale.pdf":             "fid-stale",
        }
        deleted = await client.prune_stale_pdfs(
            "drive-tok", FOLDER_ID,
            keep_file_names={"ACORD_125_filled.pdf"},
            folder_snapshot=snapshot,
        )
        assert deleted == ["fid-stale"]
        assert "GET" not in calls           # snapshot skipped the list
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_prune_accepts_200_or_204():
    """Drive sometimes returns 200 instead of 204 on delete."""
    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json={"files": [
                {"id": "fid-a", "name": "a.pdf"},
                {"id": "fid-b", "name": "b.pdf"},
            ]})
        fid = str(req.url).rsplit("/", 1)[-1]
        return httpx.Response(200 if fid == "fid-a" else 204)

    client = _client(handler)
    try:
        deleted = await client.prune_stale_pdfs(
            "drive-tok", FOLDER_ID, set(),
        )
        assert set(deleted) == {"fid-a", "fid-b"}
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_prune_partial_failure_returns_only_successes():
    """One delete fails, others succeed — batch continues, returns successes."""
    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json={"files": [
                {"id": "fid-ok",   "name": "a.pdf"},
                {"id": "fid-fail", "name": "b.pdf"},
                {"id": "fid-ok2",  "name": "c.pdf"},
            ]})
        fid = str(req.url).rsplit("/", 1)[-1]
        if fid == "fid-fail":
            return httpx.Response(500)
        return httpx.Response(204)

    client = _client(handler)
    try:
        deleted = await client.prune_stale_pdfs(
            "drive-tok", FOLDER_ID, set(),
        )
        assert set(deleted) == {"fid-ok", "fid-ok2"}
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_prune_401_on_list_raises():
    def handler(req):
        return httpx.Response(401)
    client = _client(handler)
    try:
        with pytest.raises(DriveAuthError):
            await client.prune_stale_pdfs(
                "drive-tok", FOLDER_ID, set(),
            )
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_prune_401_mid_delete_does_not_raise():
    """An individual delete 401 is treated as a per-file failure, not a
    batch-level auth re-exchange signal. The initial list succeeded, so
    the token was valid at start — if a mid-batch 401 appears it's more
    likely a per-file permission issue. Don't raise; keep going."""
    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json={"files": [
                {"id": "fid-a", "name": "a.pdf"},
                {"id": "fid-b", "name": "b.pdf"},
            ]})
        fid = str(req.url).rsplit("/", 1)[-1]
        if fid == "fid-a":
            return httpx.Response(401)
        return httpx.Response(204)

    client = _client(handler)
    try:
        deleted = await client.prune_stale_pdfs(
            "drive-tok", FOLDER_ID, set(),
        )
        assert deleted == ["fid-b"]
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_prune_network_error_on_single_delete_counted_as_failure():
    """Per-file network error is logged, doesn't abort the batch."""
    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json={"files": [
                {"id": "fid-net", "name": "a.pdf"},
                {"id": "fid-ok",  "name": "b.pdf"},
            ]})
        fid = str(req.url).rsplit("/", 1)[-1]
        if fid == "fid-net":
            raise httpx.ConnectError("boom", request=req)
        return httpx.Response(204)

    client = _client(handler)
    try:
        deleted = await client.prune_stale_pdfs(
            "drive-tok", FOLDER_ID, set(),
        )
        assert deleted == ["fid-ok"]
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_prune_disabled_returns_empty():
    def handler(req):  # pragma: no cover
        raise AssertionError("disabled drive client pruned")
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = DriveClient(settings=_settings(drive_enabled=False), http=http)
    try:
        assert await client.prune_stale_pdfs(
            "drive-tok", FOLDER_ID, set(),
        ) == []
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# Audit events
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_prune_audit_ok_no_stale(tmp_path):
    store = SessionStore(str(tmp_path / "a.db"))
    try:
        def handler(req):
            return httpx.Response(200, json={"files": [
                {"id": "fid-keep", "name": "keep.pdf"},
            ]})
        client = _client(handler, audit_store=store)
        try:
            await client.prune_stale_pdfs(
                "drive-tok", FOLDER_ID, {"keep.pdf"}, tenant="acme",
            )
        finally:
            await client.aclose()
        events = store.list_audit_events(event_type="drive.prune")
        assert len(events) == 1
        p = events[0].payload
        assert p["status"] == "ok"
        assert p["deleted"] == 0
        assert p["considered"] == 1
        assert events[0].tenant == "acme"
    finally:
        store.close()


@pytest.mark.asyncio
async def test_prune_audit_ok_with_deletes(tmp_path):
    store = SessionStore(str(tmp_path / "a.db"))
    try:
        def handler(req):
            if req.method == "GET":
                return httpx.Response(200, json={"files": [
                    {"id": "fid-keep",  "name": "keep.pdf"},
                    {"id": "fid-stale", "name": "stale.pdf"},
                ]})
            return httpx.Response(204)
        client = _client(handler, audit_store=store)
        try:
            await client.prune_stale_pdfs(
                "drive-tok", FOLDER_ID, {"keep.pdf"}, tenant="acme",
            )
        finally:
            await client.aclose()
        events = store.list_audit_events(event_type="drive.prune")
        p = events[0].payload
        assert p["status"] == "ok"
        assert p["deleted"] == 1
        assert p["failed"] == 0
        assert p["kept"] == 1
    finally:
        store.close()


@pytest.mark.asyncio
async def test_prune_audit_partial(tmp_path):
    store = SessionStore(str(tmp_path / "a.db"))
    try:
        def handler(req):
            if req.method == "GET":
                return httpx.Response(200, json={"files": [
                    {"id": "fid-ok",   "name": "a.pdf"},
                    {"id": "fid-fail", "name": "b.pdf"},
                ]})
            fid = str(req.url).rsplit("/", 1)[-1]
            return httpx.Response(204 if fid == "fid-ok" else 500)
        client = _client(handler, audit_store=store)
        try:
            await client.prune_stale_pdfs(
                "drive-tok", FOLDER_ID, set(), tenant="acme",
            )
        finally:
            await client.aclose()
        events = store.list_audit_events(event_type="drive.prune")
        p = events[0].payload
        assert p["status"] == "partial"
        assert p["deleted"] == 1
        assert p["failed"]  == 1
    finally:
        store.close()
