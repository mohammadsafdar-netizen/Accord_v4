"""Live-integration smoke test for BackendClient + DriveClient against
real copilot.inevo.ai + real Google Drive.

Skipped unless:
  ACCORD_RUN_INTEGRATION=1 in env
  BACKEND_CLIENT_SECRET set
  TEST_TENANT_SLUG set (e.g. "acme-staging")
  TEST_TENANT_DOMAIN set (e.g. "acme-staging.brocopilot.com")
  TEST_SUBMISSION_ID set — a submission pre-created in the staging backend

Cleanup: the test creates a submission folder in Drive and uploads a
single-page test PDF. It does NOT clean up on exit by design — so you can
visually verify the result in the staging Drive folder. Re-running the
test is idempotent (upload overwrites) but leaves the folder.

What this catches that MockTransport cannot:
  * OAuth scope misconfigurations (real 403 vs mock 200)
  * Backend response shape drift (fields we assumed exist that don't)
  * Drive API version differences (quota, rate-limit responses)
  * Real TLS / certificate / DNS issues
  * Multipart boundary or Content-Type nuances the Drive API rejects

On failure, the test dumps the full wire traffic (request + response
bodies) to stdout so you can diff against MockTransport expectations.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import httpx
import pytest

from accord_ai.config import Settings
from accord_ai.integrations.backend import BackendClient
from accord_ai.integrations.drive import (
    DriveAuthError,
    DriveClient,
    UploadResult,
)


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Env-driven config
# ---------------------------------------------------------------------------

_INTEGRATION_ENABLED = os.environ.get("ACCORD_RUN_INTEGRATION") == "1"
_REQUIRED_VARS = (
    "BACKEND_CLIENT_SECRET",
    "TEST_TENANT_SLUG",
    "TEST_TENANT_DOMAIN",
    "TEST_SUBMISSION_ID",
)


def _missing_vars() -> List[str]:
    return [v for v in _REQUIRED_VARS if not os.environ.get(v)]


if not _INTEGRATION_ENABLED:
    pytest.skip(
        "integration tests disabled (set ACCORD_RUN_INTEGRATION=1)",
        allow_module_level=True,
    )

_MISSING = _missing_vars()
if _MISSING:
    pytest.skip(
        f"integration env vars missing: {_MISSING}",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Wire-trace observer — every request + response body captured for diff
# ---------------------------------------------------------------------------

@dataclass
class WireEvent:
    step:       str
    method:     str
    url:        str
    status:     Optional[int] = None
    req_body:   Optional[str] = None
    resp_body:  Optional[str] = None
    error:      Optional[str] = None


@dataclass
class WireTrace:
    events: List[WireEvent] = field(default_factory=list)

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(json.dumps(asdict(e)) for e in self.events)
        )


@pytest.fixture
def trace() -> WireTrace:
    t = WireTrace()
    yield t
    # Always dump, pass or fail — diff source
    dump_dir = Path("logs") / "smoke"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dump_path = dump_dir / f"smoke_{ts}.jsonl"
    t.dump(dump_path)
    print(f"\n[smoke] wire trace → {dump_path}")
    for e in t.events:
        if e.error or (e.status and e.status >= 400):
            print(
                f"  [{e.step}] {e.method} {e.url} → "
                f"{e.status} {e.error or ''}"
            )


# Wire request/response bodies into httpx via event hooks.
def _install_tracing(
    http: httpx.AsyncClient, trace: WireTrace, step_name: str,
) -> None:
    async def _log_request(req: httpx.Request) -> None:
        try:
            body = req.content.decode("utf-8", errors="replace")[:2000]
        except Exception:
            body = "<binary>"
        trace.events.append(WireEvent(
            step=step_name, method=req.method, url=str(req.url),
            req_body=body,
        ))

    async def _log_response(resp: httpx.Response) -> None:
        await resp.aread()
        try:
            body = resp.text[:2000]
        except Exception:
            body = "<binary>"
        # Patch the last event in place
        if trace.events:
            trace.events[-1].status = resp.status_code
            trace.events[-1].resp_body = body

    http.event_hooks["request"]  = [_log_request]
    http.event_hooks["response"] = [_log_response]


# ---------------------------------------------------------------------------
# Settings + clients
# ---------------------------------------------------------------------------

def _settings() -> Settings:
    return Settings(
        backend_enabled=True,
        backend_host_suffix="copilot.inevo.ai",
        backend_client_id=os.environ.get("BACKEND_CLIENT_ID", "intake-agent"),
        backend_client_secret=os.environ["BACKEND_CLIENT_SECRET"],
        backend_timeout_s=15.0,
        backend_tls_verify=True,
        backend_token_ttl_s=600,
        drive_enabled=True,
        drive_api_base="https://www.googleapis.com",
        drive_timeout_s=30.0,
    )


# ---------------------------------------------------------------------------
# The five-step chain
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_live_five_step_chain(trace: WireTrace):
    settings = _settings()
    tenant_slug   = os.environ["TEST_TENANT_SLUG"]
    tenant_domain = os.environ["TEST_TENANT_DOMAIN"]
    submission_id = os.environ["TEST_SUBMISSION_ID"]

    # ---- Step 1: service token ----
    backend_http = httpx.AsyncClient(verify=True)
    _install_tracing(backend_http, trace, "service_token")
    backend = BackendClient(settings=settings, http=backend_http)
    try:
        service_token = await backend.get_service_token(
            tenant_slug, tenant_domain,
        )
    finally:
        await backend.aclose()
    assert service_token, (
        f"service token failed — check BACKEND_CLIENT_SECRET + tenant setup. "
        f"Last wire event: "
        f"{trace.events[-1] if trace.events else 'none'}"
    )
    assert isinstance(service_token, str) and len(service_token) > 10

    # ---- Step 2: drive token (fresh client so tracing doesn't intermix) ----
    backend_http2 = httpx.AsyncClient(verify=True)
    _install_tracing(backend_http2, trace, "drive_token")
    backend2 = BackendClient(settings=settings, http=backend_http2)
    try:
        drive_token = await backend2.get_drive_token(
            service_token, submission_id, tenant_domain,
        )
    finally:
        await backend2.aclose()
    assert drive_token, (
        "drive token failed — OAuth scope may not cover Drive access, or "
        "backend's domain-wide delegation is misconfigured"
    )

    # ---- Step 3: agent folder ----
    backend_http3 = httpx.AsyncClient(verify=True)
    _install_tracing(backend_http3, trace, "agent_folder")
    backend3 = BackendClient(settings=settings, http=backend_http3)
    try:
        agent_folder = await backend3.get_agent_folder(
            service_token, submission_id, tenant_domain,
        )
    finally:
        await backend3.aclose()
    assert isinstance(agent_folder, dict), "agent_folder must be dict"
    assert "id" in agent_folder, (
        f"MockTransport assumed agent_folder['id'] exists; real response: "
        f"{agent_folder}. Update DriveClient or the mocks."
    )
    parent_folder_id = agent_folder["id"]

    # ---- Step 4: submission folder (find-or-create) ----
    drive_http = httpx.AsyncClient(verify=True)
    _install_tracing(drive_http, trace, "submission_folder")
    drive = DriveClient(settings=settings, http=drive_http)
    # Keep drive open — step 5 reuses the same client.
    submission_folder_id = await drive.find_or_create_submission_folder(
        drive_token, submission_id, parent_folder_id, tenant=tenant_slug,
    )
    assert submission_folder_id, "submission folder create/find failed"

    # ---- Step 5: upload a small PDF ----
    _install_tracing(drive._http, trace, "upload")   # reuse the same client
    test_pdf = _build_minimal_pdf()
    try:
        result = await drive.upload_filled_pdf(
            drive_token, submission_folder_id,
            "smoke_test.pdf", test_pdf,
            tenant=tenant_slug,
        )
    except DriveAuthError as e:
        pytest.fail(
            f"drive upload 401 — token from step 2 rejected: {e}"
        )
    finally:
        await drive.aclose()

    assert isinstance(result, UploadResult)
    assert result.file_id, "upload returned no file_id"
    assert result.view_url.startswith("https://drive.google.com/")

    # ---- Cross-validation: every response shape matched assumptions ----
    print(f"\n[smoke] ✓ 5-step chain complete")
    print(f"  service_token length: {len(service_token)}")
    print(f"  drive_token length:   {len(drive_token)}")
    print(f"  agent_folder keys:    {list(agent_folder.keys())}")
    print(f"  submission_folder_id: {submission_folder_id}")
    print(f"  uploaded_file_id:     {result.file_id}")
    print(f"  view_url:             {result.view_url}")


@pytest.mark.asyncio
async def test_agent_folder_response_shape_matches_mocks(trace: WireTrace):
    """Isolated response-shape assertion — runs even if the 5-step chain
    has a downstream failure, so you always learn what the real payload
    looks like."""
    settings = _settings()
    tenant_slug   = os.environ["TEST_TENANT_SLUG"]
    tenant_domain = os.environ["TEST_TENANT_DOMAIN"]
    submission_id = os.environ["TEST_SUBMISSION_ID"]

    http = httpx.AsyncClient(verify=True)
    _install_tracing(http, trace, "shape_probe")
    backend = BackendClient(settings=settings, http=http)
    try:
        token = await backend.get_service_token(tenant_slug, tenant_domain)
        if not token:
            pytest.skip(
                "cannot validate shapes — service token failed",
            )

        folder = await backend.get_agent_folder(
            token, submission_id, tenant_domain,
        )
        if folder is None:
            pytest.skip(
                "agent folder returned None — nothing to shape-check",
            )

        # Dump the ACTUAL fields so they can be diffed against DriveClient's
        # assumptions. MockTransport tests only assert .get("id") exists.
        print(
            f"\n[smoke] agent_folder real fields: {list(folder.keys())}"
        )

        # Hard-require id; soft-flag anything unexpected.
        assert "id" in folder
        known = {"id", "name", "webViewLink", "parents", "mimeType"}
        unknown = set(folder.keys()) - known
        if unknown:
            print(
                f"[smoke] ⚠ agent_folder has fields not modeled in "
                f"mocks: {unknown}"
            )
    finally:
        await backend.aclose()


# ---------------------------------------------------------------------------
# Minimal PDF builder (no PyMuPDF dependency for this specific test)
# ---------------------------------------------------------------------------

def _build_minimal_pdf() -> bytes:
    """Smallest-possible valid PDF — 1-page, no content, no form fields.
    Used to exercise the upload path without pulling the full fill pipeline.
    """
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
        b"xref\n0 4\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"trailer\n<< /Size 4 /Root 1 0 R >>\n"
        b"startxref\n178\n%%EOF\n"
    )
