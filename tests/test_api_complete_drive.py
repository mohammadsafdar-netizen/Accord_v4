"""Tests for Drive-upload orchestration in /complete (P10.C.6).

The pipeline is exercised by substituting `intake.backend_client` and
`intake.drive_client` with AsyncMock objects that mimic the real clients'
returns. This keeps the tests fast and keeps them focused on the
orchestration logic (branching, retry, persistence) rather than on httpx
wiring — which is already covered exhaustively by the per-client test
modules.
"""
from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("fitz")

from accord_ai.api import build_fastapi_app
from accord_ai.app import build_intake_app
from accord_ai.config import Settings
from accord_ai.integrations.drive import DriveAuthError, UploadResult
from accord_ai.schema import (
    Address,
    CommercialAutoDetails,
    CustomerSubmission,
    Driver,
    GeneralLiabilityCoverage,
    GeneralLiabilityDetails,
    PolicyDates,
    Vehicle,
)
from accord_ai.testing import FakeEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _settings(tmp_path) -> Settings:
    return Settings(
        db_path=str(tmp_path / "accord.db"),
        filled_pdf_dir=str(tmp_path / "filled"),
        accord_auth_disabled=True,
        harness_max_refines=0,
    )


@pytest.fixture
def client_without_drive(tmp_path):
    """/complete client with backend_client + drive_client set to None."""
    settings = _settings(tmp_path)
    intake = build_intake_app(
        settings, engine=FakeEngine(), refiner_engine=FakeEngine(),
    )
    # Default settings → backend_enabled=False, drive_enabled=False →
    # the factories returned None. Assert that's still true so the test
    # doesn't silently cover the wrong branch.
    assert intake.backend_client is None
    assert intake.drive_client is None
    app = build_fastapi_app(settings, intake=intake)
    with TestClient(app) as c:
        yield c, intake


@pytest.fixture
def client_with_mocked_drive(tmp_path):
    """/complete client with backend_client + drive_client monkey-patched
    to AsyncMock instances. Each test customizes the mocks in-body."""
    settings = _settings(tmp_path)
    intake = build_intake_app(
        settings, engine=FakeEngine(), refiner_engine=FakeEngine(),
    )

    backend = AsyncMock()
    drive = AsyncMock()

    # Default happy-path returns; individual tests override what they need.
    backend.get_service_token.return_value = "svc-tok"
    backend.get_drive_token.return_value = "drv-tok"
    # v3 wire shape: backend returns {lob_folder_id: ...}, not {id: ...}.
    # Verified against accord_ai_v3/drive_upload_v2.py:405-407.
    backend.get_agent_folder.return_value = {"lob_folder_id": "agent-folder"}
    backend.push_fields.return_value = True

    drive.find_or_create_submission_folder.return_value = "sub-folder"
    drive.prune_stale_pdfs.return_value = []

    # IntakeApp is a frozen dataclass — mutate via object.__setattr__.
    object.__setattr__(intake, "backend_client", backend)
    object.__setattr__(intake, "drive_client", drive)

    app = build_fastapi_app(settings, intake=intake)
    with TestClient(app) as c:
        yield c, intake, backend, drive


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------

def _seed_ca_session(intake, tenant="acme") -> str:
    sid = intake.store.create_session(tenant=tenant)
    submission = CustomerSubmission(
        business_name="Acme Trucking",
        mailing_address=Address(
            line_one="123 Main", city="Austin", state="TX", zip_code="78701",
        ),
        policy_dates=PolicyDates(effective_date=date(2026, 5, 1)),
        lob_details=CommercialAutoDetails(
            drivers=[Driver(first_name="Alice", last_name="Jones")],
            vehicles=[Vehicle(year=2024, make="Freightliner", model="Cascadia")],
        ),
    )
    intake.store.update_submission(sid, submission, tenant=tenant)
    return sid


def _seed_gl_session(intake, tenant="acme") -> str:
    sid = intake.store.create_session(tenant=tenant)
    intake.store.update_submission(
        sid,
        CustomerSubmission(
            business_name="GlobeX",
            lob_details=GeneralLiabilityDetails(
                coverage=GeneralLiabilityCoverage(
                    each_occurrence_limit=1_000_000,
                ),
            ),
        ),
        tenant=tenant,
    )
    return sid


def _complete(c, sid, *, tenant_domain="acme.brocopilot.com"):
    return c.post(
        "/complete",
        json={
            "submission_id": sid,
            "tenant_slug":   "acme",
            "tenant_domain": tenant_domain,
        },
    )


# ---------------------------------------------------------------------------
# Drive disabled — the baseline /complete must still work
# ---------------------------------------------------------------------------

def test_complete_with_drive_disabled_skips_upload_and_still_succeeds(
    client_without_drive,
):
    c, intake = client_without_drive
    sid = _seed_ca_session(intake)

    resp = _complete(c, sid)
    assert resp.status_code == 200
    body = resp.json()
    assert body["drive_enabled"] is False
    assert body["drive_folder_id"] is None
    assert body["pruned_count"] == 0
    assert all(f["drive_status"] == "skipped" for f in body["forms"])
    assert all(f["drive_file_id"] is None for f in body["forms"])
    assert all(f["drive_view_url"] is None for f in body["forms"])


# ---------------------------------------------------------------------------
# Drive enabled — happy path
# ---------------------------------------------------------------------------

def test_complete_with_drive_enabled_uploads_all_forms(
    client_with_mocked_drive,
):
    c, intake, backend, drive = client_with_mocked_drive
    sid = _seed_ca_session(intake)

    # Each upload returns a unique file id so we can check per-form metadata.
    # v3 filename convention: `{form}-Form.pdf` (verified drive_upload_v2.py:480).
    def _upload(drive_token, folder_id, file_name, pdf_bytes, **_):
        form_num = file_name.replace("-Form.pdf", "")
        return UploadResult.from_id(f"fid-{form_num}")

    drive.upload_filled_pdf.side_effect = _upload

    resp = _complete(c, sid)
    assert resp.status_code == 200
    body = resp.json()
    assert body["drive_enabled"] is True
    assert body["drive_folder_id"] == "sub-folder"

    for f in body["forms"]:
        assert f["drive_status"] == "uploaded"
        assert f["drive_file_id"] == f"fid-{f['form_number']}"
        assert (
            f["drive_view_url"]
            == f"https://drive.google.com/file/d/fid-{f['form_number']}/view"
        )


def test_complete_persists_drive_file_ids_in_manifest(
    client_with_mocked_drive,
):
    c, intake, backend, drive = client_with_mocked_drive
    sid = _seed_ca_session(intake)

    # v3 filename convention: "{form}-Form.pdf" → extract form by splitting "-"
    drive.upload_filled_pdf.side_effect = lambda *a, **k: UploadResult.from_id(
        f"fid-{k.get('existing_file_id') or a[2].split('-')[0]}"
    )

    _complete(c, sid)

    # The manifest should now remember each upload's file_id.
    for form in ("125", "127", "129", "137", "163"):
        fid = intake.filled_pdf_store.get_drive_file_id(sid, "acme", form)
        assert fid == f"fid-{form}"


def test_complete_second_call_uses_existing_drive_file_id_for_patch(
    client_with_mocked_drive,
):
    c, intake, backend, drive = client_with_mocked_drive
    sid = _seed_ca_session(intake)

    # v3 filename convention: "{form}-Form.pdf" → extract form by splitting "-"
    drive.upload_filled_pdf.side_effect = lambda *a, **k: UploadResult.from_id(
        f"fid-{a[2].split('-')[0]}"
    )

    _complete(c, sid)

    # Reset call tracking for the second invocation only.
    drive.upload_filled_pdf.reset_mock()

    resp = _complete(c, sid)
    body = resp.json()

    # Every upload on the second call must carry the stashed existing_file_id.
    for call in drive.upload_filled_pdf.call_args_list:
        file_name = call.args[2]
        form_num = file_name.split("-")[0]
        assert call.kwargs.get("existing_file_id") == f"fid-{form_num}"

    assert all(f["drive_status"] == "overwritten" for f in body["forms"])


def test_complete_prunes_stale_files_on_lob_change(
    client_with_mocked_drive,
):
    c, intake, backend, drive = client_with_mocked_drive

    # First call: CA submission → 125,127,129,137,163
    sid = _seed_ca_session(intake)
    drive.upload_filled_pdf.side_effect = lambda *a, **k: UploadResult.from_id(
        "fid-anything"
    )
    _complete(c, sid)

    # Second call: swap submission to GL (keeps only 125,126).
    gl_submission = CustomerSubmission(
        business_name="Acme Pivot",
        lob_details=GeneralLiabilityDetails(
            coverage=GeneralLiabilityCoverage(each_occurrence_limit=500_000),
        ),
    )
    intake.store.update_submission(sid, gl_submission, tenant="acme")

    drive.prune_stale_pdfs.return_value = ["deleted-1", "deleted-2"]
    resp = _complete(c, sid)
    body = resp.json()

    # Verify prune was called with the new LOB's keep-set.
    last_prune_call = drive.prune_stale_pdfs.call_args
    keep = last_prune_call.args[2]
    # v3 filename convention: `{form}-Form.pdf` (verified drive_upload_v2.py:480).
    assert keep == {"125-Form.pdf", "126-Form.pdf"}
    assert body["pruned_count"] == 2


# ---------------------------------------------------------------------------
# DriveAuthError retry behavior
# ---------------------------------------------------------------------------

def test_complete_drive_auth_error_reexchanges_and_retries(
    client_with_mocked_drive,
):
    c, intake, backend, drive = client_with_mocked_drive
    sid = _seed_ca_session(intake)

    # First call: return the initial token. After that: return a fresh one.
    backend.get_drive_token.side_effect = ["drv-tok-1", "drv-tok-2"] + ["drv-tok-2"] * 10

    call_count = {"n": 0}

    def _upload(drive_token, folder_id, file_name, pdf_bytes, **_):
        call_count["n"] += 1
        # Only the very first upload 401s — everything else succeeds.
        if call_count["n"] == 1:
            raise DriveAuthError("401 on first upload")
        return UploadResult.from_id(f"fid-{file_name}")

    drive.upload_filled_pdf.side_effect = _upload

    resp = _complete(c, sid)
    body = resp.json()

    # get_drive_token called at least twice: initial + re-exchange.
    assert backend.get_drive_token.call_count >= 2
    # All forms still show uploaded.
    assert all(f["drive_status"] == "uploaded" for f in body["forms"])


def test_complete_drive_auth_error_retry_also_fails_reports_auth_failed(
    client_with_mocked_drive,
):
    c, intake, backend, drive = client_with_mocked_drive
    sid = _seed_ca_session(intake)

    backend.get_drive_token.side_effect = ["drv-tok-1", "drv-tok-2"] + ["drv-tok-2"] * 10

    # First form: both tries raise. Remaining forms: succeed.
    call_count = {"n": 0}

    def _upload(drive_token, folder_id, file_name, pdf_bytes, **_):
        call_count["n"] += 1
        if file_name == "125-Form.pdf":
            raise DriveAuthError("401 always for 125")
        return UploadResult.from_id(f"fid-{file_name}")

    drive.upload_filled_pdf.side_effect = _upload

    resp = _complete(c, sid)
    body = resp.json()

    form_125 = next(f for f in body["forms"] if f["form_number"] == "125")
    assert form_125["drive_status"] == "auth_failed"
    assert form_125["drive_file_id"] is None

    # Other forms uploaded normally.
    others = [f for f in body["forms"] if f["form_number"] != "125"]
    assert all(f["drive_status"] == "uploaded" for f in others)


# ---------------------------------------------------------------------------
# Per-form hard-fail (None return, no 401)
# ---------------------------------------------------------------------------

def test_complete_drive_hard_fail_still_returns_local_results(
    client_with_mocked_drive, tmp_path,
):
    c, intake, backend, drive = client_with_mocked_drive
    sid = _seed_ca_session(intake)

    drive.upload_filled_pdf.side_effect = lambda *a, **k: None

    resp = _complete(c, sid)
    assert resp.status_code == 200
    body = resp.json()

    # Drive was attempted but every upload failed.
    assert body["drive_enabled"] is True
    assert all(f["drive_status"] == "failed" for f in body["forms"])
    assert all(f["drive_file_id"] is None for f in body["forms"])

    # Local PDFs were still saved — /pdf must return bytes.
    pdf_resp = c.get(f"/pdf/{sid}/125", headers={"x-tenant-slug": "acme"})
    assert pdf_resp.status_code == 200
    assert pdf_resp.content.startswith(b"%PDF-")


# ---------------------------------------------------------------------------
# Pipeline short-circuit paths
# ---------------------------------------------------------------------------

def test_complete_missing_tenant_domain_skips_drive(
    client_with_mocked_drive,
):
    c, intake, backend, drive = client_with_mocked_drive
    sid = _seed_ca_session(intake)

    resp = c.post(
        "/complete",
        json={
            "submission_id": sid, "tenant_slug": "acme",
            "tenant_domain": "",
        },
    )
    body = resp.json()
    assert resp.status_code == 200
    assert body["drive_enabled"] is False
    assert all(f["drive_status"] == "skipped" for f in body["forms"])

    # We short-circuited before any backend/drive calls.
    backend.get_service_token.assert_not_called()
    drive.upload_filled_pdf.assert_not_called()


def test_complete_missing_service_token_skips_drive(
    client_with_mocked_drive,
):
    """When service-token minting fails, drive_enabled stays True (we tried)
    but no uploads happen; per-form drive_status is 'skipped'."""
    c, intake, backend, drive = client_with_mocked_drive
    sid = _seed_ca_session(intake)

    backend.get_service_token.return_value = None

    resp = _complete(c, sid)
    body = resp.json()

    assert resp.status_code == 200
    # We attempted Drive — drive_enabled reflects intent, not success.
    assert body["drive_enabled"] is True
    assert body["drive_folder_id"] is None
    assert all(f["drive_status"] == "skipped" for f in body["forms"])
    drive.upload_filled_pdf.assert_not_called()


# ---------------------------------------------------------------------------
# backend.push_fields integration
# ---------------------------------------------------------------------------

def test_complete_calls_backend_push_fields_after_uploads(
    client_with_mocked_drive,
):
    c, intake, backend, drive = client_with_mocked_drive
    sid = _seed_ca_session(intake)

    drive.upload_filled_pdf.side_effect = lambda *a, **k: UploadResult.from_id(
        "fid-xyz"
    )

    fields_payload = {"125": {"business_name": "Acme Trucking"}}
    resp = c.post(
        "/complete",
        json={
            "submission_id": sid,
            "tenant_slug":   "acme",
            "tenant_domain": "acme.brocopilot.com",
            "fields_data":   fields_payload,
        },
    )
    assert resp.status_code == 200

    backend.push_fields.assert_called_once()
    call = backend.push_fields.call_args
    # Positional args: (service_token, submission_id, tenant_domain, fields_data)
    assert call.args[0] == "svc-tok"
    assert call.args[1] == sid
    assert call.args[2] == "acme.brocopilot.com"
    assert call.args[3] == fields_payload
    assert call.kwargs["completion_percentage"] == 100.0
    assert call.kwargs["source"] == "finalize"


def test_complete_push_fields_failure_does_not_fail_response(
    client_with_mocked_drive,
):
    c, intake, backend, drive = client_with_mocked_drive
    sid = _seed_ca_session(intake)

    drive.upload_filled_pdf.side_effect = lambda *a, **k: UploadResult.from_id(
        "fid-xyz"
    )
    backend.push_fields.return_value = False  # backend says "no"

    resp = _complete(c, sid)
    assert resp.status_code == 200
    body = resp.json()
    # All forms still uploaded successfully.
    assert all(f["drive_status"] == "uploaded" for f in body["forms"])
