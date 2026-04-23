"""Drive-upload orchestration for /complete (P10.C.6).

Coordinates BackendClient + DriveClient to:
  1. Mint a service token for the tenant
  2. Exchange it for a Drive access token
  3. Resolve the tenant's agent/LOB folder
  4. Create (or find) a per-submission folder under it
  5. Upload every filled PDF — PATCH over prior file_id when one exists,
     POST fresh otherwise
  6. Prune stale files left over from a previous LOB
  7. Persist the Drive file IDs on the FilledPdfStore manifest so the next
     /complete can PATCH-in-place

Error philosophy: Drive is an accessory path. The local /pdf fallback
serves PDFs when Drive is unreachable. This function MUST NOT raise —
every per-form failure resolves to a drive_status explaining what
happened, and the call returns a summary with counts.

The only exception that propagates into retry logic is DriveAuthError;
for every other failure the DriveClient returns None and we record
"failed" on the affected form(s).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from accord_ai.audit import DRIVE_PIPELINE, record_audit_event
from accord_ai.forms.pipeline import FilledForm
from accord_ai.forms.storage import FilledPdfStore
from accord_ai.integrations.backend import BackendClient
from accord_ai.integrations.drive import DriveAuthError, DriveClient
from accord_ai.logging_config import get_logger

_logger = get_logger("forms.drive_pipeline")

# Module-level LOB folder cache: tenant_slug -> (lob_folder_id, expires_at).
# Avoids one backend.get_agent_folder() round-trip per /complete call for
# repeat submissions on the same tenant. 1h TTL matches folder stability.
_LOB_FOLDER_CACHE: Dict[str, Tuple[str, float]] = {}
_LOB_FOLDER_TTL = 3600.0


# Status tokens surfaced on each CompleteFormInfo and summarized in the audit
# event. "skipped" means the whole pipeline short-circuited (disabled,
# missing tenant_domain, token exchange failed). Per-form failures show up
# as "failed" or "auth_failed".
DRIVE_STATUS_SKIPPED:     str = "skipped"
DRIVE_STATUS_UPLOADED:    str = "uploaded"
DRIVE_STATUS_OVERWRITTEN: str = "overwritten"
DRIVE_STATUS_FAILED:      str = "failed"
DRIVE_STATUS_AUTH_FAILED: str = "auth_failed"


@dataclass(frozen=True)
class DriveFormOutcome:
    """Per-form Drive outcome — mapped onto CompleteFormInfo by the caller."""
    form_number:    str
    drive_status:   str
    drive_file_id:  Optional[str] = None
    drive_view_url: Optional[str] = None


@dataclass(frozen=True)
class DriveUploadOutcome:
    """Aggregate outcome returned from `run_drive_upload_pipeline`.

    Fields:
      drive_enabled: True when the pipeline attempted any Drive work —
        False only when the caller skipped us entirely (no clients wired,
        no tenant_domain). When True but no uploads happened (e.g. service
        token minting failed), per-form outcomes all carry "skipped".
      drive_folder_id: the resolved per-submission folder id, when we got
        that far. None when we short-circuited before folder resolution.
      pruned_count: number of stale files successfully deleted from the
        submission folder. 0 on short-circuit or on prune errors.
      form_outcomes: in-order list, one per FilledForm the caller passed.
    """
    drive_enabled:   bool
    drive_folder_id: Optional[str]
    pruned_count:    int
    form_outcomes:   List[DriveFormOutcome]


def _all_skipped(
    filled: Dict[str, FilledForm], reason: str, tenant: Optional[str],
    submission_id: str, audit_store, *, drive_enabled: bool,
) -> DriveUploadOutcome:
    """Build a uniform all-skipped outcome + emit the summary audit event."""
    outcomes = [
        DriveFormOutcome(form_number=fn, drive_status=DRIVE_STATUS_SKIPPED)
        for fn in filled.keys()
    ]
    record_audit_event(
        audit_store,
        DRIVE_PIPELINE,
        tenant=tenant,
        payload={
            "submission_id": submission_id,
            "status":        "skipped",
            "reason":        reason,
            "forms_count":   len(filled),
            "uploaded":      0,
            "overwritten":   0,
            "failed":        0,
            "auth_failed":   0,
            "pruned_count":  0,
        },
    )
    return DriveUploadOutcome(
        drive_enabled=drive_enabled,
        drive_folder_id=None,
        pruned_count=0,
        form_outcomes=outcomes,
    )


async def run_drive_upload_pipeline(
    *,
    backend:          BackendClient,
    drive:            DriveClient,
    filled_pdf_store: FilledPdfStore,
    submission_id:    str,
    tenant:           Optional[str],
    tenant_domain:    str,
    tenant_slug:      str,
    filled:           Dict[str, FilledForm],
    fields_data:      Optional[Dict[str, Dict[str, object]]] = None,
    audit_store=None,
) -> DriveUploadOutcome:
    """Run the full Drive upload pipeline for one /complete invocation.

    See module docstring for the flow. The function is "fail-soft": every
    per-form failure is recorded as a drive_status on the outcome. Only
    DriveAuthError triggers the token re-exchange + retry path; all other
    errors come back from the DriveClient as None and become "failed".

    Callers pass `filled` from `fill_submission` — the function does not
    mutate those objects. `fields_data` is forwarded verbatim to
    backend.push_fields after uploads complete; it's optional so unit
    tests can exercise the upload path without wiring FE payloads.
    """
    # Short-circuit 1: no tenant_domain — v3 subdomain routing needs it,
    # and without it we can't even mint the service token.
    if not tenant_domain:
        return _all_skipped(
            filled, "missing_tenant_domain", tenant,
            submission_id, audit_store, drive_enabled=False,
        )

    # Short-circuit 2: service token failed. We DID attempt Drive (the
    # caller wanted it), so drive_enabled stays True; per-form statuses
    # still all show skipped because nothing uploaded.
    service_token = await backend.get_service_token(
        tenant_slug, tenant_domain,
    )
    if service_token is None:
        return _all_skipped(
            filled, "no_service_token", tenant,
            submission_id, audit_store, drive_enabled=True,
        )

    # Short-circuit 3: Drive token exchange failed.
    drive_token = await backend.get_drive_token(
        service_token, submission_id, tenant_domain,
    )
    if drive_token is None:
        await _push_fields_best_effort_async(
            backend, service_token, submission_id, tenant_domain, fields_data,
        )
        return _all_skipped(
            filled, "no_drive_token", tenant,
            submission_id, audit_store, drive_enabled=True,
        )

    # Short-circuit 4: no agent folder. The backend owns the per-tenant
    # "where do filled forms go" decision; if it doesn't answer we don't
    # guess. push_fields still runs — field sync is independent of Drive.
    # v3 wire contract: the backend returns this folder id under the key
    # `lob_folder_id` (not `id`); accept `id` as a forward-compat fallback.
    #
    # LOB folder is stable per-tenant (changes only on tenant re-provisioning).
    # Cache by tenant_slug for 1h to skip the backend round-trip on repeat calls.
    _now = time.monotonic()
    _cache_entry = _LOB_FOLDER_CACHE.get(tenant_slug)
    if _cache_entry is not None and _now < _cache_entry[1]:
        lob_folder_id = _cache_entry[0]
    else:
        agent_folder = await backend.get_agent_folder(
            service_token, submission_id, tenant_domain,
        )
        lob_folder_id = None
        if isinstance(agent_folder, dict):
            lob_folder_id = (
                agent_folder.get("lob_folder_id") or agent_folder.get("id")
            )
        if lob_folder_id:
            _LOB_FOLDER_CACHE[tenant_slug] = (lob_folder_id, _now + _LOB_FOLDER_TTL)
    if not lob_folder_id:
        await _push_fields_best_effort_async(
            backend, service_token, submission_id, tenant_domain, fields_data,
        )
        return _all_skipped(
            filled, "no_agent_folder", tenant,
            submission_id, audit_store, drive_enabled=True,
        )

    # Short-circuit 5: couldn't create/find the submission folder.
    submission_folder_id = await drive.find_or_create_submission_folder(
        drive_token, submission_id, lob_folder_id, tenant=tenant,
    )
    if submission_folder_id is None:
        await _push_fields_best_effort_async(
            backend, service_token, submission_id, tenant_domain, fields_data,
        )
        return _all_skipped(
            filled, "no_submission_folder", tenant,
            submission_id, audit_store, drive_enabled=True,
        )

    # --- Per-form uploads -------------------------------------------------
    outcomes: List[DriveFormOutcome] = []
    counts = {
        DRIVE_STATUS_UPLOADED:    0,
        DRIVE_STATUS_OVERWRITTEN: 0,
        DRIVE_STATUS_FAILED:      0,
        DRIVE_STATUS_AUTH_FAILED: 0,
    }

    for form_number, ff in filled.items():
        # v3 wire contract (verified drive_upload_v2.py:480): Drive filename
        # convention is `{form_number}-Form.pdf` (e.g. "125-Form.pdf"). This
        # must match v3 exactly — v3-era folders already contain files under
        # this name and a mismatch breaks both dedup (existing_file_id probe
        # by name) and prune (keep-set comparison).
        file_name = f"{form_number}-Form.pdf"
        existing_id = filled_pdf_store.get_drive_file_id(
            submission_id, tenant, form_number,
        )

        status, result, drive_token = await _upload_with_auth_retry(
            drive=drive,
            backend=backend,
            service_token=service_token,
            submission_id=submission_id,
            tenant=tenant,
            tenant_domain=tenant_domain,
            drive_token=drive_token,
            folder_id=submission_folder_id,
            file_name=file_name,
            pdf_bytes=ff.pdf_bytes,
            existing_id=existing_id,
        )

        if result is not None:
            filled_pdf_store.set_drive_file_id(
                submission_id, tenant, form_number, result.file_id,
            )
            outcomes.append(DriveFormOutcome(
                form_number=form_number,
                drive_status=status,
                drive_file_id=result.file_id,
                drive_view_url=result.view_url,
            ))
        else:
            outcomes.append(DriveFormOutcome(
                form_number=form_number, drive_status=status,
            ))
        counts[status] += 1

    # --- Prune stale files ------------------------------------------------
    # Keep-set filenames must match the v3 convention used in upload above.
    keep = {f"{n}-Form.pdf" for n in filled.keys()}
    pruned_count = 0
    try:
        deleted = await drive.prune_stale_pdfs(
            drive_token, submission_folder_id, keep, tenant=tenant,
        )
        pruned_count = len(deleted)
    except DriveAuthError:
        # Token went bad mid-pipeline — non-fatal; stale files stay and
        # a subsequent /complete will retry prune.
        _logger.warning(
            "drive_pipeline: prune auth_failed submission=%s",
            submission_id,
        )
        pruned_count = 0

    # --- Push fields to backend (non-fatal) ------------------------------
    await _push_fields_best_effort_async(
        backend, service_token, submission_id, tenant_domain, fields_data,
    )

    # --- Summary audit ----------------------------------------------------
    record_audit_event(
        audit_store,
        DRIVE_PIPELINE,
        tenant=tenant,
        payload={
            "submission_id":  submission_id,
            "status":         "ok",
            "folder_id":      submission_folder_id,
            "forms_count":    len(filled),
            "uploaded":       counts[DRIVE_STATUS_UPLOADED],
            "overwritten":    counts[DRIVE_STATUS_OVERWRITTEN],
            "failed":         counts[DRIVE_STATUS_FAILED],
            "auth_failed":    counts[DRIVE_STATUS_AUTH_FAILED],
            "pruned_count":   pruned_count,
        },
    )

    return DriveUploadOutcome(
        drive_enabled=True,
        drive_folder_id=submission_folder_id,
        pruned_count=pruned_count,
        form_outcomes=outcomes,
    )


async def _upload_with_auth_retry(
    *,
    drive:         DriveClient,
    backend:       BackendClient,
    service_token: str,
    submission_id: str,
    tenant:        Optional[str],
    tenant_domain: str,
    drive_token:   str,
    folder_id:     str,
    file_name:     str,
    pdf_bytes:     bytes,
    existing_id:   Optional[str],
):
    """Upload one file; re-exchange the Drive token once on DriveAuthError.

    Returns (status, upload_result_or_None, current_drive_token). The
    returned drive_token reflects any mid-call re-exchange so the caller
    uses the fresh token for subsequent uploads / prune.
    """
    try:
        result = await drive.upload_filled_pdf(
            drive_token, folder_id, file_name, pdf_bytes,
            existing_file_id=existing_id, tenant=tenant,
        )
    except DriveAuthError:
        # One re-exchange, one retry. Spec mandates exactly one — a second
        # 401 means the token exchange itself is degraded; degrade gracefully.
        new_token = await backend.get_drive_token(
            service_token, submission_id, tenant_domain,
        )
        if new_token and new_token != drive_token:
            drive_token = new_token
            try:
                result = await drive.upload_filled_pdf(
                    drive_token, folder_id, file_name, pdf_bytes,
                    existing_file_id=existing_id, tenant=tenant,
                )
            except DriveAuthError:
                return DRIVE_STATUS_AUTH_FAILED, None, drive_token
        else:
            return DRIVE_STATUS_AUTH_FAILED, None, drive_token

    if result is None:
        return DRIVE_STATUS_FAILED, None, drive_token

    status = (
        DRIVE_STATUS_OVERWRITTEN if existing_id else DRIVE_STATUS_UPLOADED
    )
    return status, result, drive_token


async def _push_fields_best_effort_async(
    backend:       BackendClient,
    service_token: str,
    submission_id: str,
    tenant_domain: str,
    fields_data:   Optional[Dict[str, Dict[str, object]]],
) -> None:
    """Call backend.push_fields — swallow failures, never propagate.

    Drive and field-sync are independent paths. A broken field sync
    shouldn't prevent Drive uploads from landing, and vice versa.
    """
    try:
        await backend.push_fields(
            service_token, submission_id, tenant_domain,
            fields_data or {},
            completion_percentage=100.0,
            source="finalize",
        )
    except (RuntimeError, ValueError) as exc:
        # BackendClient swallows its own http errors and returns False;
        # this catch is defense-in-depth for any programmer error that
        # might leak out. Never fail /complete on field-sync woes.
        _logger.warning(
            "drive_pipeline: push_fields raised err=%s submission=%s",
            exc, submission_id,
        )


