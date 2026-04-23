"""Google Drive client — direct API calls for folder + file operations.

Token minting lives on BackendClient (get_drive_token / get_agent_folder)
because the tenant's backend holds Google's domain-wide delegation, not us.
This module handles the Drive-direct parts: finding or creating submission
folders, uploading files (10.C.4), pruning stale ones (10.C.5).

Design mirrors BackendClient:
  * httpx.AsyncClient constructor-injected → tests use MockTransport
  * Every failure mode swallowed → None return + audit event
  * Opt-in via settings.drive_enabled; build_drive_client returns None
    when disabled

Drive query-string escaping: Drive's q-parameter uses a custom syntax
where single quotes delimit values. Caller-supplied names must have their
single quotes backslash-escaped or the query becomes malformed (and Drive
silently returns empty results instead of 400-ing).
"""
from __future__ import annotations

import asyncio as _asyncio
import json as _json
import os as _os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import httpx

from accord_ai.http_client import get_client as _get_http_client

from accord_ai.audit import record_audit_event
from accord_ai.config import Settings
from accord_ai.core.store import SessionStore
from accord_ai.logging_config import get_logger

_logger = get_logger("integrations.drive")


# Drive q-parameter value escaping: backslash first, then single quote.
# Order matters — if we did quotes first, the backslashes we inserted
# would themselves get doubled in the backslash-escape step.
def _escape_q_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


# ---------------------------------------------------------------------------
# Upload result + auth-failure signal (P10.C.4)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UploadResult:
    file_id:          str
    view_url:         str
    web_content_link: str

    @classmethod
    def from_id(cls, file_id: str) -> "UploadResult":
        return cls(
            file_id=file_id,
            view_url=f"https://drive.google.com/file/d/{file_id}/view",
            web_content_link=(
                f"https://drive.google.com/uc?export=download&id={file_id}"
            ),
        )


class DriveAuthError(RuntimeError):
    """Drive returned 401. Caller should re-exchange the service token
    (via BackendClient.get_drive_token) and retry the upload.

    The ONLY failure mode in DriveClient that raises — every other failure
    returns None so orchestrators can silently degrade to the local /pdf
    fallback. Auth failure is distinct because it has a concrete recovery.
    """


class DriveClient:
    """Thin Google Drive v3 client.

    Everything returns None on failure; nothing raises. Drive is an
    accessory path — if it's down the local /pdf fallback keeps brokers
    unblocked. Callers check return values rather than try/except.
    """

    FOLDER_MIME = "application/vnd.google-apps.folder"

    def __init__(
        self,
        *,
        settings: Settings,
        http: httpx.AsyncClient,
        audit_store: Optional[SessionStore] = None,
    ) -> None:
        self._settings = settings
        self._http = http
        self._audit_store = audit_store
        self._base = settings.drive_api_base.rstrip("/")

    async def aclose(self) -> None:
        pass  # shared client lifecycle is managed by the FastAPI lifespan

    # --- Folder operations ---

    async def find_or_create_submission_folder(
        self,
        drive_token: str,
        submission_id: str,
        parent_folder_id: str,
        *,
        tenant: Optional[str] = None,
    ) -> Optional[str]:
        """Return the Drive folder ID for this submission under parent.

        Two paths:
          1. Query for an existing folder with the name `submission_id`
             under `parent_folder_id`. If present, return its id.
          2. Otherwise POST a create request, return the new folder's id.

        Both operations are idempotent from the caller's perspective —
        repeat calls return the same folder id for a given submission.
        A 401 on either path returns None (token expired or misconfigured);
        callers can detect and re-exchange.

        `tenant` is only used for audit tagging; the Drive API itself
        is tenant-agnostic at the transport layer.
        """
        if not self._settings.drive_enabled or not drive_token:
            return None

        # Step 1 — query for existing folder.
        escaped_name = _escape_q_value(submission_id)
        escaped_parent = _escape_q_value(parent_folder_id)
        q = (
            f"name='{escaped_name}' and "
            f"'{escaped_parent}' in parents and "
            f"mimeType='{self.FOLDER_MIME}' and "
            f"trashed=false"
        )

        try:
            resp = await self._http.get(
                f"{self._base}/drive/v3/files",
                params={
                    "q": q,
                    "fields": "files(id,name)",
                    "spaces": "drive",
                },
                headers={"authorization": f"Bearer {drive_token}"},
                timeout=self._settings.drive_timeout_s,
            )
        except httpx.HTTPError as exc:
            _logger.warning("drive.find_folder: network err=%s", exc)
            self._audit_folder(tenant, submission_id, "error", {
                "stage": "query", "error_class": type(exc).__name__,
            })
            return None

        if resp.status_code == 401:
            self._audit_folder(
                tenant, submission_id, "auth_failed", {"stage": "query"},
            )
            return None
        if resp.status_code == 200:
            try:
                files = resp.json().get("files", [])
            except ValueError:
                files = []
            if files and isinstance(files[0], dict) and files[0].get("id"):
                folder_id = files[0]["id"]
                self._audit_folder(
                    tenant, submission_id, "found", {"folder_id": folder_id},
                )
                return folder_id
        elif resp.status_code not in (200, 404):
            # Non-404/401 query failure — don't blindly create (would risk
            # duplicate folders if the backend is flaky). Bail.
            _logger.warning(
                "drive.find_folder: query http=%s", resp.status_code,
            )
            self._audit_folder(tenant, submission_id, "http_error", {
                "stage": "query", "http_status": resp.status_code,
            })
            return None

        # Step 2 — create.
        try:
            resp = await self._http.post(
                f"{self._base}/drive/v3/files",
                json={
                    "name":     submission_id,
                    "mimeType": self.FOLDER_MIME,
                    "parents":  [parent_folder_id],
                },
                headers={
                    "authorization": f"Bearer {drive_token}",
                    "content-type":  "application/json",
                },
                timeout=self._settings.drive_timeout_s,
            )
        except httpx.HTTPError as exc:
            _logger.warning("drive.create_folder: network err=%s", exc)
            self._audit_folder(tenant, submission_id, "error", {
                "stage": "create", "error_class": type(exc).__name__,
            })
            return None

        if resp.status_code == 401:
            self._audit_folder(
                tenant, submission_id, "auth_failed", {"stage": "create"},
            )
            return None
        if resp.status_code in (200, 201):
            try:
                new_id = resp.json().get("id")
            except ValueError:
                new_id = None
            if isinstance(new_id, str) and new_id:
                self._audit_folder(
                    tenant, submission_id, "created", {"folder_id": new_id},
                )
                return new_id

        _logger.warning(
            "drive.create_folder: http=%s", resp.status_code,
        )
        self._audit_folder(tenant, submission_id, "http_error", {
            "stage": "create", "http_status": resp.status_code,
        })
        return None

    # --- File upload (P10.C.4) ---

    async def upload_filled_pdf(
        self,
        drive_token: str,
        folder_id: str,
        file_name: str,
        pdf_bytes: bytes,
        *,
        existing_file_id: Optional[str] = None,
        tenant: Optional[str] = None,
    ) -> Optional[UploadResult]:
        """Upload (or overwrite) a PDF in `folder_id`.

        Routing:
          existing_file_id given → PATCH /upload/drive/v3/files/{fid}
              On 404 → POST (file was deleted externally)
          existing_file_id None + probe finds match → PATCH with found id
          existing_file_id None + probe empty      → POST fresh

        Raises DriveAuthError on 401 — caller re-exchanges and retries.
        All other failures return None.
        """
        if not self._settings.drive_enabled or not drive_token:
            return None

        fid = existing_file_id

        # --- Step 1: probe by name when no fid provided.
        if fid is None:
            try:
                fid = await self._probe_file_id(
                    drive_token, folder_id, file_name,
                )
            except DriveAuthError:
                self._audit_upload(
                    tenant, file_name, "auth_failed", {"stage": "probe"},
                )
                raise

        # --- Step 2: upload.
        body, headers = self._build_multipart(
            pdf_bytes, file_name, folder_id, is_update=bool(fid),
        )
        headers["authorization"] = f"Bearer {drive_token}"

        try:
            if fid:
                resp = await self._http.patch(
                    f"{self._base}/upload/drive/v3/files/{fid}"
                    f"?uploadType=multipart",
                    content=body,
                    headers=headers,
                    timeout=self._settings.drive_timeout_s,
                )
                if resp.status_code == 404:
                    # File was externally deleted. Re-upload as POST — the
                    # metadata shape differs (must include parents).
                    body, headers = self._build_multipart(
                        pdf_bytes, file_name, folder_id, is_update=False,
                    )
                    headers["authorization"] = f"Bearer {drive_token}"
                    resp = await self._http.post(
                        f"{self._base}/upload/drive/v3/files"
                        f"?uploadType=multipart",
                        content=body,
                        headers=headers,
                        timeout=self._settings.drive_timeout_s,
                    )
                    # Recover: the PATCH-404 fallback succeeded as a fresh
                    # create. Forget the stale fid so the audit correctly
                    # reports "ok" (created fresh), not "overwritten".
                    fid = None
            else:
                resp = await self._http.post(
                    f"{self._base}/upload/drive/v3/files"
                    f"?uploadType=multipart",
                    content=body,
                    headers=headers,
                    timeout=self._settings.drive_timeout_s,
                )
        except httpx.HTTPError as exc:
            _logger.warning("drive.upload: network err=%s", exc)
            self._audit_upload(tenant, file_name, "error", {
                "error_class": type(exc).__name__,
            })
            return None

        if resp.status_code == 401:
            self._audit_upload(
                tenant, file_name, "auth_failed", {"stage": "upload"},
            )
            raise DriveAuthError(f"drive upload 401 for {file_name!r}")

        if resp.status_code in (200, 201):
            try:
                returned_id = resp.json().get("id")
            except ValueError:
                returned_id = None
            final_id = returned_id or fid
            if not isinstance(final_id, str) or not final_id:
                self._audit_upload(tenant, file_name, "missing_id", {})
                return None
            self._audit_upload(
                tenant, file_name,
                "overwritten" if fid else "ok", {
                    "file_id":   final_id,
                    "byte_size": len(pdf_bytes),
                },
            )
            return UploadResult.from_id(final_id)

        _logger.warning(
            "drive.upload: http=%s name=%s", resp.status_code, file_name,
        )
        self._audit_upload(tenant, file_name, "http_error", {
            "http_status": resp.status_code,
        })
        return None

    # --- List + prune (P10.C.5) ---

    async def list_folder_children(
        self,
        drive_token: str,
        folder_id: str,
    ) -> Dict[str, str]:
        """Return {file_name: file_id} for all non-folder, non-trashed files.

        Used for two purposes:
          1. Warm the file-id cache so uploads skip the per-file probe.
          2. Snapshot for prune_stale_pdfs below.

        Returns {} on any failure — caller treats empty as "nothing to
        reconcile" which is safe for both use cases. Raises DriveAuthError
        on 401 so callers can re-exchange (same pattern as upload).

        Filter excludes mimeType=application/vnd.google-apps.folder so
        subfolders never land in the keep-or-delete decision.
        """
        if not self._settings.drive_enabled or not drive_token:
            return {}

        q = (
            f"'{_escape_q_value(folder_id)}' in parents and "
            f"trashed=false and "
            f"mimeType!='{self.FOLDER_MIME}'"
        )
        try:
            resp = await self._http.get(
                f"{self._base}/drive/v3/files",
                params={
                    "q":        q,
                    "fields":   "files(id,name)",
                    "spaces":   "drive",
                    "pageSize": "1000",
                },
                headers={"authorization": f"Bearer {drive_token}"},
                timeout=self._settings.drive_timeout_s,
            )
        except httpx.HTTPError as exc:
            _logger.warning("drive.list_children: network err=%s", exc)
            return {}

        if resp.status_code == 401:
            raise DriveAuthError(f"drive list 401 for folder {folder_id!r}")
        if resp.status_code != 200:
            _logger.warning("drive.list_children: http=%s", resp.status_code)
            return {}

        try:
            files = resp.json().get("files", [])
        except ValueError:
            return {}

        out: Dict[str, str] = {}
        for f in files:
            if not isinstance(f, dict):
                continue
            name, fid = f.get("name"), f.get("id")
            if (
                isinstance(name, str) and isinstance(fid, str)
                and name and fid
            ):
                out[name] = fid
        return out

    async def prune_stale_pdfs(
        self,
        drive_token: str,
        folder_id: str,
        keep_file_names: Set[str],
        *,
        tenant: Optional[str] = None,
        folder_snapshot: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        """Delete files in `folder_id` whose names are NOT in `keep_file_names`.

        Use case: LOB drop-shift (CA → GL) leaves stale form PDFs in the
        submission folder. Caller passes the canonical keep-set from the
        current LOB's forms_for_lob list.

        Returns list of deleted file_ids (only the ones that actually went).
        Per-file delete failures are logged + audited but don't abort the
        batch — Drive occasionally returns 500 on a single delete while
        others succeed.

        `folder_snapshot` lets the caller skip the list call when they
        already have the {name: id} map from a prior list_folder_children
        (typical for the /complete orchestrator that also uses the snapshot
        for upload probe-skip).

        401 on the initial list raises DriveAuthError — caller re-exchanges
        token and retries. 401 on an individual delete within the batch is
        logged + counted as failure; the list succeeded so the token was
        valid at start. Per-file 401 is more likely a permission issue than
        a token-expiry, so we don't raise mid-batch and orphan the rest.
        """
        if not self._settings.drive_enabled or not drive_token:
            return []

        snapshot = folder_snapshot
        if snapshot is None:
            # list_folder_children may raise DriveAuthError — propagate.
            snapshot = await self.list_folder_children(
                drive_token, folder_id,
            )

        to_delete = [
            (name, fid) for name, fid in snapshot.items()
            if name not in keep_file_names
        ]
        if not to_delete:
            self._audit_prune(tenant, folder_id, {
                "status":     "ok",
                "deleted":    0,
                "considered": len(snapshot),
                "kept":       len(snapshot),
                "failed":     0,
            })
            return []

        async def _delete_one(name: str, fid: str) -> Optional[str]:
            try:
                resp = await self._http.delete(
                    f"{self._base}/drive/v3/files/{fid}",
                    headers={"authorization": f"Bearer {drive_token}"},
                    timeout=self._settings.drive_timeout_s,
                )
            except httpx.HTTPError as exc:
                _logger.warning(
                    "drive.prune: delete network err file=%s err=%s",
                    name, exc,
                )
                return None
            if resp.status_code in (200, 204):
                return fid
            _logger.warning(
                "drive.prune: delete http=%s file=%s",
                resp.status_code, name,
            )
            return None

        results = await _asyncio.gather(
            *[_delete_one(name, fid) for name, fid in to_delete],
            return_exceptions=False,
        )
        deleted = [fid for fid in results if fid]

        self._audit_prune(tenant, folder_id, {
            "status":     "ok" if len(deleted) == len(to_delete) else "partial",
            "deleted":    len(deleted),
            "failed":     len(to_delete) - len(deleted),
            "considered": len(snapshot),
            "kept":       len(snapshot) - len(to_delete),
        })
        return deleted

    def _audit_prune(
        self, tenant: Optional[str], folder_id: str, payload: dict,
    ) -> None:
        if self._audit_store is None:
            return
        record_audit_event(
            self._audit_store,
            "drive.prune",
            tenant=tenant,
            payload={"folder_id": folder_id, **payload},
        )

    # --- Internal helpers ---

    async def _probe_file_id(
        self, drive_token: str, folder_id: str, file_name: str,
    ) -> Optional[str]:
        """Return the file id for `file_name` in `folder_id`, or None.

        Raises DriveAuthError on 401 so upload_filled_pdf can surface the
        signal without swallowing it as 'no existing file, POST fresh'.
        """
        q = (
            f"name='{_escape_q_value(file_name)}' and "
            f"'{_escape_q_value(folder_id)}' in parents and "
            f"trashed=false"
        )
        try:
            resp = await self._http.get(
                f"{self._base}/drive/v3/files",
                params={"q": q, "fields": "files(id)", "spaces": "drive"},
                headers={"authorization": f"Bearer {drive_token}"},
                timeout=self._settings.drive_timeout_s,
            )
        except httpx.HTTPError:
            return None
        if resp.status_code == 401:
            raise DriveAuthError(f"drive probe 401 for {file_name!r}")
        if resp.status_code != 200:
            return None
        try:
            files = resp.json().get("files", [])
        except ValueError:
            return None
        if files and isinstance(files[0], dict):
            fid = files[0].get("id")
            return fid if isinstance(fid, str) and fid else None
        return None

    @staticmethod
    def _build_multipart(
        pdf_bytes: bytes,
        file_name: str,
        folder_id: str,
        *,
        is_update: bool,
    ) -> Tuple[bytes, dict]:
        """Build the multipart/related body Drive's upload endpoint expects.

        PATCH (update) metadata carries ONLY name — parents are immutable
        on update. POST (create) metadata carries name + parents so Drive
        knows where to place the new file.
        """
        boundary = "----Boundary" + _os.urandom(8).hex()
        metadata = (
            {"name": file_name} if is_update
            else {"name": file_name, "parents": [folder_id]}
        )
        body = b"".join([
            f"--{boundary}\r\n".encode(),
            b"Content-Type: application/json; charset=UTF-8\r\n\r\n",
            _json.dumps(metadata).encode(),
            f"\r\n--{boundary}\r\n".encode(),
            b"Content-Type: application/pdf\r\n\r\n",
            pdf_bytes,
            f"\r\n--{boundary}--\r\n".encode(),
        ])
        headers = {
            "content-type": f"multipart/related; boundary={boundary}",
        }
        return body, headers

    def _audit_upload(
        self,
        tenant: Optional[str],
        file_name: str,
        status: str,
        extra: dict,
    ) -> None:
        if self._audit_store is None:
            return
        record_audit_event(
            self._audit_store,
            "drive.upload",
            tenant=tenant,
            payload={"file_name": file_name, "status": status, **extra},
        )

    # --- Internal: folder audit ---

    def _audit_folder(
        self,
        tenant: Optional[str],
        submission_id: str,
        status: str,
        extra: dict,
    ) -> None:
        if self._audit_store is None:
            return
        record_audit_event(
            self._audit_store,
            "drive.submission_folder_resolved",
            tenant=tenant,
            payload={
                "submission_id": submission_id,
                "status": status,
                **extra,
            },
        )


def build_drive_client(
    settings: Settings,
    *,
    audit_store: Optional[SessionStore] = None,
) -> Optional[DriveClient]:
    """Return a ready-to-use DriveClient, or None when disabled."""
    if not settings.drive_enabled:
        return None
    http = _get_http_client()
    return DriveClient(
        settings=settings, http=http, audit_store=audit_store,
    )
