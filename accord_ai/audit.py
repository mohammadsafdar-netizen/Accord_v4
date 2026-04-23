"""Audit events — append-only structured trail for security/compliance.

Foundation module: helpers that pull request-context from contextvars and
record events via SessionStore. No consumers yet — individual write paths
get instrumented as Phase 10.A/B/C land.

Design:
  - Best-effort: record_audit_event swallows storage errors and logs them
    rather than failing the caller's operation. Audit must never block.
  - Event types are free-form strings by convention; the constants below
    are the canonical taxonomy — rename one and you break every consumer
    grepping the audit table.
  - Contextvars (request_id, tenant, session_id) auto-populate when not
    passed explicitly. Explicit kwargs always win.
"""
from __future__ import annotations

from typing import Optional

from accord_ai.core.store import AuditEvent, SessionStore  # noqa: F401
from accord_ai.logging_config import get_logger
from accord_ai.request_context import (
    get_request_id,
    get_session_id,
    get_tenant,
)

_logger = get_logger("audit")


# --- Canonical event types ---------------------------------------------------
#
# Keep this list short and stable. Consumers grep for these strings; renaming
# one is a breaking change for anyone parsing the audit table.

SESSION_CREATED        = "session.created"
SESSION_FINALIZED      = "session.finalized"
SUBMISSION_UPDATED     = "submission.updated"
SUBMISSION_COMPLETED   = "submission.completed"
COMPLETE_OVERRIDES_APPLIED = "complete.overrides_applied"
AUTH_FAILURE           = "auth.failure"
CONCURRENCY_CONFLICT   = "concurrency.conflict"
BACKEND_TOKEN_ISSUED   = "backend.token_issued"
BACKEND_PUSH_FIELDS    = "backend.push_fields"
BACKEND_READ_FIELDS    = "backend.read_fields"
BACKEND_DRIVE_TOKEN    = "backend.drive_token"
BACKEND_AGENT_FOLDER   = "backend.agent_folder"
DRIVE_FOLDER_RESOLVED  = "drive.submission_folder_resolved"
DRIVE_UPLOAD           = "drive.upload"
DRIVE_PRUNE            = "drive.prune"
DRIVE_PIPELINE         = "drive.pipeline"


def record_audit_event(
    store: SessionStore,
    event_type: str,
    *,
    tenant: Optional[str] = None,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None,
    payload: Optional[dict] = None,
) -> Optional[str]:
    """Record one audit event. Returns audit_id, or None if storage failed.

    Unset kwargs are filled from request contextvars. Empty-string values
    (either explicit or resolved from contextvars) normalize back to None
    for storage — audit rows never carry "" as a meaningful tenant/session.
    """
    if not event_type:
        _logger.error("audit: empty event_type — dropping")
        return None

    resolved_tenant     = tenant     if tenant     is not None else get_tenant()
    resolved_session    = session_id if session_id is not None else get_session_id()
    resolved_request_id = request_id if request_id is not None else get_request_id()

    # Normalize empty-string sentinels back to None for storage.
    resolved_tenant     = resolved_tenant     or None
    resolved_session    = resolved_session    or None
    resolved_request_id = resolved_request_id or None

    try:
        return store.insert_audit_event(
            event_type,
            tenant=resolved_tenant,
            session_id=resolved_session,
            request_id=resolved_request_id,
            payload=payload,
        )
    except Exception as exc:
        # Best-effort — never fail the caller because audit couldn't write.
        _logger.exception(
            "audit: insert failed event_type=%s err=%s", event_type, exc,
        )
        return None
