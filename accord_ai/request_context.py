"""Per-request contextvars for request_id / tenant / session_id.

Contextvars are per-asyncio-task. A log line emitted inside one coroutine
carries only that task's context, even under asyncio.gather — the core
requirement for multi-tenant isolation.
"""
from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar
from typing import Optional

_request_id: ContextVar[Optional[str]] = ContextVar("accord_request_id", default=None)
_tenant: ContextVar[Optional[str]] = ContextVar("accord_tenant", default=None)
_session_id: ContextVar[Optional[str]] = ContextVar("accord_session_id", default=None)


def new_request_id() -> str:
    """12-hex-char id — short enough for human-readable logs, wide enough for uniqueness."""
    return uuid.uuid4().hex[:12]


def set_context(
    *,
    request_id: Optional[str] = None,
    tenant: Optional[str] = None,
    session_id: Optional[str] = None,
) -> None:
    """Set one or more context fields. Unprovided fields are left unchanged."""
    if request_id is not None:
        _request_id.set(request_id)
    if tenant is not None:
        _tenant.set(tenant)
    if session_id is not None:
        _session_id.set(session_id)


def clear_context() -> None:
    """Reset all context fields to None."""
    _request_id.set(None)
    _tenant.set(None)
    _session_id.set(None)


def get_request_id() -> Optional[str]:
    return _request_id.get()


def get_tenant() -> Optional[str]:
    return _tenant.get()


def get_session_id() -> Optional[str]:
    return _session_id.get()


class ContextInjectionFilter(logging.Filter):
    """Attach request_id / tenant / session_id to every LogRecord.

    Runs as a logging.Filter so the values are available to any formatter
    (text, JSON, whatever). Missing values render as '-'.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get() or "-"
        record.tenant = _tenant.get() or "-"
        record.session_id = _session_id.get() or "-"
        return True
