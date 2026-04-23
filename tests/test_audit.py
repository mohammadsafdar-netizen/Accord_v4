"""Tests for the audit log foundation (P10.0.c)."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import pytest

from accord_ai.audit import (
    AUTH_FAILURE,
    SESSION_CREATED,
    SUBMISSION_UPDATED,
    record_audit_event,
)
from accord_ai.core.store import SessionStore
from accord_ai.request_context import clear_context, set_context


@pytest.fixture
def store(tmp_path):
    s = SessionStore(str(tmp_path / "audit.db"))
    yield s
    s.close()


@pytest.fixture
def audit_caplog(caplog):
    """caplog that reliably captures accord_ai.audit records.

    configure_logging() (called by build_fastapi_app in other test files)
    sets propagate=False on the 'accord_ai' logger — which lingers at
    process level and defeats propagation-based caplog. Attaching the
    caplog handler directly to 'accord_ai.audit' works regardless.
    """
    audit_logger = logging.getLogger("accord_ai.audit")
    audit_logger.addHandler(caplog.handler)
    original_level = audit_logger.level
    audit_logger.setLevel(logging.DEBUG)
    try:
        yield caplog
    finally:
        audit_logger.removeHandler(caplog.handler)
        audit_logger.setLevel(original_level)


# --- Migration ---------------------------------------------------------------

def test_migration_creates_audit_events_table(tmp_path):
    s = SessionStore(str(tmp_path / "m.db"))
    try:
        rows = s._conn().execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='audit_events'"
        ).fetchall()
        assert len(rows) == 1
        version = s._conn().execute("PRAGMA user_version").fetchone()[0]
        assert version == 5
    finally:
        s.close()


def test_migration_is_idempotent(tmp_path):
    path = str(tmp_path / "idem.db")
    SessionStore(path).close()
    # Second open must not re-run migration 3 (would fail: "table already exists")
    s2 = SessionStore(path)
    try:
        version = s2._conn().execute("PRAGMA user_version").fetchone()[0]
        assert version == 5
    finally:
        s2.close()


# --- Insert + retrieve -------------------------------------------------------

def test_insert_and_list_roundtrip(store):
    clear_context()
    audit_id = record_audit_event(
        store,
        SESSION_CREATED,
        tenant="acme",
        session_id="sess-1",
        request_id="req-1",
        payload={"actor": "api", "n": 3},
    )
    assert audit_id and len(audit_id) == 32

    events = store.list_audit_events()
    assert len(events) == 1
    e = events[0]
    assert e.audit_id == audit_id
    assert e.event_type == SESSION_CREATED
    assert e.tenant == "acme"
    assert e.session_id == "sess-1"
    assert e.request_id == "req-1"
    assert e.payload == {"actor": "api", "n": 3}
    assert isinstance(e.created_at, datetime)


# --- Filtering ---------------------------------------------------------------

def test_list_filters_by_event_type_and_tenant(store):
    clear_context()
    record_audit_event(store, SESSION_CREATED, tenant="acme")
    record_audit_event(store, SESSION_CREATED, tenant="globex")
    record_audit_event(store, AUTH_FAILURE,    tenant="acme")
    record_audit_event(store, SUBMISSION_UPDATED, tenant="acme", session_id="s1")

    acme_created = store.list_audit_events(
        tenant="acme", event_type=SESSION_CREATED,
    )
    assert len(acme_created) == 1
    assert acme_created[0].tenant == "acme"

    by_session = store.list_audit_events(session_id="s1")
    assert len(by_session) == 1
    assert by_session[0].event_type == SUBMISSION_UPDATED


def test_list_after_filter_is_exclusive(store):
    clear_context()
    record_audit_event(store, SESSION_CREATED, tenant="t")
    # Capture a timestamp strictly between the two inserts.
    time.sleep(0.01)
    cutoff = datetime.now(timezone.utc)
    time.sleep(0.01)
    record_audit_event(store, SESSION_CREATED, tenant="t")

    newer = store.list_audit_events(after=cutoff)
    assert len(newer) == 1


def test_list_limit_and_order(store):
    clear_context()
    for i in range(5):
        record_audit_event(store, SESSION_CREATED, payload={"i": i})

    top3 = store.list_audit_events(limit=3)
    assert len(top3) == 3
    # Most-recent-first.
    assert [e.payload["i"] for e in top3] == [4, 3, 2]


# --- ContextVar capture ------------------------------------------------------

def test_context_vars_auto_populate(store):
    clear_context()
    set_context(
        request_id="req-ctx", tenant="ctx-tenant", session_id="ctx-sess",
    )
    try:
        record_audit_event(store, SESSION_CREATED)
    finally:
        clear_context()

    e = store.list_audit_events()[0]
    assert e.tenant == "ctx-tenant"
    assert e.session_id == "ctx-sess"
    assert e.request_id == "req-ctx"


def test_explicit_kwargs_override_context_vars(store):
    clear_context()
    set_context(
        request_id="req-ctx", tenant="ctx-tenant", session_id="ctx-sess",
    )
    try:
        record_audit_event(
            store,
            SESSION_CREATED,
            tenant="explicit",
            session_id="explicit-s",
            request_id="explicit-r",
        )
    finally:
        clear_context()

    e = store.list_audit_events()[0]
    assert e.tenant == "explicit"
    assert e.session_id == "explicit-s"
    assert e.request_id == "explicit-r"


# --- Robustness --------------------------------------------------------------

def test_empty_event_type_returns_none_and_logs(store, audit_caplog):
    result = record_audit_event(store, "")
    assert result is None
    assert store.list_audit_events() == []
    assert any(
        "empty event_type" in r.getMessage()
        for r in audit_caplog.records
    )


def test_insert_failure_is_swallowed(store, audit_caplog, monkeypatch):
    def _boom(*a, **kw):
        raise RuntimeError("disk full")
    monkeypatch.setattr(store, "insert_audit_event", _boom)

    result = record_audit_event(store, SESSION_CREATED)
    assert result is None
    assert any(
        "insert failed" in r.getMessage()
        for r in audit_caplog.records
    )


def test_payload_defaults_to_empty_object(store):
    clear_context()
    record_audit_event(store, SESSION_CREATED)
    e = store.list_audit_events()[0]
    assert e.payload == {}
