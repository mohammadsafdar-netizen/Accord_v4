import sqlite3
import threading
import uuid as _uuid
from dataclasses import FrozenInstanceError
from datetime import datetime

import pytest

from accord_ai.core.store import Message, Session, SessionStore
from accord_ai.schema import CustomerSubmission


# --- schema / open ---

def test_store_creates_db_file(tmp_path):
    db = tmp_path / "test.db"
    assert not db.exists()
    SessionStore(str(db))
    assert db.exists()


def test_sessions_table_exists(tmp_path):
    db = tmp_path / "test.db"
    SessionStore(str(db))
    with sqlite3.connect(str(db)) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"
        ).fetchone()
    assert row is not None


def test_sessions_table_has_expected_columns(tmp_path):
    db = tmp_path / "test.db"
    SessionStore(str(db))
    with sqlite3.connect(str(db)) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(sessions)").fetchall()}
    assert {"session_id", "tenant", "created_at", "updated_at", "status", "submission_json"}.issubset(cols)


def test_wal_journal_mode_enabled(tmp_path):
    db = tmp_path / "test.db"
    SessionStore(str(db))
    with sqlite3.connect(str(db)) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_indexes_created(tmp_path):
    db = tmp_path / "test.db"
    SessionStore(str(db))
    with sqlite3.connect(str(db)) as conn:
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='sessions'"
        ).fetchall()}
    assert "idx_sessions_tenant" in names
    assert "idx_sessions_status" in names


def test_open_twice_is_idempotent(tmp_path):
    db = tmp_path / "test.db"
    SessionStore(str(db))
    SessionStore(str(db))


def test_conn_is_per_thread(tmp_path):
    store = SessionStore(str(tmp_path / "test.db"))
    main_conn = store._conn()
    sibling: list = []

    def in_other_thread():
        sibling.append(store._conn())

    t = threading.Thread(target=in_other_thread)
    t.start()
    t.join()
    assert sibling[0] is not main_conn


def test_close_releases_current_thread_conn(tmp_path):
    store = SessionStore(str(tmp_path / "test.db"))
    _ = store._conn()
    store.close()
    assert getattr(store._local, "conn", None) is None


# --- migrations + pragmas ---

def test_migration_runner_sets_user_version(tmp_path):
    db = tmp_path / "db.sqlite"
    SessionStore(str(db))
    with sqlite3.connect(str(db)) as conn:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
    assert version == 5   # migrations 1–5 applied


def test_foreign_keys_pragma_is_on(tmp_path):
    store = SessionStore(str(tmp_path / "db.sqlite"))
    value = store._conn().execute("PRAGMA foreign_keys").fetchone()[0]
    assert value == 1


def test_check_constraint_rejects_invalid_status(tmp_path):
    db = tmp_path / "db.sqlite"
    store = SessionStore(str(db))
    sid = store.create_session()
    with sqlite3.connect(str(db)) as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE sessions SET status='gibberish' WHERE session_id=?", (sid,))


# --- create_session ---

def test_create_session_returns_hex_id(tmp_path):
    store = SessionStore(str(tmp_path / "db.sqlite"))
    sid = store.create_session()
    assert len(sid) == 32 and all(c in "0123456789abcdef" for c in sid)


def test_create_session_persists_row(tmp_path):
    db = tmp_path / "db.sqlite"
    store = SessionStore(str(db))
    sid = store.create_session()
    with sqlite3.connect(str(db)) as conn:
        row = conn.execute(
            "SELECT session_id, status, submission_json FROM sessions WHERE session_id = ?",
            (sid,),
        ).fetchone()
    assert row is not None and row[1] == "active" and row[2]


def test_create_session_stores_tenant(tmp_path):
    store = SessionStore(str(tmp_path / "db.sqlite"))
    sid = store.create_session(tenant="acme")
    assert store.get_session(sid).tenant == "acme"


def test_create_session_produces_unique_ids(tmp_path):
    store = SessionStore(str(tmp_path / "db.sqlite"))
    assert len({store.create_session() for _ in range(10)}) == 10


# --- get_session ---

def test_get_session_returns_none_for_missing_id(tmp_path):
    store = SessionStore(str(tmp_path / "db.sqlite"))
    assert store.get_session("nope") is None


def test_get_session_returns_session_for_existing_id(tmp_path):
    store = SessionStore(str(tmp_path / "db.sqlite"))
    sid = store.create_session(tenant="acme")
    s = store.get_session(sid)
    assert isinstance(s, Session) and s.tenant == "acme" and s.status == "active"


def test_get_session_empty_submission_is_valid_default(tmp_path):
    store = SessionStore(str(tmp_path / "db.sqlite"))
    s = store.get_session(store.create_session())
    assert isinstance(s.submission, CustomerSubmission)
    assert s.submission.business_name is None


def test_get_session_timestamps_are_utc_datetimes(tmp_path):
    store = SessionStore(str(tmp_path / "db.sqlite"))
    s = store.get_session(store.create_session())
    assert isinstance(s.created_at, datetime) and s.created_at.tzinfo is not None


def test_session_is_frozen(tmp_path):
    store = SessionStore(str(tmp_path / "db.sqlite"))
    s = store.get_session(store.create_session())
    with pytest.raises(FrozenInstanceError):
        s.status = "finalized"


# --- tenant isolation ---

def test_get_session_scoped_by_tenant(tmp_path):
    store = SessionStore(str(tmp_path / "db.sqlite"))
    sid_a = store.create_session(tenant="acme")
    sid_b = store.create_session(tenant="globex")
    assert store.get_session(sid_a, tenant="acme") is not None
    assert store.get_session(sid_b, tenant="acme") is None
    assert store.get_session(sid_a) is not None
    assert store.get_session(sid_b) is not None


def test_update_submission_scoped_by_tenant(tmp_path):
    store = SessionStore(str(tmp_path / "db.sqlite"))
    sid = store.create_session(tenant="acme")
    with pytest.raises(KeyError):
        store.update_submission(sid, CustomerSubmission(business_name="pwned"), tenant="globex")
    store.update_submission(sid, CustomerSubmission(business_name="ok"), tenant="acme")
    assert store.get_session(sid).submission.business_name == "ok"


# --- update_submission ---

def test_update_submission_persists_changes(tmp_path):
    store = SessionStore(str(tmp_path / "db.sqlite"))
    sid = store.create_session()
    sub = CustomerSubmission(
        business_name="Acme", ein="12-3456789",
        lob_details={"lob": "commercial_auto", "drivers": [{"first_name": "Alice"}]},
    )
    store.update_submission(sid, sub)
    restored = store.get_session(sid)
    assert restored.submission.lob_details.lob == "commercial_auto"
    assert restored.submission.lob_details.drivers[0].first_name == "Alice"


def test_update_submission_bumps_updated_at(tmp_path, monkeypatch):
    import accord_ai.core.store as store_module
    times = iter([
        "2026-04-18T10:00:00.000000+00:00",
        "2026-04-18T10:00:01.000000+00:00",
    ])
    monkeypatch.setattr(store_module, "_now_utc_iso", lambda: next(times))

    store = SessionStore(str(tmp_path / "db.sqlite"))
    sid = store.create_session()
    before = store.get_session(sid).updated_at
    store.update_submission(sid, CustomerSubmission(business_name="Acme"))
    after = store.get_session(sid).updated_at
    assert after > before


def test_update_submission_raises_for_missing_session(tmp_path):
    store = SessionStore(str(tmp_path / "db.sqlite"))
    with pytest.raises(KeyError):
        store.update_submission("no-such-id", CustomerSubmission())


# --- messages: migration 2 schema ---

def test_messages_table_exists(tmp_path):
    db = tmp_path / "test.db"
    SessionStore(str(db))
    with sqlite3.connect(str(db)) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='messages'"
        ).fetchone()
    assert row is not None


def test_messages_has_expected_columns(tmp_path):
    db = tmp_path / "test.db"
    SessionStore(str(db))
    with sqlite3.connect(str(db)) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(messages)").fetchall()}
    assert {"message_id", "session_id", "created_at", "role", "content"}.issubset(cols)


def test_fk_prevents_orphan_message_insert(tmp_path):
    """Direct SQL insert with a nonexistent session_id must fail — FK enforced."""
    db = tmp_path / "test.db"
    store = SessionStore(str(db))
    conn = store._conn()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO messages (message_id, session_id, created_at, role, content) "
            "VALUES (?, ?, ?, ?, ?)",
            (_uuid.uuid4().hex, "no-such-session", "2026-04-18T10:00:00+00:00", "user", "x"),
        )


def test_check_constraint_rejects_invalid_role(tmp_path):
    db = tmp_path / "test.db"
    store = SessionStore(str(db))
    sid = store.create_session()
    conn = store._conn()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO messages (message_id, session_id, created_at, role, content) "
            "VALUES (?, ?, ?, ?, ?)",
            (_uuid.uuid4().hex, sid, "2026-04-18T10:00:00+00:00", "admin", "x"),
        )


# --- append_message ---

def test_append_message_returns_hex_id(tmp_path):
    store = SessionStore(str(tmp_path / "test.db"))
    sid = store.create_session()
    mid = store.append_message(sid, "user", "hello")
    assert len(mid) == 32 and all(c in "0123456789abcdef" for c in mid)


def test_append_message_persists_row(tmp_path):
    db = tmp_path / "test.db"
    store = SessionStore(str(db))
    sid = store.create_session()
    mid = store.append_message(sid, "user", "hello world")
    with sqlite3.connect(str(db)) as conn:
        row = conn.execute(
            "SELECT session_id, role, content FROM messages WHERE message_id=?",
            (mid,),
        ).fetchone()
    assert row == (sid, "user", "hello world")


def test_append_message_bumps_session_updated_at(tmp_path, monkeypatch):
    import accord_ai.core.store as store_module
    times = iter([
        "2026-04-18T10:00:00.000000+00:00",  # create_session
        "2026-04-18T10:00:05.000000+00:00",  # append_message
    ])
    monkeypatch.setattr(store_module, "_now_utc_iso", lambda: next(times))

    store = SessionStore(str(tmp_path / "test.db"))
    sid = store.create_session()
    before = store.get_session(sid).updated_at
    store.append_message(sid, "user", "hi")
    after = store.get_session(sid).updated_at
    assert after > before


def test_append_to_nonexistent_session_raises(tmp_path):
    store = SessionStore(str(tmp_path / "test.db"))
    with pytest.raises(KeyError):
        store.append_message("no-such-id", "user", "hello")


def test_append_message_scoped_by_tenant(tmp_path):
    """Wrong-tenant append raises KeyError — no silent cross-tenant write."""
    store = SessionStore(str(tmp_path / "test.db"))
    sid = store.create_session(tenant="acme")
    with pytest.raises(KeyError):
        store.append_message(sid, "user", "stealth", tenant="globex")
    store.append_message(sid, "user", "legit", tenant="acme")


# --- get_messages ---

def test_get_messages_empty_for_new_session(tmp_path):
    store = SessionStore(str(tmp_path / "test.db"))
    sid = store.create_session()
    assert store.get_messages(sid) == []


def test_get_messages_returns_empty_for_missing_session(tmp_path):
    """Read on a nonexistent session = empty list, not an exception."""
    store = SessionStore(str(tmp_path / "test.db"))
    assert store.get_messages("no-such-id") == []


def test_get_messages_returns_in_insert_order(tmp_path, monkeypatch):
    """Oldest-first ordering. Clock monkeypatched so timestamps differ deterministically."""
    import accord_ai.core.store as store_module
    times = iter([
        "2026-04-18T10:00:00.000000+00:00",  # create_session
        "2026-04-18T10:00:01.000000+00:00",  # append 1
        "2026-04-18T10:00:02.000000+00:00",  # append 2
        "2026-04-18T10:00:03.000000+00:00",  # append 3
    ])
    monkeypatch.setattr(store_module, "_now_utc_iso", lambda: next(times))

    store = SessionStore(str(tmp_path / "test.db"))
    sid = store.create_session()
    store.append_message(sid, "user", "first")
    store.append_message(sid, "assistant", "second")
    store.append_message(sid, "user", "third")

    msgs = store.get_messages(sid)
    assert [m.content for m in msgs] == ["first", "second", "third"]
    assert [m.role for m in msgs] == ["user", "assistant", "user"]


def test_get_messages_scoped_by_tenant_miss_returns_empty(tmp_path):
    """Cross-tenant read returns [] — not an exception, not cross-tenant data."""
    store = SessionStore(str(tmp_path / "test.db"))
    sid = store.create_session(tenant="acme")
    store.append_message(sid, "user", "secret", tenant="acme")

    assert len(store.get_messages(sid, tenant="acme")) == 1
    assert store.get_messages(sid, tenant="globex") == []
    assert len(store.get_messages(sid)) == 1  # admin sees all


def test_get_messages_returns_message_objects(tmp_path):
    """Rows are reconstructed as frozen Message dataclasses with typed datetime."""
    store = SessionStore(str(tmp_path / "test.db"))
    sid = store.create_session()
    store.append_message(sid, "user", "hello")

    [m] = store.get_messages(sid)
    assert isinstance(m, Message)
    assert isinstance(m.created_at, datetime)
    assert m.created_at.tzinfo is not None
    assert m.role == "user"
    assert m.session_id == sid


def test_cascade_delete_removes_messages(tmp_path):
    """FK ON DELETE CASCADE from messages.session_id → sessions.session_id."""
    db = tmp_path / "test.db"
    store = SessionStore(str(db))
    sid = store.create_session()
    store.append_message(sid, "user", "keep me?")
    assert len(store.get_messages(sid)) == 1

    # Raw SQL delete (no delete_session method yet)
    conn = store._conn()
    conn.execute("DELETE FROM sessions WHERE session_id = ?", (sid,))
    conn.commit()

    assert store.get_messages(sid) == []


# ============================================================
# P9.1 — optimistic concurrency on update_submission
# ============================================================

from accord_ai.core.store import ConcurrencyError


def test_concurrency_error_is_value_error():
    assert issubclass(ConcurrencyError, ValueError)


def test_update_submission_matching_expected_updated_at_succeeds(store):
    sid = store.create_session()
    baseline = store.get_session(sid).updated_at

    store.update_submission(
        sid,
        CustomerSubmission(business_name="Acme"),
        expected_updated_at=baseline,
    )
    assert store.get_session(sid).submission.business_name == "Acme"


def test_update_submission_mismatched_expected_updated_at_raises(store, frozen_clock):
    """Stale baseline → ConcurrencyError, NOT KeyError."""
    frozen_clock(
        "2026-04-18T10:00:00+00:00",  # create_session
        "2026-04-18T10:00:05+00:00",  # append_message bumps updated_at
        "2026-04-18T10:00:10+00:00",  # attempted update_submission (consumed, then raises)
    )
    sid = store.create_session()
    stale_baseline = store.get_session(sid).updated_at

    # Someone else (simulated) commits — appends a message, bumping updated_at
    store.append_message(sid, "user", "stealth write")

    # Now try to update with the stale baseline
    with pytest.raises(ConcurrencyError):
        store.update_submission(
            sid,
            CustomerSubmission(business_name="lost-update"),
            expected_updated_at=stale_baseline,
        )

    # Submission NOT modified
    assert store.get_session(sid).submission.business_name is None


def test_update_submission_expected_none_preserves_old_behavior(store):
    """expected_updated_at=None (default) = no version check, existing contract."""
    sid = store.create_session()
    # Bump updated_at — the stale-baseline scenario doesn't apply here
    store.append_message(sid, "user", "hi")

    # No expected_updated_at passed — update succeeds regardless of version
    store.update_submission(sid, CustomerSubmission(business_name="Acme"))
    assert store.get_session(sid).submission.business_name == "Acme"


def test_update_submission_missing_session_still_raises_key_error(store):
    """expected_updated_at=some-value on a nonexistent session → KeyError, not ConcurrencyError."""
    from datetime import datetime, timezone

    with pytest.raises(KeyError):
        store.update_submission(
            "no-such-id",
            CustomerSubmission(business_name="ghost"),
            expected_updated_at=datetime.now(timezone.utc),
        )


def test_update_submission_finalized_session_raises_key_error_even_with_matching_updated_at(store):
    """expected_updated_at match BUT session is finalized — KeyError wins (state-machine guard)."""
    sid = store.create_session()
    baseline = store.get_session(sid).updated_at
    store.finalize_session(sid)

    with pytest.raises(KeyError):
        store.update_submission(
            sid,
            CustomerSubmission(business_name="late"),
            expected_updated_at=baseline,
        )


# ============================================================
# 5.e — lifecycle + state machine + list_sessions + SessionSummary
# ============================================================

from accord_ai.core.store import SessionSummary


# --- immutability of migration list ---

def test_migrations_is_immutable():
    import accord_ai.core.store as store_module
    assert isinstance(store_module._MIGRATIONS, tuple)
    with pytest.raises(AttributeError):
        store_module._MIGRATIONS.append((99, "boom"))   # tuples have no append


# --- finalize_session ---

def test_finalize_transitions_to_finalized(tmp_path):
    store = SessionStore(str(tmp_path / "t.db"))
    sid = store.create_session()
    store.finalize_session(sid)
    assert store.get_session(sid).status == "finalized"


def test_finalize_idempotent_when_already_finalized(tmp_path):
    store = SessionStore(str(tmp_path / "t.db"))
    sid = store.create_session()
    store.finalize_session(sid)
    store.finalize_session(sid)   # no error
    assert store.get_session(sid).status == "finalized"


def test_finalize_raises_valueerror_if_already_expired(tmp_path):
    store = SessionStore(str(tmp_path / "t.db"))
    sid = store.create_session()
    store.expire_session(sid)
    with pytest.raises(ValueError):
        store.finalize_session(sid)


def test_finalize_raises_for_missing_id(tmp_path):
    store = SessionStore(str(tmp_path / "t.db"))
    with pytest.raises(KeyError):
        store.finalize_session("no-such-id")


def test_finalize_scoped_by_tenant(tmp_path):
    store = SessionStore(str(tmp_path / "t.db"))
    sid = store.create_session(tenant="acme")
    with pytest.raises(KeyError):
        store.finalize_session(sid, tenant="globex")
    assert store.get_session(sid).status == "active"   # untouched
    store.finalize_session(sid, tenant="acme")
    assert store.get_session(sid).status == "finalized"


def test_finalize_bumps_updated_at(tmp_path, monkeypatch):
    import accord_ai.core.store as store_module
    times = iter([
        "2026-04-18T10:00:00.000000+00:00",
        "2026-04-18T10:00:10.000000+00:00",
    ])
    monkeypatch.setattr(store_module, "_now_utc_iso", lambda: next(times))

    store = SessionStore(str(tmp_path / "t.db"))
    sid = store.create_session()
    before = store.get_session(sid).updated_at
    store.finalize_session(sid)
    after = store.get_session(sid).updated_at
    assert after > before


# --- expire_session ---

def test_expire_transitions_to_expired(tmp_path):
    store = SessionStore(str(tmp_path / "t.db"))
    sid = store.create_session()
    store.expire_session(sid)
    assert store.get_session(sid).status == "expired"


def test_expire_idempotent_when_already_expired(tmp_path):
    store = SessionStore(str(tmp_path / "t.db"))
    sid = store.create_session()
    store.expire_session(sid)
    store.expire_session(sid)
    assert store.get_session(sid).status == "expired"


def test_expire_raises_valueerror_if_finalized(tmp_path):
    store = SessionStore(str(tmp_path / "t.db"))
    sid = store.create_session()
    store.finalize_session(sid)
    with pytest.raises(ValueError):
        store.expire_session(sid)


def test_expire_raises_for_missing_id(tmp_path):
    store = SessionStore(str(tmp_path / "t.db"))
    with pytest.raises(KeyError):
        store.expire_session("no-such-id")


# --- state-machine guards on write paths ---

def test_update_submission_rejects_finalized_session(tmp_path):
    """No mutation after finalize — the 'finalized means locked' invariant."""
    store = SessionStore(str(tmp_path / "t.db"))
    sid = store.create_session()
    store.finalize_session(sid)
    with pytest.raises(KeyError):
        store.update_submission(sid, CustomerSubmission(business_name="late-edit"))


def test_update_submission_rejects_expired_session(tmp_path):
    store = SessionStore(str(tmp_path / "t.db"))
    sid = store.create_session()
    store.expire_session(sid)
    with pytest.raises(KeyError):
        store.update_submission(sid, CustomerSubmission(business_name="ghost"))


def test_append_message_rejects_finalized_session(tmp_path):
    """No new messages after finalize — session is sealed."""
    store = SessionStore(str(tmp_path / "t.db"))
    sid = store.create_session()
    store.finalize_session(sid)
    with pytest.raises(KeyError):
        store.append_message(sid, "user", "after the gate")


def test_append_message_rejects_expired_session(tmp_path):
    store = SessionStore(str(tmp_path / "t.db"))
    sid = store.create_session()
    store.expire_session(sid)
    with pytest.raises(KeyError):
        store.append_message(sid, "user", "too late")


# --- list_sessions + SessionSummary ---

def test_list_sessions_returns_summaries_not_full(tmp_path):
    """SessionSummary has no 'submission' attr — avoids parsing JSON per row."""
    store = SessionStore(str(tmp_path / "t.db"))
    store.create_session()
    [summary] = store.list_sessions()
    assert isinstance(summary, SessionSummary)
    assert not hasattr(summary, "submission")


def test_list_sessions_filters_by_tenant(tmp_path):
    store = SessionStore(str(tmp_path / "t.db"))
    a1 = store.create_session(tenant="acme")
    a2 = store.create_session(tenant="acme")
    g = store.create_session(tenant="globex")

    assert {s.session_id for s in store.list_sessions(tenant="acme")} == {a1, a2}
    assert {s.session_id for s in store.list_sessions(tenant="globex")} == {g}


def test_list_sessions_admin_view_sees_all(tmp_path):
    store = SessionStore(str(tmp_path / "t.db"))
    a = store.create_session(tenant="acme")
    g = store.create_session(tenant="globex")
    none = store.create_session()

    assert {s.session_id for s in store.list_sessions()} == {a, g, none}


def test_list_sessions_filters_by_status(tmp_path):
    store = SessionStore(str(tmp_path / "t.db"))
    active_sid = store.create_session()
    final_sid = store.create_session()
    store.finalize_session(final_sid)

    active_only = store.list_sessions(status="active")
    assert {s.session_id for s in active_only} == {active_sid}

    finalized_only = store.list_sessions(status="finalized")
    assert {s.session_id for s in finalized_only} == {final_sid}


def test_list_sessions_newest_first_by_updated_at(tmp_path, monkeypatch):
    """Ordering: updated_at DESC with session_id DESC tiebreak."""
    import accord_ai.core.store as store_module
    times = iter([
        "2026-04-18T10:00:00.000000+00:00",
        "2026-04-18T11:00:00.000000+00:00",
        "2026-04-18T12:00:00.000000+00:00",
    ])
    monkeypatch.setattr(store_module, "_now_utc_iso", lambda: next(times))

    store = SessionStore(str(tmp_path / "t.db"))
    s1 = store.create_session()
    s2 = store.create_session()
    s3 = store.create_session()

    assert [s.session_id for s in store.list_sessions()] == [s3, s2, s1]


def test_list_sessions_respects_limit(tmp_path):
    store = SessionStore(str(tmp_path / "t.db"))
    for _ in range(5):
        store.create_session()
    assert len(store.list_sessions(limit=2)) == 2
    assert len(store.list_sessions(limit=10)) == 5   # limit > N is harmless


# --- get_messages limit (new kw from 5.e) ---

def test_get_messages_respects_limit(tmp_path, monkeypatch):
    import accord_ai.core.store as store_module
    times = iter([
        "2026-04-18T10:00:00.000000+00:00",  # create
        "2026-04-18T10:00:01.000000+00:00",  # append 1
        "2026-04-18T10:00:02.000000+00:00",  # append 2
        "2026-04-18T10:00:03.000000+00:00",  # append 3
    ])
    monkeypatch.setattr(store_module, "_now_utc_iso", lambda: next(times))

    store = SessionStore(str(tmp_path / "t.db"))
    sid = store.create_session()
    for i in range(3):
        store.append_message(sid, "user", f"m{i}")

    first_two = store.get_messages(sid, limit=2)
    assert [m.content for m in first_two] == ["m0", "m1"]


# ============================================================
# 6.e — apply_submission_diff (Phase 3 close-out)
# ============================================================

from accord_ai.core.diff import LobTransitionError


def test_apply_submission_diff_merges_and_persists(store):
    sid = store.create_session()
    store.update_submission(sid, CustomerSubmission(business_name="Acme"))

    diff = CustomerSubmission(ein="12-3456789", email="ops@acme.com")
    merged = store.apply_submission_diff(sid, diff)

    # Returned value has the merge
    assert merged.business_name == "Acme"
    assert merged.ein == "12-3456789"
    assert merged.email == "ops@acme.com"
    # And it's been persisted
    restored = store.get_session(sid).submission
    assert restored.business_name == "Acme"
    assert restored.ein == "12-3456789"
    assert restored.email == "ops@acme.com"


def test_apply_submission_diff_loose_removal_protection(store):
    """Scalar=None in diff leaves current field untouched."""
    sid = store.create_session()
    store.update_submission(sid, CustomerSubmission(business_name="Acme"))

    store.apply_submission_diff(sid, CustomerSubmission(business_name=None))
    assert store.get_session(sid).submission.business_name == "Acme"


def test_apply_submission_diff_raises_for_missing_session(store):
    with pytest.raises(KeyError):
        store.apply_submission_diff("no-such-id", CustomerSubmission(business_name="x"))


def test_apply_submission_diff_rejects_finalized_session(store):
    """State-machine guard: finalized session can't be mutated."""
    sid = store.create_session()
    store.finalize_session(sid)
    with pytest.raises(KeyError):
        store.apply_submission_diff(sid, CustomerSubmission(business_name="late"))


def test_apply_submission_diff_rejects_expired_session(store):
    sid = store.create_session()
    store.expire_session(sid)
    with pytest.raises(KeyError):
        store.apply_submission_diff(sid, CustomerSubmission(business_name="ghost"))


def test_apply_submission_diff_scoped_by_tenant(store):
    """Wrong-tenant attempt surfaces as KeyError — no silent cross-tenant write."""
    sid = store.create_session(tenant="acme")
    with pytest.raises(KeyError):
        store.apply_submission_diff(
            sid, CustomerSubmission(business_name="pwned"), tenant="globex"
        )
    # Original untouched
    assert store.get_session(sid).submission.business_name is None
    # Same-tenant works
    store.apply_submission_diff(
        sid, CustomerSubmission(business_name="ok"), tenant="acme"
    )
    assert store.get_session(sid).submission.business_name == "ok"


def test_apply_submission_diff_rolls_back_on_lob_transition(store):
    """LobTransitionError inside _tx -> rollback. Stored LOB unchanged."""
    sid = store.create_session()
    store.apply_submission_diff(
        sid, CustomerSubmission(
            business_name="Acme",
            lob_details={"lob": "commercial_auto", "drivers": [{"first_name": "Alice"}]},
        ),
    )

    bad_diff = CustomerSubmission(
        business_name="should-not-apply",
        lob_details={"lob": "general_liability", "employee_count": 5},
    )
    with pytest.raises(LobTransitionError):
        store.apply_submission_diff(sid, bad_diff)

    # Stored state pristine — neither business_name nor lob_details changed
    stored = store.get_session(sid).submission
    assert stored.business_name == "Acme"
    assert stored.lob_details.lob == "commercial_auto"


def test_apply_submission_diff_bumps_updated_at(store, frozen_clock):
    frozen_clock(
        "2026-04-18T10:00:00+00:00",
        "2026-04-18T10:00:05+00:00",
    )
    sid = store.create_session()
    before = store.get_session(sid).updated_at
    store.apply_submission_diff(sid, CustomerSubmission(business_name="Acme"))
    after = store.get_session(sid).updated_at
    assert after > before


def test_apply_submission_diff_returns_merged_submission(store):
    """Callers skip a second get_session — the return value is authoritative."""
    sid = store.create_session()
    store.update_submission(sid, CustomerSubmission(
        business_name="Acme",
        business_address={"city": "Detroit"},
    ))
    merged = store.apply_submission_diff(
        sid, CustomerSubmission(business_address={"state": "MI", "zip_code": "48201"})
    )
    # Nested merge (6.b semantics)
    assert merged.business_name == "Acme"
    assert merged.business_address.city == "Detroit"
    assert merged.business_address.state == "MI"
    assert merged.business_address.zip_code == "48201"


def test_apply_submission_diff_update_rejects_concurrent_finalize(store, monkeypatch):
    """Race guard: if a concurrent finalize lands between SELECT and UPDATE
    inside apply_submission_diff, the UPDATE's status='active' clause must
    make it a no-op → KeyError, not silent cross-state write.

    Simulated by monkeypatching apply_diff (as imported by the store module)
    to run finalize_session during the merge step — i.e., after the SELECT
    has returned a row, before our UPDATE fires.
    """
    from accord_ai.core import diff as diff_module
    import accord_ai.core.store as store_module

    sid = store.create_session()
    store.update_submission(sid, CustomerSubmission(business_name="Acme"))

    real_apply_diff = diff_module.apply_diff

    def racing_apply_diff(current, diff):
        # "Another actor" finalizes the session mid-merge
        store.finalize_session(sid)
        return real_apply_diff(current, diff)

    monkeypatch.setattr(store_module, "apply_diff", racing_apply_diff)

    with pytest.raises(KeyError):
        store.apply_submission_diff(
            sid, CustomerSubmission(business_name="racy-update")
        )

    # The concurrent finalize persisted; the diff UPDATE did NOT.
    final = store.get_session(sid)
    assert final.status == "finalized"
    assert final.submission.business_name == "Acme"   # not "racy-update"
