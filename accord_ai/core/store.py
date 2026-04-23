"""Session storage — SQLite per-session persistence + conversation log.

IDs:
  session_id / message_id -> 32-hex UUIDs (persisted, wide entropy)
  request_id (see accord_ai.request_context) -> 12-hex (ephemeral, logs only)
  These are deliberately different schemes — don't unify.
"""
from __future__ import annotations

import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator, List, Literal, Optional, Tuple

from accord_ai.schema import CustomerSubmission
from accord_ai.logging_config import get_logger
from accord_ai.core.diff import apply_diff

_logger = get_logger("store")

SessionStatus = Literal["active", "finalized", "expired"]
MessageRole = Literal["user", "assistant", "system"]

# Immutable — prevents runtime mutation of the migration sequence.
_MIGRATIONS: Tuple[Tuple[int, str], ...] = (
    (1, """
        CREATE TABLE sessions (
            session_id       TEXT PRIMARY KEY,
            tenant           TEXT,
            created_at       TEXT NOT NULL,
            updated_at       TEXT NOT NULL,
            status           TEXT NOT NULL DEFAULT 'active'
                CHECK (status IN ('active', 'finalized', 'expired')),
            submission_json  TEXT NOT NULL
        );
        CREATE INDEX idx_sessions_tenant ON sessions(tenant);
        CREATE INDEX idx_sessions_status ON sessions(status);
    """),
    (2, """
        CREATE TABLE messages (
            message_id   TEXT PRIMARY KEY,
            session_id   TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
            created_at   TEXT NOT NULL,
            role         TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
            content      TEXT NOT NULL
        );
        CREATE INDEX idx_messages_session_id ON messages(session_id);
    """),
    (3, """
        CREATE TABLE audit_events (
            audit_id    TEXT PRIMARY KEY,
            created_at  TEXT NOT NULL,
            event_type  TEXT NOT NULL,
            tenant      TEXT,
            session_id  TEXT,
            request_id  TEXT,
            payload     TEXT NOT NULL DEFAULT '{}'
        );
        CREATE INDEX idx_audit_event_type  ON audit_events(event_type);
        CREATE INDEX idx_audit_tenant      ON audit_events(tenant);
        CREATE INDEX idx_audit_session     ON audit_events(session_id);
        CREATE INDEX idx_audit_created_at  ON audit_events(created_at);
    """),
    (4, """
        CREATE TABLE IF NOT EXISTS corrections (
            id                  TEXT PRIMARY KEY,
            tenant              TEXT NOT NULL,
            session_id          TEXT NOT NULL,
            turn                INTEGER NOT NULL,
            field_path          TEXT NOT NULL,
            wrong_value_json    TEXT,
            correct_value_json  TEXT,
            explanation         TEXT,
            correction_type     TEXT NOT NULL DEFAULT 'value_correction',
            status              TEXT NOT NULL DEFAULT 'pending',
            created_at          TEXT NOT NULL,
            graduated_at        TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_corrections_tenant
            ON corrections(tenant);
        CREATE INDEX IF NOT EXISTS idx_corrections_session
            ON corrections(session_id);
        CREATE INDEX IF NOT EXISTS idx_corrections_tenant_status
            ON corrections(tenant, status);
        CREATE TABLE IF NOT EXISTS feedback (
            id          TEXT PRIMARY KEY,
            tenant      TEXT NOT NULL,
            session_id  TEXT NOT NULL,
            turn        INTEGER,
            rating      INTEGER NOT NULL,
            comment     TEXT,
            created_at  TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_feedback_tenant
            ON feedback(tenant);
        CREATE INDEX IF NOT EXISTS idx_feedback_session
            ON feedback(session_id);
        CREATE TABLE IF NOT EXISTS training_pairs (
            id             TEXT PRIMARY KEY,
            tenant         TEXT NOT NULL,
            prompt         TEXT NOT NULL,
            chosen         TEXT NOT NULL,
            rejected       TEXT NOT NULL,
            field_path     TEXT,
            correction_id  TEXT,
            explanation    TEXT,
            status         TEXT NOT NULL DEFAULT 'pending',
            created_at     TEXT NOT NULL,
            exported_at    TEXT,
            FOREIGN KEY (correction_id) REFERENCES corrections(id) ON DELETE SET NULL
        );
        CREATE INDEX IF NOT EXISTS idx_training_pairs_tenant_status
            ON training_pairs(tenant, status);
    """),
    (5, """
        ALTER TABLE sessions ADD COLUMN flow_state_json TEXT DEFAULT NULL;
    """),
)


def _run_migrations(conn: sqlite3.Connection) -> None:
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    for version, sql in _MIGRATIONS:
        if version > current:
            conn.executescript(sql)
            conn.execute(f"PRAGMA user_version = {version}")
    conn.commit()


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConcurrencyError(ValueError):
    """Raised when optimistic-concurrency check on a store write fails.

    A caller passed `expected_updated_at` to update_submission, and the row's
    actual `updated_at` didn't match — another writer committed first. The
    attempted update was NOT applied. Caller decides to retry / merge / 409.
    """


@dataclass(frozen=True)
class Session:
    """Full session snapshot — includes the parsed submission."""
    session_id: str
    tenant: Optional[str]
    created_at: datetime
    updated_at: datetime
    status: SessionStatus
    submission: CustomerSubmission
    flow_state_json: Optional[str] = None  # persisted FlowState (migration 5)


@dataclass(frozen=True)
class SessionSummary:
    """Lightweight session metadata — no submission parsing. Use for list views."""
    session_id: str
    tenant: Optional[str]
    created_at: datetime
    updated_at: datetime
    status: SessionStatus


@dataclass(frozen=True)
class Message:
    message_id: str
    session_id: str
    created_at: datetime
    role: MessageRole
    content: str


@dataclass(frozen=True)
class AuditEvent:
    """One row from the append-only audit trail.

    payload is the parsed JSON object — never a string. Insert-side
    normalization in insert_audit_event keeps on-disk JSON compact and
    deterministic (sort_keys=True) so event diffs stay readable.
    """
    audit_id: str
    created_at: datetime
    event_type: str
    tenant: Optional[str]
    session_id: Optional[str]
    request_id: Optional[str]
    payload: dict


def _row_to_session(r) -> Session:
    return Session(
        session_id=r["session_id"],
        tenant=r["tenant"],
        created_at=datetime.fromisoformat(r["created_at"]),
        updated_at=datetime.fromisoformat(r["updated_at"]),
        status=r["status"],
        submission=CustomerSubmission.model_validate_json(r["submission_json"]),
        flow_state_json=r["flow_state_json"],
    )


def _row_to_summary(r) -> SessionSummary:
    return SessionSummary(
        session_id=r["session_id"],
        tenant=r["tenant"],
        created_at=datetime.fromisoformat(r["created_at"]),
        updated_at=datetime.fromisoformat(r["updated_at"]),
        status=r["status"],
    )


class SessionStore:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._local = threading.local()

        init_conn = sqlite3.connect(db_path)
        try:
            init_conn.execute("PRAGMA journal_mode=WAL;")
            init_conn.execute("PRAGMA foreign_keys=ON;")
            _run_migrations(init_conn)
        finally:
            init_conn.close()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path)
            conn.execute("PRAGMA foreign_keys=ON;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            # WAL mode is a per-database-file flag set once at __init__ time;
            # it persists across connections so we don't re-assert it here.
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        """Transaction helper — commits on clean exit, rolls back on exception."""
        conn = self._conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def close(self) -> None:
        conn: Optional[sqlite3.Connection] = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # --- Session CRUD ---

    def create_session(self, tenant: Optional[str] = None) -> str:
        session_id = uuid.uuid4().hex
        now = _now_utc_iso()
        submission_json = CustomerSubmission().model_dump_json()
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO sessions "
                "(session_id, tenant, created_at, updated_at, status, submission_json) "
                "VALUES (?, ?, ?, ?, 'active', ?)",
                (session_id, tenant, now, now, submission_json),
            )
        _logger.debug("session created: id=%s tenant=%s", session_id, tenant)
        return session_id

    def get_session(
        self, session_id: str, *, tenant: Optional[str] = None
    ) -> Optional[Session]:
        conn = self._conn()
        if tenant is None:
            row = conn.execute(
                "SELECT session_id, tenant, created_at, updated_at, status, "
                "submission_json, flow_state_json "
                "FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT session_id, tenant, created_at, updated_at, status, "
                "submission_json, flow_state_json "
                "FROM sessions WHERE session_id = ? AND tenant IS ?",
                (session_id, tenant),
            ).fetchone()
        return _row_to_session(row) if row is not None else None

    def update_submission(
        self,
        session_id: str,
        submission: CustomerSubmission,
        *,
        tenant: Optional[str] = None,
        expected_updated_at: Optional[datetime] = None,
    ) -> None:
        """Replace submission JSON. Only valid on active sessions.

        If `expected_updated_at` is provided, the update is optimistic: it
        commits only if the row's current `updated_at` matches. On mismatch,
        raises ConcurrencyError (not KeyError) so the caller can tell a
        version conflict apart from a missing session.
        """
        sql = (
            "UPDATE sessions SET submission_json = ?, updated_at = ? "
            "WHERE session_id = ? AND status = 'active'"
        )
        params: List = [submission.model_dump_json(), _now_utc_iso(), session_id]
        if tenant is not None:
            sql += " AND tenant IS ?"
            params.append(tenant)
        if expected_updated_at is not None:
            sql += " AND updated_at = ?"
            params.append(expected_updated_at.isoformat())

        with self._tx() as conn:
            if conn.execute(sql, params).rowcount == 0:
                # Disambiguate: version-mismatch vs missing/wrong-tenant/terminal.
                # Cheap extra SELECT on the error path only.
                if expected_updated_at is not None:
                    check_sql = (
                        "SELECT updated_at FROM sessions "
                        "WHERE session_id = ? AND status = 'active'"
                    )
                    check_params: List = [session_id]
                    if tenant is not None:
                        check_sql += " AND tenant IS ?"
                        check_params.append(tenant)
                    row = conn.execute(check_sql, check_params).fetchone()
                    if row is not None:
                        raise ConcurrencyError(
                            f"session {session_id} was modified concurrently "
                            f"(expected updated_at={expected_updated_at.isoformat()}, "
                            f"actual={row['updated_at']})"
                        )
                raise KeyError(f"session not found: {session_id}")
            _logger.debug("submission updated: session=%s tenant=%s", session_id, tenant)

    def apply_submission_diff(
        self,
        session_id: str,
        diff: CustomerSubmission,
        *,
        tenant: Optional[str] = None,
    ) -> CustomerSubmission:
        """Load current submission, merge with diff, persist. Fully atomic.

        Returns the merged submission so callers don't need a second read.

        Raises:
            KeyError: session not found, wrong tenant, or not active
            LobTransitionError: diff would switch the LOB discriminator
                (raised by apply_diff — rollback is a no-op since the SELECT
                has run but no UPDATE has)
        """
        with self._tx() as conn:
            if tenant is None:
                row = conn.execute(
                    "SELECT submission_json FROM sessions "
                    "WHERE session_id = ? AND status = 'active'",
                    (session_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT submission_json FROM sessions "
                    "WHERE session_id = ? AND status = 'active' AND tenant IS ?",
                    (session_id, tenant),
                ).fetchone()

            if row is None:
                raise KeyError(f"session not found: {session_id}")

            current = CustomerSubmission.model_validate_json(row["submission_json"])
            merged = apply_diff(current, diff)   # may raise LobTransitionError

            # Re-check status + tenant on the UPDATE. Closes the race between
            # the SELECT above and this write — a concurrent finalize_session
            # (or tenant reassignment) would otherwise let this diff land on
            # a session that's no longer eligible.
            update_sql = (
                "UPDATE sessions SET submission_json = ?, updated_at = ? "
                "WHERE session_id = ? AND status = 'active'"
            )
            update_params: List = [merged.model_dump_json(), _now_utc_iso(), session_id]
            if tenant is not None:
                update_sql += " AND tenant IS ?"
                update_params.append(tenant)
            if conn.execute(update_sql, update_params).rowcount == 0:
                raise KeyError(f"session not found: {session_id}")
            _logger.debug(
                "submission diff applied: session=%s tenant=%s", session_id, tenant
            )
        return merged

    def update_flow_state(
        self,
        session_id: str,
        flow_state_json: str,
        *,
        tenant: Optional[str] = None,
    ) -> None:
        """Persist serialised FlowState JSON for an active session."""
        sql = (
            "UPDATE sessions SET flow_state_json = ?, updated_at = ? "
            "WHERE session_id = ? AND status = 'active'"
        )
        params: List = [flow_state_json, _now_utc_iso(), session_id]
        if tenant is not None:
            sql += " AND tenant IS ?"
            params.append(tenant)
        with self._tx() as conn:
            if conn.execute(sql, params).rowcount == 0:
                raise KeyError(f"session not found: {session_id}")

    # --- Messages ---

    def append_message(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        *,
        tenant: Optional[str] = None,
    ) -> str:
        """Append a message to an active session. Atomic — no TOCTOU.

        Uses INSERT … SELECT so the FK check, tenant match, and state-machine
        guard all happen in a single statement. rowcount=0 means the session
        is missing, wrong-tenant, or not active — all surface as KeyError.
        """
        message_id = uuid.uuid4().hex
        now = _now_utc_iso()

        sql = (
            "INSERT INTO messages (message_id, session_id, created_at, role, content) "
            "SELECT ?, session_id, ?, ?, ? FROM sessions "
            "WHERE session_id = ? AND status = 'active'"
        )
        params: List = [message_id, now, role, content, session_id]
        if tenant is not None:
            sql += " AND tenant IS ?"
            params.append(tenant)

        with self._tx() as conn:
            cur = conn.execute(sql, params)
            if cur.rowcount == 0:
                raise KeyError(f"session not found: {session_id}")
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )
            _logger.debug(
                "message appended: session=%s role=%s message=%s",
                session_id, role, message_id,
            )
        return message_id

    def get_messages(
        self,
        session_id: str,
        *,
        tenant: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Message]:
        """All messages for a session, oldest-first.

        limit: when set, returns the FIRST N messages (oldest-first). For
        "last N", callers pass limit and then reverse — cursor API deferred.
        """
        conn = self._conn()
        if tenant is None:
            sql = (
                "SELECT message_id, session_id, created_at, role, content "
                "FROM messages WHERE session_id = ? "
                "ORDER BY created_at ASC, message_id ASC"
            )
            params: List = [session_id]
        else:
            sql = (
                "SELECT m.message_id, m.session_id, m.created_at, m.role, m.content "
                "FROM messages m INNER JOIN sessions s ON s.session_id = m.session_id "
                "WHERE m.session_id = ? AND s.tenant IS ? "
                "ORDER BY m.created_at ASC, m.message_id ASC"
            )
            params = [session_id, tenant]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)

        rows = conn.execute(sql, params).fetchall()
        return [
            Message(
                message_id=r["message_id"],
                session_id=r["session_id"],
                created_at=datetime.fromisoformat(r["created_at"]),
                role=r["role"],
                content=r["content"],
            )
            for r in rows
        ]

    # --- Lifecycle ---

    def count_corrections_for_session(
        self, session_id: str, *, tenant: Optional[str] = None
    ) -> int:
        """Count corrections logged against a session (any status except deleted)."""
        conn = self._conn()
        if tenant is None:
            sql = "SELECT COUNT(*) FROM corrections WHERE session_id = ?"
            params: List = [session_id]
        else:
            sql = (
                "SELECT COUNT(*) FROM corrections c "
                "INNER JOIN sessions s ON s.session_id = c.session_id "
                "WHERE c.session_id = ? AND s.tenant IS ?"
            )
            params = [session_id, tenant]
        return conn.execute(sql, params).fetchone()[0]

    def finalize_session(self, session_id: str, *, tenant: Optional[str] = None) -> None:
        """active -> finalized. Idempotent when already finalized. ValueError on expired."""
        self._transition(session_id, target="finalized", tenant=tenant)

    def expire_session(self, session_id: str, *, tenant: Optional[str] = None) -> None:
        """active -> expired. Idempotent when already expired. ValueError on finalized."""
        self._transition(session_id, target="expired", tenant=tenant)

    def _transition(
        self, session_id: str, *, target: SessionStatus, tenant: Optional[str]
    ) -> None:
        """Atomic UPDATE with WHERE status='active'. Diagnose rowcount=0 to
        distinguish idempotent re-call from illegal transition from missing."""
        now = _now_utc_iso()
        sql = (
            f"UPDATE sessions SET status = ?, updated_at = ? "
            f"WHERE session_id = ? AND status = 'active'"
        )
        params: List = [target, now, session_id]
        if tenant is not None:
            sql += " AND tenant IS ?"
            params.append(tenant)

        with self._tx() as conn:
            if conn.execute(sql, params).rowcount == 1:
                _logger.debug(
                    "session transitioned: id=%s -> %s tenant=%s",
                    session_id, target, tenant,
                )
                return  # clean transition

            # rowcount=0: missing, wrong-tenant, or already in a terminal state
            existing = self.get_session(session_id, tenant=tenant)
            if existing is None:
                raise KeyError(f"session not found: {session_id}")
            if existing.status == target:
                return  # idempotent — already in target
            raise ValueError(
                f"cannot transition session from {existing.status!r} to {target!r}"
            )

    # --- List ---

    def list_sessions(
        self,
        *,
        tenant: Optional[str] = None,
        status: Optional[SessionStatus] = None,
        limit: Optional[int] = None,
    ) -> List[SessionSummary]:
        """Sessions newest-first, by last-activity (updated_at).

        tenant=None -> admin view (all tenants including NULL).
        status=None -> any status.
        Returns SessionSummary (no submission parsed) — cheap for list views.
        """
        clauses: List[str] = []
        params: List = []
        if tenant is not None:
            clauses.append("tenant IS ?")
            params.append(tenant)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = (
            "SELECT session_id, tenant, created_at, updated_at, status "
            f"FROM sessions {where} "
            "ORDER BY updated_at DESC, session_id DESC"
        )
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)

        rows = self._conn().execute(sql, tuple(params)).fetchall()
        return [_row_to_summary(r) for r in rows]

    # --- Audit ---

    def insert_audit_event(
        self,
        event_type: str,
        *,
        tenant: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        payload: Optional[dict] = None,
    ) -> str:
        """Append a single audit row. Returns the generated audit_id.

        Never raises on empty event_type / unknown tenant — audit is
        best-effort and must not block the caller's operation. Validation
        lives one layer up in accord_ai.audit.
        """
        import json as _json
        audit_id = uuid.uuid4().hex
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO audit_events "
                "(audit_id, created_at, event_type, tenant, session_id, "
                "request_id, payload) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    audit_id,
                    _now_utc_iso(),
                    event_type,
                    tenant,
                    session_id,
                    request_id,
                    _json.dumps(payload or {}, separators=(",", ":"), sort_keys=True),
                ),
            )
        return audit_id

    def list_audit_events(
        self,
        *,
        tenant: Optional[str] = None,
        event_type: Optional[str] = None,
        session_id: Optional[str] = None,
        after: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Most-recent-first. All filters AND-combined. `after` is exclusive."""
        import json as _json
        sql = (
            "SELECT audit_id, created_at, event_type, tenant, session_id, "
            "request_id, payload FROM audit_events WHERE 1=1"
        )
        params: List = []
        if tenant is not None:
            # IS ? (not = ?) mirrors session-side tenant handling — allows
            # tenant=None to match the NULL-tenant rows correctly.
            sql += " AND tenant IS ?"
            params.append(tenant)
        if event_type is not None:
            sql += " AND event_type = ?"
            params.append(event_type)
        if session_id is not None:
            sql += " AND session_id = ?"
            params.append(session_id)
        if after is not None:
            sql += " AND created_at > ?"
            params.append(after.isoformat())
        sql += " ORDER BY created_at DESC, audit_id DESC LIMIT ?"
        params.append(int(limit))

        rows = self._conn().execute(sql, tuple(params)).fetchall()
        return [
            AuditEvent(
                audit_id=r["audit_id"],
                created_at=datetime.fromisoformat(r["created_at"]),
                event_type=r["event_type"],
                tenant=r["tenant"],
                session_id=r["session_id"],
                request_id=r["request_id"],
                payload=_json.loads(r["payload"]),
            )
            for r in rows
        ]
