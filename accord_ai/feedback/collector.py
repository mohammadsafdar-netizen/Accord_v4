"""SQLite-backed store for user corrections and feedback (Phase 2.2).

Replaces the JSONL stopgap from Phase 1.9. PII is redacted at the write
boundary via PIIFilter before values reach the DB. Thread-safe via per-thread
sqlite3 connections (same pattern as SessionStore).
"""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator, List, Optional

from accord_ai.logging_config import redact_pii_text


# ---------------------------------------------------------------------------
# PII filter
# ---------------------------------------------------------------------------


class PIIFilter:
    """Wraps redact_pii_text for structured values (dict/list/scalar)."""

    def redact(self, text: Optional[str]) -> Optional[str]:
        if text is None:
            return None
        result = redact_pii_text(text)
        return result if result is not None else text

    def redact_json(self, value: Any) -> Any:
        """Recursively redact PII from string leaves in dict/list/scalar."""
        if isinstance(value, dict):
            return {k: self.redact_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.redact_json(v) for v in value]
        if isinstance(value, str):
            result = redact_pii_text(value)
            return result if result is not None else value
        return value


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Correction:
    id: str
    tenant: str
    session_id: str
    turn: int
    field_path: str
    wrong_value: Any
    correct_value: Any
    explanation: Optional[str]
    correction_type: str
    status: str
    created_at: datetime
    graduated_at: Optional[datetime]


@dataclass(frozen=True)
class Feedback:
    id: str
    tenant: str
    session_id: str
    turn: Optional[int]
    rating: int
    comment: Optional[str]
    created_at: datetime


# ---------------------------------------------------------------------------
# Row converters
# ---------------------------------------------------------------------------


def _row_to_correction(r: sqlite3.Row) -> Correction:
    return Correction(
        id=r["id"],
        tenant=r["tenant"],
        session_id=r["session_id"],
        turn=r["turn"],
        field_path=r["field_path"],
        wrong_value=json.loads(r["wrong_value_json"]) if r["wrong_value_json"] else None,
        correct_value=json.loads(r["correct_value_json"]) if r["correct_value_json"] else None,
        explanation=r["explanation"],
        correction_type=r["correction_type"] or "value_correction",
        status=r["status"],
        created_at=datetime.fromisoformat(r["created_at"]),
        graduated_at=datetime.fromisoformat(r["graduated_at"]) if r["graduated_at"] else None,
    )


def _row_to_feedback(r: sqlite3.Row) -> Feedback:
    return Feedback(
        id=r["id"],
        tenant=r["tenant"],
        session_id=r["session_id"],
        turn=r["turn"],
        rating=r["rating"],
        comment=r["comment"],
        created_at=datetime.fromisoformat(r["created_at"]),
    )


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


class CorrectionCollector:
    """Thread-safe SQLite-backed store for corrections and feedback.

    Uses thread-local connections (same pattern as SessionStore). The DB
    must already have the v4 schema applied — run SessionStore.__init__ first.
    """

    def __init__(self, db_path: str, pii_filter: PIIFilter) -> None:
        self._db_path = db_path
        self._pii = pii_filter
        self._local = threading.local()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path)
            conn.execute("PRAGMA foreign_keys=ON;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        conn = self._conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # --- Write ---

    def record_correction(
        self,
        *,
        tenant: str,
        session_id: str,
        turn: int,
        field_path: str,
        wrong_value: Any,
        correct_value: Any,
        explanation: Optional[str] = None,
        correction_type: str = "value_correction",
    ) -> str:
        """Persist a field correction. Returns the generated UUID."""
        correction_id = str(uuid.uuid4())
        wrong_redacted = self._pii.redact_json(wrong_value)
        correct_redacted = self._pii.redact_json(correct_value)
        explanation_redacted = self._pii.redact(explanation)

        with self._tx() as conn:
            conn.execute(
                """INSERT INTO corrections
                   (id, tenant, session_id, turn, field_path,
                    wrong_value_json, correct_value_json,
                    explanation, correction_type, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)""",
                (
                    correction_id, tenant, session_id, turn, field_path,
                    json.dumps(wrong_redacted), json.dumps(correct_redacted),
                    explanation_redacted, correction_type, _now_utc_iso(),
                ),
            )
        return correction_id

    def record_feedback(
        self,
        *,
        tenant: str,
        session_id: str,
        turn: Optional[int],
        rating: int,
        comment: Optional[str] = None,
    ) -> str:
        """Persist a turn-level feedback rating. Returns the generated UUID."""
        feedback_id = str(uuid.uuid4())
        comment_redacted = self._pii.redact(comment)

        with self._tx() as conn:
            conn.execute(
                """INSERT INTO feedback
                   (id, tenant, session_id, turn, rating, comment, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (feedback_id, tenant, session_id, turn, rating, comment_redacted, _now_utc_iso()),
            )
        return feedback_id

    # --- Read ---

    def list_corrections(
        self,
        *,
        tenant: str,
        session_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Correction]:
        sql = "SELECT * FROM corrections WHERE tenant = ?"
        args: List[Any] = [tenant]
        if session_id:
            sql += " AND session_id = ?"
            args.append(session_id)
        if status:
            sql += " AND status = ?"
            args.append(status)
        sql += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        return [_row_to_correction(r) for r in self._conn().execute(sql, args).fetchall()]

    def list_feedback(
        self,
        *,
        tenant: str,
        session_id: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Feedback]:
        sql = "SELECT * FROM feedback WHERE tenant = ?"
        args: List[Any] = [tenant]
        if session_id:
            sql += " AND session_id = ?"
            args.append(session_id)
        sql += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        return [_row_to_feedback(r) for r in self._conn().execute(sql, args).fetchall()]

    def count_pending(self, tenant: str) -> int:
        row = self._conn().execute(
            "SELECT COUNT(*) FROM corrections WHERE tenant = ? AND status = 'pending'",
            (tenant,),
        ).fetchone()
        return row[0] if row else 0
