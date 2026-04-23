"""Tenant-scoped retrieval of recent corrections for extraction prompt injection (Phase 2.4).

CorrectionMemory queries the corrections table (SQL only — no vector DB) and
returns a compact list of recent fixes the broker has made, so the extractor can
avoid re-making the same mistakes.
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator, List, Optional

import logging

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connection helper (same pattern as DPOManager — fresh conn per call)
# ---------------------------------------------------------------------------


@contextmanager
def _get_conn(db_path: str) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorrectionMemoryEntry:
    field_path: str
    wrong_value: str    # already PII-redacted in storage
    correct_value: str
    explanation: Optional[str]
    created_at: datetime

    def as_prompt_line(self) -> str:
        line = (
            f"- field \"{self.field_path}\": "
            f"was {_truncate(self.wrong_value, 40)} "
            f"→ {_truncate(self.correct_value, 40)}"
        )
        if self.explanation:
            line += f" ({_truncate(self.explanation, 60)})"
        return line


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


class CorrectionMemory:
    """Tenant-scoped retrieval of recent corrections for prompt injection.

    Queries across all sessions for this tenant — cross-session pattern
    memorization is the goal.  Tenant scoping is enforced in SQL.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    def get_relevant(
        self,
        tenant: str,
        *,
        limit: int = 5,
        max_age_days: int = 30,
        field_path_prefix: Optional[str] = None,
    ) -> List[CorrectionMemoryEntry]:
        """Most recent N corrections for a tenant (newest first).

        Parameters
        ----------
        tenant:
            Must match the corrections.tenant column exactly.
        limit:
            Max entries returned.
        max_age_days:
            Corrections older than this are ignored.
        field_path_prefix:
            If set, only corrections whose field_path starts with this prefix
            are returned.  Useful when the extractor knows which LOB sub-tree
            is being collected (Phase 3+).
        """
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=max_age_days)
        ).isoformat()

        sql = """
            SELECT field_path, wrong_value_json, correct_value_json, explanation, created_at
            FROM corrections
            WHERE tenant = ?
              AND created_at >= ?
              AND status IN ('pending', 'graduated')
        """
        args: List[Any] = [tenant, cutoff]

        if field_path_prefix:
            sql += " AND field_path LIKE ?"
            args.append(f"{field_path_prefix}%")

        sql += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)

        t0 = time.monotonic()
        with _get_conn(self._db_path) as conn:
            rows = conn.execute(sql, args).fetchall()
        elapsed_ms = (time.monotonic() - t0) * 1000

        if elapsed_ms > 50:
            _logger.warning(
                "correction_memory_query_slow tenant=%s ms=%.1f", tenant, elapsed_ms
            )

        return [
            CorrectionMemoryEntry(
                field_path=row["field_path"],
                wrong_value=_deserialize(row["wrong_value_json"]),
                correct_value=_deserialize(row["correct_value_json"]),
                explanation=row["explanation"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deserialize(json_str: Optional[str]) -> str:
    """JSON-stored value → flat string for prompt display."""
    if json_str is None:
        return "null"
    try:
        val = json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return str(json_str)
    if isinstance(val, (dict, list)):
        return json.dumps(val)
    return str(val)


def _truncate(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "\u2026"
