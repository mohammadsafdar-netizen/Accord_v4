"""Tests for store.py schema migration v4 (Phase 2.1) — 4 tests."""
from __future__ import annotations

import sqlite3

import pytest

from accord_ai.core.store import SessionStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _store(tmp_path) -> SessionStore:
    return SessionStore(db_path=str(tmp_path / "test.db"))


def _tables(db_path: str) -> set[str]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    conn.close()
    return {r[0] for r in rows}


def _user_version(db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    v = conn.execute("PRAGMA user_version").fetchone()[0]
    conn.close()
    return v


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_fresh_db_creates_all_tables(tmp_path):
    """Instantiating SessionStore on a new DB creates all 6 tables."""
    db_path = str(tmp_path / "fresh.db")
    SessionStore(db_path=db_path)
    tables = _tables(db_path)
    assert "sessions" in tables
    assert "messages" in tables
    assert "audit_events" in tables
    assert "corrections" in tables
    assert "feedback" in tables
    assert "training_pairs" in tables


def test_migration_idempotent(tmp_path):
    """Running migrations twice leaves user_version=4 exactly once."""
    db_path = str(tmp_path / "idem.db")
    SessionStore(db_path=db_path)
    # Second open — should be a no-op
    SessionStore(db_path=db_path)
    assert _user_version(db_path) == 5


def test_existing_db_upgrades_cleanly(tmp_path):
    """A DB with only v1–v3 tables gains v4 tables on next SessionStore open."""
    db_path = str(tmp_path / "upgrade.db")

    # Simulate a v3 database by bootstrapping without the v4 migration
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY,
            tenant TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            submission_json TEXT NOT NULL
        );
        CREATE TABLE messages (
            message_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL
        );
        CREATE TABLE audit_events (
            audit_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            event_type TEXT NOT NULL,
            tenant TEXT,
            session_id TEXT,
            request_id TEXT,
            payload TEXT NOT NULL DEFAULT '{}'
        );
    """)
    conn.execute("PRAGMA user_version = 3")
    conn.commit()
    conn.close()

    # Now open with SessionStore — should apply v4 migration
    SessionStore(db_path=db_path)

    tables = _tables(db_path)
    assert "corrections" in tables, "corrections table should exist after migration"
    assert "feedback" in tables
    assert "training_pairs" in tables
    assert _user_version(db_path) == 5


def test_foreign_keys_enabled_and_on_delete_set_null(tmp_path):
    """FK enforcement is on: invalid correction_id raises; ON DELETE SET NULL works."""
    db_path = str(tmp_path / "fk.db")
    SessionStore(db_path=db_path)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.row_factory = sqlite3.Row

    # Insert a real correction first
    conn.execute(
        """INSERT INTO corrections
           (id, tenant, session_id, turn, field_path, status, created_at)
           VALUES ('c1', 'acme', 's1', 1, 'ein', 'pending', '2026-01-01T00:00:00')"""
    )
    # Insert training pair referencing that correction
    conn.execute(
        """INSERT INTO training_pairs
           (id, tenant, prompt, chosen, rejected, correction_id, status, created_at)
           VALUES ('tp1', 'acme', 'p', 'c', 'r', 'c1', 'pending', '2026-01-01T00:00:00')"""
    )
    conn.commit()

    # Delete the correction — ON DELETE SET NULL should null out the FK
    conn.execute("DELETE FROM corrections WHERE id = 'c1'")
    conn.commit()
    row = conn.execute("SELECT correction_id FROM training_pairs WHERE id = 'tp1'").fetchone()
    assert row["correction_id"] is None, "FK should be NULL after parent deletion"

    # Inserting a training pair with a non-existent correction_id must fail
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO training_pairs
               (id, tenant, prompt, chosen, rejected, correction_id, status, created_at)
               VALUES ('tp2', 'acme', 'p', 'c', 'r', 'nonexistent', 'pending', '2026-01-01T00:00:00')"""
        )
    conn.close()
