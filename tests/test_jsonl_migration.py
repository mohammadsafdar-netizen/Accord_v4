"""Tests for scripts/migrate_jsonl_to_sqlite.py (Phase 2.1) — 2 tests."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

from accord_ai.core.store import SessionStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_db(tmp_path: Path) -> str:
    db_path = str(tmp_path / "accord.db")
    SessionStore(db_path=db_path)  # run migrations
    return db_path


def _write_correction_jsonl(log_dir: Path, tenant: str, records: list[dict]) -> Path:
    path = log_dir / f"corrections_incoming_{tenant}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return path


def _write_feedback_jsonl(log_dir: Path, tenant: str, records: list[dict]) -> Path:
    path = log_dir / f"feedback_incoming_{tenant}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return path


def _run_migration(db_path: str, log_dir: Path) -> tuple[int, int]:
    """Import and call the migration directly (avoids subprocess + path issues)."""
    scripts_dir = Path(__file__).parent.parent / "scripts"
    sys.path.insert(0, str(scripts_dir.parent))

    from scripts.migrate_jsonl_to_sqlite import (
        _get_conn,
        migrate_corrections,
        migrate_feedback,
    )

    conn = _get_conn(db_path)
    try:
        c_count = migrate_corrections(conn, log_dir)
        f_count = migrate_feedback(conn, log_dir)
    finally:
        conn.close()
    return c_count, f_count


def _count_rows(db_path: str, table: str) -> int:
    conn = sqlite3.connect(db_path)
    n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    conn.close()
    return n


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_migrate_jsonl_idempotent(tmp_path):
    """Running migration twice produces the same row count — no duplicates."""
    db_path = _setup_db(tmp_path)
    log_dir = tmp_path / "logs"

    records = [
        {
            "id": f"corr-{i}",
            "session_id": f"sess-{i}",
            "turn": i,
            "field_path": "ein",
            "wrong_value": "bad",
            "correct_value": "good",
            "explanation": None,
            "captured_at": "2026-01-01T00:00:00",
        }
        for i in range(5)
    ]
    _write_correction_jsonl(log_dir, "acme", records)

    # First run
    c1, _ = _run_migration(db_path, log_dir)
    assert c1 == 5

    # Source files were archived — recreate them for the second run
    _write_correction_jsonl(log_dir, "acme", records)

    # Second run — INSERT OR IGNORE means no new rows
    c2, _ = _run_migration(db_path, log_dir)
    assert c2 == 5  # lines processed, but no new DB rows
    assert _count_rows(db_path, "corrections") == 5


def test_migrate_jsonl_archives_source_files(tmp_path):
    """After migration, source JSONL files appear in logs/archive/, not logs/."""
    db_path = _setup_db(tmp_path)
    log_dir = tmp_path / "logs"

    corr_records = [
        {
            "id": "c1",
            "session_id": "s1",
            "turn": 0,
            "field_path": "ein",
            "wrong_value": "bad",
            "correct_value": "good",
            "captured_at": "2026-01-01T00:00:00",
        }
    ]
    fb_records = [
        {
            "id": "f1",
            "session_id": "s1",
            "turn": 1,
            "rating": 3,
            "comment": None,
            "captured_at": "2026-01-01T00:00:00",
        }
    ]
    corr_path = _write_correction_jsonl(log_dir, "acme", corr_records)
    fb_path = _write_feedback_jsonl(log_dir, "acme", fb_records)

    _run_migration(db_path, log_dir)

    # Source files must no longer be in logs/
    assert not corr_path.exists(), "source correction JSONL should be archived"
    assert not fb_path.exists(), "source feedback JSONL should be archived"

    # Archives must exist
    archive_dir = log_dir / "archive"
    assert (archive_dir / "corrections_incoming_acme.jsonl").exists()
    assert (archive_dir / "feedback_incoming_acme.jsonl").exists()

    # DB rows were inserted
    assert _count_rows(db_path, "corrections") == 1
    assert _count_rows(db_path, "feedback") == 1
