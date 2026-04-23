#!/usr/bin/env python
"""One-time migration: move JSONL-stopgap corrections/feedback into SQLite.

Idempotent — re-running is safe because INSERT OR IGNORE skips rows with
existing PKs (the id field is carried over from the JSONL record).

After migration, source JSONL files are moved to logs/archive/ to preserve
a forensic trail while keeping the active log directory clean.

Usage:
    uv run python scripts/migrate_jsonl_to_sqlite.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path


def _get_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.row_factory = sqlite3.Row
    return conn


def migrate_corrections(conn: sqlite3.Connection, log_dir: Path) -> int:
    imported = 0
    for path in sorted(log_dir.glob("corrections_incoming_*.jsonl")):
        tenant = path.stem.removeprefix("corrections_incoming_")
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"  WARN: skipping malformed line in {path.name}", file=sys.stderr)
                continue

            conn.execute(
                """INSERT OR IGNORE INTO corrections
                   (id, tenant, session_id, turn, field_path,
                    wrong_value_json, correct_value_json,
                    explanation, correction_type, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'value_correction', 'pending', ?)""",
                (
                    record.get("id"),
                    tenant,
                    record.get("session_id", ""),
                    record.get("turn", 0),
                    record.get("field_path", ""),
                    json.dumps(record.get("wrong_value")),
                    json.dumps(record.get("correct_value")),
                    record.get("explanation"),
                    record.get("captured_at", ""),
                ),
            )
            imported += 1
        conn.commit()

        archive_dir = log_dir / "archive"
        archive_dir.mkdir(exist_ok=True)
        path.rename(archive_dir / path.name)
        print(f"  archived: {path.name}")

    return imported


def migrate_feedback(conn: sqlite3.Connection, log_dir: Path) -> int:
    imported = 0
    for path in sorted(log_dir.glob("feedback_incoming_*.jsonl")):
        tenant = path.stem.removeprefix("feedback_incoming_")
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"  WARN: skipping malformed line in {path.name}", file=sys.stderr)
                continue

            conn.execute(
                """INSERT OR IGNORE INTO feedback
                   (id, tenant, session_id, turn, rating, comment, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.get("id"),
                    tenant,
                    record.get("session_id", ""),
                    record.get("turn"),
                    record.get("rating", 0),
                    record.get("comment"),
                    record.get("captured_at", ""),
                ),
            )
            imported += 1
        conn.commit()

        archive_dir = log_dir / "archive"
        archive_dir.mkdir(exist_ok=True)
        path.rename(archive_dir / path.name)
        print(f"  archived: {path.name}")

    return imported


def main() -> None:
    import os
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from accord_ai.config import Settings
    settings = Settings()
    db_path = settings.db_path
    log_dir = Path("logs")

    if not Path(db_path).exists():
        print(f"ERROR: DB not found at {db_path!r}. Run the app once to initialize.", file=sys.stderr)
        sys.exit(1)

    print(f"Migrating JSONL → SQLite: db={db_path!r}, log_dir={log_dir!r}")

    conn = _get_conn(db_path)
    try:
        corrections_imported = migrate_corrections(conn, log_dir)
        feedback_imported = migrate_feedback(conn, log_dir)
    finally:
        conn.close()

    print(f"Done. Imported: {corrections_imported} corrections, {feedback_imported} feedback records.")


if __name__ == "__main__":
    main()
