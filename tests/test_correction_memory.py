"""Tests for CorrectionMemory (Phase 2.4) — 8 tests."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from accord_ai.core.store import SessionStore
from accord_ai.feedback.memory import CorrectionMemory, CorrectionMemoryEntry, _truncate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_db(tmp_path: Path) -> str:
    db_path = str(tmp_path / "test.db")
    SessionStore(db_path=db_path)
    return db_path


def _insert_correction(
    db_path: str,
    *,
    tenant: str = "acme",
    session_id: str = "s1",
    field_path: str = "business_name",
    wrong_value: str = "Acme Inc",
    correct_value: str = "Acme LLC",
    explanation: str | None = None,
    status: str = "pending",
    created_at: str | None = None,
) -> str:
    cid = uuid.uuid4().hex
    ts = created_at or datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute(
        """
        INSERT INTO corrections
            (id, tenant, session_id, turn, field_path,
             wrong_value_json, correct_value_json, explanation,
             correction_type, status, created_at)
        VALUES (?, ?, ?, 1, ?, ?, ?, ?, 'value_correction', ?, ?)
        """,
        (cid, tenant, session_id, field_path,
         json.dumps(wrong_value), json.dumps(correct_value),
         explanation, status, ts),
    )
    conn.commit()
    conn.close()
    return cid


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_get_relevant_returns_recent_first(tmp_path):
    """Newest corrections come back first."""
    db_path = _setup_db(tmp_path)
    base = datetime.now(timezone.utc)
    for i in range(3):
        ts = (base + timedelta(seconds=i)).isoformat()
        _insert_correction(
            db_path, tenant="acme", session_id=f"s{i}",
            field_path=f"field_{i}", wrong_value=f"old{i}", correct_value=f"new{i}",
            created_at=ts,
        )

    mem = CorrectionMemory(db_path)
    entries = mem.get_relevant("acme")
    assert len(entries) == 3
    assert entries[0].field_path == "field_2"  # newest first
    assert entries[1].field_path == "field_1"
    assert entries[2].field_path == "field_0"


def test_get_relevant_tenant_isolated(tmp_path):
    """Tenant A's corrections never appear for tenant B."""
    db_path = _setup_db(tmp_path)
    _insert_correction(db_path, tenant="acme", field_path="ein")
    _insert_correction(db_path, tenant="globex", field_path="phone")

    mem = CorrectionMemory(db_path)
    acme = mem.get_relevant("acme")
    globex = mem.get_relevant("globex")
    assert all(e.field_path == "ein" for e in acme)
    assert all(e.field_path == "phone" for e in globex)
    assert len(acme) == 1
    assert len(globex) == 1


def test_get_relevant_respects_limit(tmp_path):
    """limit= caps the returned count."""
    db_path = _setup_db(tmp_path)
    for i in range(10):
        _insert_correction(db_path, tenant="acme", session_id=f"s{i}",
                           field_path=f"f{i}")

    mem = CorrectionMemory(db_path)
    entries = mem.get_relevant("acme", limit=3)
    assert len(entries) == 3


def test_get_relevant_respects_max_age(tmp_path):
    """Corrections older than max_age_days are excluded."""
    db_path = _setup_db(tmp_path)
    old_ts = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
    recent_ts = datetime.now(timezone.utc).isoformat()

    _insert_correction(db_path, tenant="acme", session_id="old",
                       field_path="old_field", created_at=old_ts)
    _insert_correction(db_path, tenant="acme", session_id="new",
                       field_path="new_field", created_at=recent_ts)

    mem = CorrectionMemory(db_path)
    entries = mem.get_relevant("acme", max_age_days=30)
    assert len(entries) == 1
    assert entries[0].field_path == "new_field"


def test_get_relevant_filters_by_field_prefix(tmp_path):
    """field_path_prefix filters to matching paths only."""
    db_path = _setup_db(tmp_path)
    _insert_correction(db_path, tenant="acme", session_id="s1",
                       field_path="lob_details.vehicles[0].vin",
                       wrong_value="BADVIN", correct_value="GOODVIN")
    _insert_correction(db_path, tenant="acme", session_id="s2",
                       field_path="business_name",
                       wrong_value="Old", correct_value="New")

    mem = CorrectionMemory(db_path)
    vehicle_entries = mem.get_relevant("acme", field_path_prefix="lob_details")
    assert len(vehicle_entries) == 1
    assert vehicle_entries[0].field_path == "lob_details.vehicles[0].vin"


def test_get_relevant_excludes_rejected_status(tmp_path):
    """Corrections with status='rejected' are excluded."""
    db_path = _setup_db(tmp_path)
    _insert_correction(db_path, tenant="acme", session_id="s1",
                       status="rejected", field_path="rejected_field")
    _insert_correction(db_path, tenant="acme", session_id="s2",
                       status="pending", field_path="pending_field")

    mem = CorrectionMemory(db_path)
    entries = mem.get_relevant("acme")
    assert len(entries) == 1
    assert entries[0].field_path == "pending_field"


def test_get_relevant_includes_graduated_status(tmp_path):
    """Graduated corrections still feed the memory."""
    db_path = _setup_db(tmp_path)
    _insert_correction(db_path, tenant="acme", session_id="s1",
                       status="graduated", field_path="graduated_field")

    mem = CorrectionMemory(db_path)
    entries = mem.get_relevant("acme")
    assert len(entries) == 1
    assert entries[0].field_path == "graduated_field"


def test_get_relevant_empty_for_new_tenant(tmp_path):
    """No corrections → empty list, no crash."""
    db_path = _setup_db(tmp_path)
    mem = CorrectionMemory(db_path)
    entries = mem.get_relevant("unknown_tenant")
    assert entries == []
