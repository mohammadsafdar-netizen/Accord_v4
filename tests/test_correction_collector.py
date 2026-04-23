"""Tests for CorrectionCollector (Phase 2.2) — 8 tests."""
from __future__ import annotations

import threading
from datetime import datetime

import pytest

from accord_ai.core.store import SessionStore
from accord_ai.feedback.collector import Correction, CorrectionCollector, Feedback, PIIFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collector(tmp_path) -> CorrectionCollector:
    db_path = str(tmp_path / "test.db")
    SessionStore(db_path=db_path)  # run migrations
    return CorrectionCollector(db_path=db_path, pii_filter=PIIFilter())


def _base_correction(collector: CorrectionCollector, **kwargs) -> str:
    defaults = dict(
        tenant="acme",
        session_id="sess-1",
        turn=3,
        field_path="ein",
        wrong_value="11-1111111",
        correct_value="22-2222222",
    )
    defaults.update(kwargs)
    return collector.record_correction(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_record_correction_roundtrip(tmp_path):
    """record_correction() persists and list_corrections() retrieves the row."""
    col = _collector(tmp_path)
    cid = _base_correction(col, explanation="user typed wrong EIN")
    corrections = col.list_corrections(tenant="acme")
    assert len(corrections) == 1
    c = corrections[0]
    assert c.id == cid
    assert c.tenant == "acme"
    assert c.session_id == "sess-1"
    assert c.turn == 3
    assert c.field_path == "ein"
    assert c.status == "pending"
    assert c.correction_type == "value_correction"
    assert isinstance(c.created_at, datetime)
    assert c.graduated_at is None


def test_record_correction_redacts_pii(tmp_path):
    """wrong_value containing SSN-like text is redacted before DB write."""
    col = _collector(tmp_path)
    _base_correction(col, wrong_value="SSN: 123-45-6789", correct_value="SSN: 987-65-4321")
    c = col.list_corrections(tenant="acme")[0]
    stored = str(c.wrong_value) + str(c.correct_value)
    # Raw SSN digits should not appear in stored values
    assert "123-45-6789" not in stored
    assert "987-65-4321" not in stored


def test_record_correction_tenant_isolated(tmp_path):
    """Tenant A's corrections are not visible to tenant B."""
    col = _collector(tmp_path)
    _base_correction(col, tenant="acme")
    _base_correction(col, tenant="beta", session_id="sess-2")

    acme_rows = col.list_corrections(tenant="acme")
    beta_rows = col.list_corrections(tenant="beta")
    assert len(acme_rows) == 1
    assert len(beta_rows) == 1
    assert acme_rows[0].tenant == "acme"
    assert beta_rows[0].tenant == "beta"


def test_record_correction_concurrent_writes(tmp_path):
    """100 threads inserting simultaneously all land in DB with no crashes."""
    col = _collector(tmp_path)
    results: list[str] = []
    errors: list[Exception] = []
    lock = threading.Lock()

    def _write(i: int) -> None:
        try:
            cid = col.record_correction(
                tenant="acme",
                session_id=f"sess-{i}",
                turn=i,
                field_path="field",
                wrong_value=i,
                correct_value=i + 1,
            )
            with lock:
                results.append(cid)
        except Exception as exc:
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=_write, args=(i,)) for i in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"concurrent write errors: {errors}"
    assert len(results) == 100
    # All IDs unique
    assert len(set(results)) == 100
    assert col.count_pending("acme") == 100


def test_list_corrections_by_status(tmp_path):
    """list_corrections(status='pending') excludes non-pending rows."""
    col = _collector(tmp_path)
    _base_correction(col, tenant="acme")
    cid2 = _base_correction(col, tenant="acme", field_path="business_name")

    # Manually graduate one row
    import sqlite3
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE corrections SET status='graduated' WHERE id=?", (cid2,))
    conn.commit()
    conn.close()

    pending = col.list_corrections(tenant="acme", status="pending")
    graduated = col.list_corrections(tenant="acme", status="graduated")
    assert len(pending) == 1
    assert len(graduated) == 1
    assert pending[0].field_path == "ein"
    assert graduated[0].field_path == "business_name"


def test_count_pending(tmp_path):
    """count_pending returns only pending count, not graduated."""
    col = _collector(tmp_path)
    for _ in range(5):
        _base_correction(col, tenant="acme")
    for _ in range(3):
        cid = _base_correction(col, tenant="acme", field_path="name")
        # Manually graduate
        import sqlite3
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("UPDATE corrections SET status='graduated' WHERE id=?", (cid,))
        conn.commit()
        conn.close()

    assert col.count_pending("acme") == 5


def test_record_feedback_roundtrip(tmp_path):
    """record_feedback() persists and list_feedback() retrieves the row."""
    col = _collector(tmp_path)
    fid = col.record_feedback(
        tenant="acme",
        session_id="sess-1",
        turn=2,
        rating=4,
        comment="Good extraction, missed the EIN",
    )
    feedback = col.list_feedback(tenant="acme")
    assert len(feedback) == 1
    f = feedback[0]
    assert f.id == fid
    assert f.tenant == "acme"
    assert f.rating == 4
    assert isinstance(f.created_at, datetime)


def test_record_feedback_rating_at_db_layer(tmp_path):
    """DB layer accepts any integer rating — bounds validation is the endpoint's job."""
    col = _collector(tmp_path)
    # DB itself imposes no check constraint on rating — just INTEGER NOT NULL
    for rating in (1, 3, 5, 99):
        col.record_feedback(
            tenant="acme", session_id="s", turn=None, rating=rating
        )
    rows = col.list_feedback(tenant="acme")
    assert len(rows) == 4
    stored_ratings = {r.rating for r in rows}
    assert {1, 3, 5, 99} == stored_ratings
