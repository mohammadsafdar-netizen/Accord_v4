"""Tests for DPOManager (Phase 2.3) — 14 tests."""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from pathlib import Path

import pytest

from accord_ai.core.store import SessionStore
from accord_ai.feedback.collector import CorrectionCollector, PIIFilter
from accord_ai.feedback.dpo import DPOManager, _schema_fragment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_db(tmp_path: Path) -> str:
    db_path = str(tmp_path / "test.db")
    SessionStore(db_path=db_path)  # run migrations
    return db_path


def _insert_correction(
    db_path: str,
    *,
    tenant: str = "acme",
    session_id: str = "s1",
    turn: int = 1,
    field_path: str = "business_name",
    wrong_value: str = "Acme Inc",
    correct_value: str = "Acme LLC",
    explanation: str | None = None,
    status: str = "pending",
) -> str:
    cid = uuid.uuid4().hex
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = OFF")  # no sessions table required
    conn.execute(
        """
        INSERT INTO corrections
            (id, tenant, session_id, turn, field_path,
             wrong_value_json, correct_value_json, explanation,
             correction_type, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'value_correction', ?, datetime('now'))
        """,
        (cid, tenant, session_id, turn, field_path,
         json.dumps(wrong_value), json.dumps(correct_value),
         explanation, status),
    )
    conn.commit()
    conn.close()
    return cid


def _insert_message(
    db_path: str,
    *,
    session_id: str = "s1",
    content: str = "My company is Acme LLC",
) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute(
        """
        INSERT INTO messages (message_id, session_id, created_at, role, content)
        VALUES (?, ?, datetime('now'), 'user', ?)
        """,
        (uuid.uuid4().hex, session_id, content),
    )
    conn.commit()
    conn.close()


def _make_mgr(db_path: str, output_dir: Path, threshold: int = 3) -> DPOManager:
    return DPOManager(db_path=db_path, output_dir=output_dir, threshold=threshold)


# ---------------------------------------------------------------------------
# Count / eligibility
# ---------------------------------------------------------------------------


def test_count_pending_tenant_isolated(tmp_path):
    """count_pending is scoped to tenant."""
    db_path = _setup_db(tmp_path)
    for i in range(5):
        _insert_correction(db_path, tenant="acme", session_id=f"a{i}")
    for i in range(3):
        _insert_correction(db_path, tenant="globex", session_id=f"g{i}")

    mgr = _make_mgr(db_path, tmp_path / "out")
    assert mgr.count_pending("acme") == 5
    assert mgr.count_pending("globex") == 3
    assert mgr.count_pending("unknown") == 0


def test_eligible_false_below_threshold(tmp_path):
    db_path = _setup_db(tmp_path)
    _insert_correction(db_path, tenant="acme")
    mgr = _make_mgr(db_path, tmp_path / "out", threshold=5)
    assert mgr.eligible_for_training("acme") is False


def test_eligible_true_at_threshold(tmp_path):
    db_path = _setup_db(tmp_path)
    for i in range(5):
        _insert_correction(db_path, tenant="acme", session_id=f"s{i}")
    mgr = _make_mgr(db_path, tmp_path / "out", threshold=5)
    assert mgr.eligible_for_training("acme") is True


# ---------------------------------------------------------------------------
# build_pairs
# ---------------------------------------------------------------------------


def test_build_pairs_basic_shape(tmp_path):
    """3 corrections + matching messages → 3 DPOPairs with prompt/chosen/rejected."""
    db_path = _setup_db(tmp_path)
    for i in range(3):
        sid = f"s{i}"
        _insert_message(db_path, session_id=sid, content=f"User message {i}")
        _insert_correction(
            db_path, tenant="acme", session_id=sid, turn=1,
            field_path="business_name",
            wrong_value=f"Wrong {i}", correct_value=f"Correct {i}",
        )

    mgr = _make_mgr(db_path, tmp_path / "out")
    pairs = mgr.build_pairs("acme")
    assert len(pairs) == 3
    for p in pairs:
        assert "business_name" in p.prompt
        assert "business_name" in p.chosen or "business_name" in p.rejected
        parsed_chosen = json.loads(p.chosen)
        parsed_rejected = json.loads(p.rejected)
        assert "business_name" in parsed_chosen
        assert "business_name" in parsed_rejected
        assert p.template_version == "v1"
        assert p.correction_id


@pytest.fixture
def dpo_caplog(caplog):
    """Attach caplog directly to accord_ai.feedback.dpo — survives configure_logging()."""
    import logging
    dpo_logger = logging.getLogger("accord_ai.feedback.dpo")
    dpo_logger.addHandler(caplog.handler)
    original_level = dpo_logger.level
    dpo_logger.setLevel(logging.DEBUG)
    try:
        yield caplog
    finally:
        dpo_logger.removeHandler(caplog.handler)
        dpo_logger.setLevel(original_level)


def test_build_pairs_skips_missing_message(tmp_path, dpo_caplog):
    """Correction with no matching message is skipped, warning logged."""
    db_path = _setup_db(tmp_path)
    _insert_correction(db_path, tenant="acme", session_id="ghost", turn=1)

    mgr = _make_mgr(db_path, tmp_path / "out")
    pairs = mgr.build_pairs("acme")
    assert len(pairs) == 0
    assert any("dpo_skipped_missing_message" in r.message for r in dpo_caplog.records)


def test_build_pairs_skips_identical_values(tmp_path, dpo_caplog):
    """Skip when wrong_value and correct_value are identical JSON."""
    db_path = _setup_db(tmp_path)
    _insert_message(db_path, session_id="s1")
    _insert_correction(
        db_path, tenant="acme", session_id="s1", turn=1,
        wrong_value="Acme Inc", correct_value="Acme Inc",
    )

    mgr = _make_mgr(db_path, tmp_path / "out")
    pairs = mgr.build_pairs("acme")
    assert len(pairs) == 0
    assert any("dpo_skipped_identical_values" in r.message for r in dpo_caplog.records)


def test_build_pairs_tolerates_schema_drift(tmp_path):
    """Unknown field_path falls back to {'type': 'string'} schema fragment."""
    db_path = _setup_db(tmp_path)
    _insert_message(db_path, session_id="s1")
    _insert_correction(
        db_path, tenant="acme", session_id="s1", turn=1,
        field_path="nonexistent_field_xyz",
        wrong_value="old", correct_value="new",
    )

    mgr = _make_mgr(db_path, tmp_path / "out")
    pairs = mgr.build_pairs("acme")
    assert len(pairs) == 1
    assert '"type": "string"' in pairs[0].prompt


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def test_export_writes_jsonl_with_expected_lines(tmp_path):
    """5 pairs → JSONL has 5 lines, each parseable as {prompt, chosen, rejected, metadata}."""
    db_path = _setup_db(tmp_path)
    for i in range(5):
        sid = f"s{i}"
        _insert_message(db_path, session_id=sid)
        _insert_correction(db_path, tenant="acme", session_id=sid, turn=1,
                           wrong_value=f"W{i}", correct_value=f"C{i}")

    mgr = _make_mgr(db_path, tmp_path / "out", threshold=1)
    result = mgr.export("acme")
    assert result.count == 5
    assert result.path is not None
    lines = result.path.read_text().splitlines()
    assert len(lines) == 5
    for line in lines:
        rec = json.loads(line)
        assert "prompt" in rec
        assert "chosen" in rec
        assert "rejected" in rec
        assert "metadata" in rec
        assert "field_path" in rec["metadata"]
        assert "correction_id" in rec["metadata"]
        assert "template_version" in rec["metadata"]


def test_export_graduates_source_corrections(tmp_path):
    """Post-export, all exported corrections have status='graduated'."""
    db_path = _setup_db(tmp_path)
    cids = []
    for i in range(3):
        sid = f"s{i}"
        _insert_message(db_path, session_id=sid)
        cids.append(
            _insert_correction(db_path, tenant="acme", session_id=sid, turn=1,
                               wrong_value=f"W{i}", correct_value=f"C{i}")
        )

    mgr = _make_mgr(db_path, tmp_path / "out", threshold=1)
    mgr.export("acme")

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT status FROM corrections WHERE tenant='acme'"
    ).fetchall()
    conn.close()
    assert all(r[0] == "graduated" for r in rows)


def test_export_creates_training_pairs_rows(tmp_path):
    """Post-export, training_pairs has one row per exported correction."""
    db_path = _setup_db(tmp_path)
    for i in range(5):
        sid = f"s{i}"
        _insert_message(db_path, session_id=sid)
        _insert_correction(db_path, tenant="acme", session_id=sid, turn=1,
                           wrong_value=f"W{i}", correct_value=f"C{i}")

    mgr = _make_mgr(db_path, tmp_path / "out", threshold=1)
    mgr.export("acme")

    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM training_pairs WHERE tenant='acme'").fetchone()[0]
    conn.close()
    assert count == 5


def test_export_idempotent_second_call_empty(tmp_path):
    """Second export returns count=0 — graduated corrections are not re-exported."""
    db_path = _setup_db(tmp_path)
    for i in range(3):
        sid = f"s{i}"
        _insert_message(db_path, session_id=sid)
        _insert_correction(db_path, tenant="acme", session_id=sid, turn=1,
                           wrong_value=f"W{i}", correct_value=f"C{i}")

    mgr = _make_mgr(db_path, tmp_path / "out", threshold=1)
    r1 = mgr.export("acme")
    r2 = mgr.export("acme")
    assert r1.count == 3
    assert r2.count == 0
    assert r2.path is None


def test_export_versions_file_monotonic(tmp_path):
    """dpo_v1.jsonl exists → next export produces dpo_v2.jsonl."""
    db_path = _setup_db(tmp_path)
    out = tmp_path / "out"

    # First batch
    for i in range(2):
        sid = f"a{i}"
        _insert_message(db_path, session_id=sid)
        _insert_correction(db_path, tenant="acme", session_id=sid, turn=1,
                           wrong_value=f"W{i}", correct_value=f"C{i}")
    mgr = _make_mgr(db_path, out, threshold=1)
    r1 = mgr.export("acme")
    assert r1.path.name == "dpo_v1.jsonl"

    # Second batch
    for i in range(2, 4):
        sid = f"b{i}"
        _insert_message(db_path, session_id=sid)
        _insert_correction(db_path, tenant="acme", session_id=sid, turn=1,
                           wrong_value=f"W{i}", correct_value=f"C{i}")
    r2 = mgr.export("acme")
    assert r2.path.name == "dpo_v2.jsonl"


# ---------------------------------------------------------------------------
# Failure isolation
# ---------------------------------------------------------------------------


def test_export_write_failure_rolls_back_graduation(tmp_path, monkeypatch):
    """If JSONL write fails, corrections remain 'pending' — no partial graduation."""
    db_path = _setup_db(tmp_path)
    for i in range(3):
        sid = f"s{i}"
        _insert_message(db_path, session_id=sid)
        _insert_correction(db_path, tenant="acme", session_id=sid, turn=1,
                           wrong_value=f"W{i}", correct_value=f"C{i}")

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    mgr = _make_mgr(db_path, tmp_path / "out", threshold=1)
    monkeypatch.setattr("pathlib.Path.open", _boom)

    with pytest.raises(OSError):
        mgr.export("acme")

    conn = sqlite3.connect(db_path)
    count = conn.execute(
        "SELECT COUNT(*) FROM corrections WHERE tenant='acme' AND status='pending'"
    ).fetchone()[0]
    conn.close()
    assert count == 3


def test_export_parallel_different_tenants(tmp_path):
    """Two threads exporting different tenants simultaneously produce no cross-contamination."""
    db_path = _setup_db(tmp_path)
    out = tmp_path / "out"

    for t in ("alpha", "beta"):
        for i in range(3):
            sid = f"{t}-{i}"
            _insert_message(db_path, session_id=sid, content=f"Message from {t} {i}")
            _insert_correction(db_path, tenant=t, session_id=sid, turn=1,
                               wrong_value=f"{t}-wrong-{i}", correct_value=f"{t}-correct-{i}")

    results = {}
    errors = []

    def _export(tenant):
        try:
            mgr = _make_mgr(db_path, out, threshold=1)
            results[tenant] = mgr.export(tenant)
        except Exception as exc:
            errors.append(exc)

    t1 = threading.Thread(target=_export, args=("alpha",))
    t2 = threading.Thread(target=_export, args=("beta",))
    t1.start(); t2.start()
    t1.join(); t2.join()

    assert not errors, f"Export errors: {errors}"
    assert results["alpha"].count == 3
    assert results["beta"].count == 3

    conn = sqlite3.connect(db_path)
    alpha_pairs = conn.execute(
        "SELECT COUNT(*) FROM training_pairs WHERE tenant='alpha'"
    ).fetchone()[0]
    beta_pairs = conn.execute(
        "SELECT COUNT(*) FROM training_pairs WHERE tenant='beta'"
    ).fetchone()[0]
    conn.close()
    assert alpha_pairs == 3
    assert beta_pairs == 3
