"""Tests for POST /finalize SFT capture wiring (Phase 2.7) — 2 tests."""
from __future__ import annotations

import json
import sqlite3
import uuid

import pytest
from fastapi.testclient import TestClient

from accord_ai.api import build_fastapi_app
from accord_ai.app import build_intake_app
from accord_ai.config import Settings
from accord_ai.llm.fake_engine import FakeEngine


def _make_app(tmp_path, monkeypatch):
    training_dir = tmp_path / "training"
    monkeypatch.setenv("ACCORD_AUTH_DISABLED", "true")
    settings = Settings(
        db_path=str(tmp_path / "api.db"),
        filled_pdf_dir=str(tmp_path / "filled"),
        training_data_dir=training_dir,
        enable_transcript_capture=True,
        harness_max_refines=0,
    )
    # start-session triggers a greeting (1 call), then /answer triggers
    # extraction (1 call) + reply (1 call) — 3 total on the shared engine.
    engine = FakeEngine(["Welcome!", {"business_name": "Acme"}, "Got it, noted."])
    intake = build_intake_app(settings, engine=engine, refiner_engine=FakeEngine())
    app = build_fastapi_app(settings, intake=intake)
    return app, training_dir


def _seed_correction(db_path: str, session_id: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute(
        """
        INSERT INTO corrections
            (id, tenant, session_id, turn, field_path,
             wrong_value_json, correct_value_json, correction_type, status, created_at)
        VALUES (?, 'acme', ?, 1, 'business_name', ?, ?, 'value_correction', 'pending', datetime('now'))
        """,
        (uuid.uuid4().hex, session_id, json.dumps("Wrong"), json.dumps("Correct")),
    )
    conn.commit()
    conn.close()


def test_clean_session_triggers_sft_capture(tmp_path, monkeypatch):
    """Finalize of a session with no corrections writes to transcripts.jsonl."""
    app, training_dir = _make_app(tmp_path, monkeypatch)

    with TestClient(app) as client:
        sid = client.post("/start-session").json()["submission_id"]
        # Add a user message so _reconstruct_turns has content
        client.post("/answer", json={"session_id": sid, "message": "We are Acme Inc."})
        r = client.post("/finalize", json={"session_id": sid})

    assert r.status_code == 200
    assert r.json()["status"] == "finalized"

    # Transcript file must exist — tenant defaults to None → "default"
    candidates = list(training_dir.rglob("transcripts.jsonl"))
    assert len(candidates) == 1, f"Expected 1 transcripts.jsonl, found: {candidates}"
    lines = candidates[0].read_text().splitlines()
    assert len(lines) >= 1
    rec = json.loads(lines[0])
    assert "prompt" in rec
    assert "completion" in rec
    assert rec["metadata"]["session_id"] == sid


def test_session_with_correction_skips_sft_capture(tmp_path, monkeypatch):
    """Finalize of a session with corrections logged does NOT write transcripts."""
    # Each _make_app call creates a fresh FakeEngine with 2 queued responses.
    app, training_dir = _make_app(tmp_path, monkeypatch)
    db_path = str(tmp_path / "api.db")

    with TestClient(app) as client:
        sid = client.post("/start-session").json()["submission_id"]
        client.post("/answer", json={"session_id": sid, "message": "We are Acme Inc."})
        # Seed a correction before finalize
        _seed_correction(db_path, sid)
        r = client.post("/finalize", json={"session_id": sid})

    assert r.status_code == 200
    assert r.json()["status"] == "finalized"

    # No transcript written — session had a correction
    candidates = list(training_dir.rglob("transcripts.jsonl"))
    assert len(candidates) == 0
