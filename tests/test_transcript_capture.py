"""Tests for TranscriptCapture (Phase 2.7)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from accord_ai.feedback.transcript_capture import (
    CaptureResult,
    TranscriptCapture,
    TranscriptCaptureConfig,
    Turn,
)


def _capturer(tmp_path: Path, enabled: bool = True) -> TranscriptCapture:
    return TranscriptCapture(
        TranscriptCaptureConfig(output_dir=tmp_path / "training_data", enabled=enabled)
    )


def test_capture_appends_per_turn_records(tmp_path):
    cap = _capturer(tmp_path)
    turns = [
        Turn(user_text="We are Acme.", extracted_diff={"business_name": "Acme"}),
        Turn(user_text="EIN is 12-3456789.", extracted_diff={"ein": "12-3456789"}),
    ]
    result = cap.capture(tenant="acme", session_id="s1", turns=turns)

    assert result.count == 2
    assert result.skipped is None
    assert result.path is not None

    lines = result.path.read_text().splitlines()
    assert len(lines) == 2

    rec0 = json.loads(lines[0])
    assert "We are Acme." in rec0["prompt"]
    assert json.loads(rec0["completion"]) == {"business_name": "Acme"}
    assert rec0["metadata"]["tenant"] == "acme"
    assert rec0["metadata"]["session_id"] == "s1"
    assert rec0["metadata"]["turn_idx"] == 0


def test_capture_skips_empty_user_text(tmp_path):
    cap = _capturer(tmp_path)
    turns = [
        Turn(user_text="", extracted_diff={"business_name": "Acme"}),
        Turn(user_text="  ", extracted_diff={"ein": "12-3456789"}),
        Turn(user_text="Real message", extracted_diff={"phone": "555-1234"}),
    ]
    result = cap.capture(tenant="acme", session_id="s1", turns=turns)
    assert result.count == 1


def test_capture_skips_empty_diff(tmp_path):
    cap = _capturer(tmp_path)
    turns = [
        Turn(user_text="Hello there", extracted_diff={}),
        Turn(user_text="Acme Inc", extracted_diff={"business_name": "Acme"}),
    ]
    result = cap.capture(tenant="acme", session_id="s1", turns=turns)
    assert result.count == 1


def test_capture_disabled_returns_skipped(tmp_path):
    cap = _capturer(tmp_path, enabled=False)
    result = cap.capture(
        tenant="acme",
        session_id="s1",
        turns=[Turn(user_text="msg", extracted_diff={"business_name": "Acme"})],
    )
    assert result.count == 0
    assert result.path is None
    assert result.skipped == "disabled"


def test_capture_tenant_isolation(tmp_path):
    cap = _capturer(tmp_path)
    turns = [Turn(user_text="msg", extracted_diff={"business_name": "X"})]

    cap.capture(tenant="acme", session_id="s1", turns=turns)
    cap.capture(tenant="globex", session_id="s2", turns=turns)

    acme_path = tmp_path / "training_data" / "acme" / "transcripts.jsonl"
    globex_path = tmp_path / "training_data" / "globex" / "transcripts.jsonl"
    assert acme_path.exists()
    assert globex_path.exists()
    assert len(acme_path.read_text().splitlines()) == 1
    assert len(globex_path.read_text().splitlines()) == 1


def test_capture_appends_across_calls(tmp_path):
    cap = _capturer(tmp_path)
    turn = Turn(user_text="msg", extracted_diff={"business_name": "X"})

    r1 = cap.capture(tenant="acme", session_id="s1", turns=[turn])
    r2 = cap.capture(tenant="acme", session_id="s2", turns=[turn, turn])

    assert r1.count == 1
    assert r2.count == 2
    lines = r1.path.read_text().splitlines()
    assert len(lines) == 3  # 1 + 2 appended
