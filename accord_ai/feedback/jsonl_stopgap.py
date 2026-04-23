"""JSONL-stopgap persistence for /correction and /feedback.

Phase 2.2 replaces these writes with CorrectionCollector.record() + a proper
SQLite table. Until then this keeps the data — one file per tenant so the
future migration is a single-file → INSERT loop per tenant.

PII is redacted before writing so the log files are safe to grep / tail
in production without exposing raw user data.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from accord_ai.logging_config import redact_pii_text


def write_correction(
    payload: dict[str, Any],
    tenant: str,
    log_dir: str = "logs",
) -> str:
    """Redact + persist a correction record. Returns the assigned UUID."""
    record = _build_record(payload, tenant)
    _append(log_dir, f"corrections_incoming_{tenant}.jsonl", record)
    return record["id"]


def write_feedback(
    payload: dict[str, Any],
    tenant: str,
    log_dir: str = "logs",
) -> str:
    """Redact + persist a feedback record. Returns the assigned UUID."""
    record = _build_record(payload, tenant)
    _append(log_dir, f"feedback_incoming_{tenant}.jsonl", record)
    return record["id"]


def _build_record(payload: dict[str, Any], tenant: str) -> dict[str, Any]:
    redacted = _redact_dict(payload)
    redacted["id"] = str(uuid.uuid4())
    redacted["captured_at"] = datetime.now(tz=timezone.utc).isoformat()
    redacted["tenant"] = tenant
    return redacted


def _redact_dict(d: Any) -> Any:
    """Recursively redact PII from string values in a dict/list/scalar."""
    if isinstance(d, dict):
        return {k: _redact_dict(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_redact_dict(v) for v in d]
    if isinstance(d, str):
        result = redact_pii_text(d)
        return result if result is not None else d
    return d


def _append(log_dir: str, filename: str, record: dict[str, Any]) -> None:
    path = Path(log_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
