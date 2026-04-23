"""Session-transcript capture for SFT training data (Phase 2.7).

Appends per-turn (prompt, completion) pairs from clean finalized sessions to
a per-tenant JSONL file.  One call to capture() per finalized session.

Design notes:
- Append mode — each clean session adds lines to the tenant's rolling file.
- Per-turn records. One session = one record per user turn that had extraction output.
- Schema hint is compact (None → '{}' in prompt); full schema would bloat training context.
- No PII redaction: per-broker training data stays within the broker's isolated adapter.
  Cross-broker seed adapters (Phase 4.11) will add mandatory redaction.
- Template version in metadata for Phase 4.5 curriculum filtering.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

_logger = logging.getLogger(__name__)

SFT_TEMPLATE_VERSION = "v1"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Turn:
    user_text: str
    extracted_diff: Dict  # extraction delta for this turn
    schema_hint: Optional[Dict] = None  # compact schema subset; None → {}


@dataclass(frozen=True)
class CaptureResult:
    count: int
    path: Optional[Path]
    skipped: Optional[str] = None


@dataclass(frozen=True)
class TranscriptCaptureConfig:
    output_dir: Path
    enabled: bool = True


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


class TranscriptCapture:
    """Append per-turn SFT pairs to tenant-scoped JSONL files."""

    def __init__(self, config: TranscriptCaptureConfig) -> None:
        self._cfg = config

    def capture(
        self,
        *,
        tenant: str,
        session_id: str,
        turns: List[Turn],
    ) -> CaptureResult:
        """Append per-turn SFT pairs to training_data/{tenant}/transcripts.jsonl."""
        if not self._cfg.enabled:
            return CaptureResult(count=0, path=None, skipped="disabled")

        tenant_dir = self._cfg.output_dir / tenant
        tenant_dir.mkdir(parents=True, exist_ok=True)
        path = tenant_dir / "transcripts.jsonl"

        count = 0
        with path.open("a") as fh:
            for i, turn in enumerate(turns):
                if not turn.user_text.strip():
                    continue
                if not turn.extracted_diff:
                    continue

                record = {
                    "prompt": self._build_prompt(turn.user_text, turn.schema_hint),
                    "completion": json.dumps(turn.extracted_diff, sort_keys=True),
                    "metadata": {
                        "tenant": tenant,
                        "session_id": session_id,
                        "turn_idx": i,
                        "template_version": SFT_TEMPLATE_VERSION,
                        "captured_at": datetime.now(timezone.utc).isoformat(),
                    },
                }
                fh.write(json.dumps(record) + "\n")
                count += 1

        return CaptureResult(count=count, path=path)

    @staticmethod
    def _build_prompt(user_text: str, schema_hint: Optional[Dict]) -> str:
        schema_line = json.dumps(schema_hint) if schema_hint else "{}"
        return (
            "You are an insurance intake extraction assistant.\n\n"
            "Extract all relevant fields from the user message into JSON "
            "matching the schema.\n\n"
            f"SCHEMA: {schema_line}\n"
            f"USER MESSAGE: {user_text}\n\n"
            "Output JSON only, no commentary."
        )
