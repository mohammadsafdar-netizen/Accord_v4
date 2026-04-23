"""JSON validity tracker for Step 25 extraction mode experiment.

Counts JSON parse outcomes per extraction call and writes per-turn JSONL to
logs/step25_validity.jsonl. Aggregated counts via get_summary() for eval-end reporting.

Note on `valid_after_retry`: the extractor does not currently implement a retry
path — parse either succeeds or raises ExtractionOutputError. The after-retry
counters are wired for future use but will always be 0 in the current implementation.

Thread-safety: int counters are non-atomic under concurrent asyncio load (GIL-safe
for eval runs with concurrency=1; do not use under multi-coroutine API load without
a threading.Lock). The singleton TRACKER is acceptable for eval use only.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class JsonValidityTracker:
    total_attempts: int = 0
    json_valid_first_try: int = 0
    json_valid_after_retry: int = 0
    json_invalid_after_retry: int = 0
    _log_path: Optional[Path] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._log_path is None:
            self._log_path = Path("logs/step25_validity.jsonl")

    def record(
        self,
        *,
        valid_first_try: bool,
        valid_after_retry: Optional[bool] = None,
        mode: str = "xgrammar",
        harness: str = "none",
        extracted_field_count: int = 0,
    ) -> None:
        self.total_attempts += 1
        if valid_first_try:
            self.json_valid_first_try += 1
        elif valid_after_retry is True:
            self.json_valid_after_retry += 1
        elif valid_after_retry is False:
            self.json_invalid_after_retry += 1

        entry = {
            "ts": time.time(),
            "mode": mode,
            "harness": harness,
            "valid_first_try": valid_first_try,
            "valid_after_retry": valid_after_retry,
            "extracted_field_count": extracted_field_count,
        }
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            pass  # non-fatal — eval still runs

    def get_summary(self) -> dict:
        total = self.total_attempts
        return {
            "total_attempts": total,
            "json_valid_first_try": self.json_valid_first_try,
            "json_valid_after_retry": self.json_valid_after_retry,
            "json_invalid_after_retry": self.json_invalid_after_retry,
            "first_try_rate": self.json_valid_first_try / total if total else 0.0,
            "post_retry_rate": (
                (self.json_valid_first_try + self.json_valid_after_retry) / total
                if total else 0.0
            ),
        }

    def reset(self) -> None:
        self.total_attempts = 0
        self.json_valid_first_try = 0
        self.json_valid_after_retry = 0
        self.json_invalid_after_retry = 0


# Module-level singleton — shared across all Extractor instances in one process.
# Reset between eval variants via tracker.reset().
TRACKER = JsonValidityTracker()
