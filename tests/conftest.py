"""Shared pytest fixtures.

Deliberately NOT retrofitted to existing tests — they work fine. New tests
from Phase 3 onwards should adopt these.
"""
from __future__ import annotations

from typing import Iterator

import pytest

from accord_ai.core.store import SessionStore


def pytest_configure(config):
    """Register custom markers so pytest doesn't warn on unknown markers."""
    config.addinivalue_line(
        "markers",
        "integration: integration test requiring an external service "
        "(vLLM, etc.); gated by ACCORD_LLM_INTEGRATION=1",
    )


@pytest.fixture
def store(tmp_path) -> Iterator[SessionStore]:
    """A fresh SessionStore backed by a temp DB. Closes after the test."""
    s = SessionStore(str(tmp_path / "store.db"))
    yield s
    s.close()


@pytest.fixture
def frozen_clock(monkeypatch):
    """Deterministic clock for store ops.

    Usage:
        def test_x(store, frozen_clock):
            frozen_clock("2026-04-18T10:00:00+00:00",
                         "2026-04-18T10:00:05+00:00")
            sid = store.create_session()      # consumes the 1st timestamp
            store.append_message(sid, "user", "x")  # consumes the 2nd

    Call `frozen_clock(*timestamps)` once to queue a deterministic sequence;
    the store will pop them in FIFO order. Running out raises StopIteration
    — a good signal that a test's clock expectation is off.
    """
    import accord_ai.core.store as store_module

    queue: list = []

    def provider() -> str:
        return queue.pop(0)

    monkeypatch.setattr(store_module, "_now_utc_iso", provider)

    def enqueue(*timestamps: str) -> None:
        queue.extend(timestamps)

    return enqueue
