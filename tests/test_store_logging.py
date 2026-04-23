"""Integration test — store writes produce DEBUG log lines."""
import logging
from pathlib import Path

from accord_ai.config import Settings
from accord_ai.logging_config import configure_logging


def test_store_writes_emit_debug_lines(store, tmp_path, monkeypatch):
    """create/update/finalize produce DEBUG lines with session_id + tenant."""
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    s = Settings()
    configure_logging(s)

    sid = store.create_session(tenant="acme")
    store.finalize_session(sid, tenant="acme")

    for h in logging.getLogger("accord_ai").handlers:
        h.flush()

    content = (Path(s.log_dir) / "app.log").read_text()
    assert "session created" in content
    assert "session transitioned" in content
    assert "acme" in content
    assert sid in content


def test_fixtures_provide_isolated_store(store):
    """The `store` fixture hands a working, empty store."""
    assert store.list_sessions() == []
    sid = store.create_session()
    assert len(store.list_sessions()) == 1


def test_frozen_clock_queues_timestamps(store, frozen_clock):
    """frozen_clock lets tests assert deterministic timestamps."""
    from datetime import datetime, timezone

    frozen_clock(
        "2026-04-18T10:00:00+00:00",
        "2026-04-18T10:00:05+00:00",
    )
    sid = store.create_session()
    store.finalize_session(sid)

    session = store.get_session(sid)
    assert session.created_at == datetime(2026, 4, 18, 10, 0, 0, tzinfo=timezone.utc)
    assert session.updated_at == datetime(2026, 4, 18, 10, 0, 5, tzinfo=timezone.utc)
