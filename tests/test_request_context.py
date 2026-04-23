import asyncio
import logging
from pathlib import Path

import pytest

from accord_ai.config import Settings
from accord_ai.logging_config import configure_logging, get_logger
from accord_ai.request_context import (
    ContextInjectionFilter,
    clear_context,
    get_request_id,
    get_session_id,
    get_tenant,
    new_request_id,
    set_context,
)


@pytest.fixture(autouse=True)
def _reset_context():
    """Every test starts with a clean context and leaves a clean context."""
    clear_context()
    yield
    clear_context()


def _flush(logger):
    for h in logger.handlers:
        h.flush()


# --- contextvar basics ---

def test_default_context_is_all_none():
    assert get_request_id() is None
    assert get_tenant() is None
    assert get_session_id() is None


def test_set_context_persists_within_same_task():
    set_context(request_id="abc", tenant="acme", session_id="s1")
    assert get_request_id() == "abc"
    assert get_tenant() == "acme"
    assert get_session_id() == "s1"


def test_clear_context_resets_all():
    set_context(request_id="abc", tenant="acme")
    clear_context()
    assert get_request_id() is None
    assert get_tenant() is None


def test_partial_set_leaves_other_fields_alone():
    set_context(request_id="abc", tenant="acme")
    set_context(session_id="s9")
    assert get_request_id() == "abc"
    assert get_tenant() == "acme"
    assert get_session_id() == "s9"


def test_new_request_id_is_12_hex():
    rid = new_request_id()
    assert isinstance(rid, str)
    assert len(rid) == 12
    assert all(c in "0123456789abcdef" for c in rid)


def test_new_request_id_is_unique():
    assert new_request_id() != new_request_id()


# --- asyncio task isolation (the whole point) ---

def test_context_is_isolated_per_asyncio_task():
    """asyncio.gather runs each coro as its own Task with a copied context.
    Mutations in one task must NOT leak to siblings — core multi-tenant guarantee."""

    async def task(rid):
        set_context(request_id=rid)
        await asyncio.sleep(0)        # yield so other tasks run
        return get_request_id()

    async def run():
        return await asyncio.gather(task("a"), task("b"), task("c"))

    results = asyncio.run(run())
    assert sorted(results) == ["a", "b", "c"]


# --- ContextInjectionFilter ---

def _make_record():
    return logging.LogRecord(
        name="x", level=logging.INFO, pathname="x.py", lineno=1,
        msg="hi", args=None, exc_info=None,
    )


def test_filter_attaches_set_values():
    set_context(request_id="abc123", tenant="acme", session_id="s9")
    rec = _make_record()
    assert ContextInjectionFilter().filter(rec) is True
    assert rec.request_id == "abc123"
    assert rec.tenant == "acme"
    assert rec.session_id == "s9"


def test_filter_uses_dash_when_unset():
    rec = _make_record()
    ContextInjectionFilter().filter(rec)
    assert rec.request_id == "-"
    assert rec.tenant == "-"
    assert rec.session_id == "-"


# --- End-to-end via configure_logging ---

def test_log_output_contains_request_id_when_set(tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    s = Settings()
    configure_logging(s)
    set_context(request_id="req-abc")
    logger = get_logger()
    logger.info("work done")
    _flush(logger)
    content = (Path(s.log_dir) / "app.log").read_text()
    assert "req-abc" in content
    assert "work done" in content


def test_log_output_shows_dash_when_context_unset(tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    s = Settings()
    configure_logging(s)
    logger = get_logger()
    logger.info("orphan")
    _flush(logger)
    content = (Path(s.log_dir) / "app.log").read_text()
    assert "[-]" in content
