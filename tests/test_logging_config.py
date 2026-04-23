from pathlib import Path

import pytest

from accord_ai.config import Settings
from accord_ai.logging_config import configure_logging, get_logger


def _flush(logger):
    for h in logger.handlers:
        h.flush()


def test_configure_creates_log_dir(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    monkeypatch.setenv("LOG_DIR", str(log_dir))
    assert not log_dir.exists()
    configure_logging(Settings())
    assert log_dir.exists()


def test_writes_to_app_log(tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    s = Settings()
    configure_logging(s)
    logger = get_logger()
    logger.info("hello world")
    _flush(logger)
    app_log = Path(s.log_dir) / "app.log"
    assert app_log.exists()
    content = app_log.read_text()
    assert "hello world" in content
    assert "INFO" in content


def test_log_level_filters_below_threshold(tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    s = Settings()
    configure_logging(s)
    logger = get_logger()
    logger.info("should-be-suppressed")
    logger.warning("should-appear")
    _flush(logger)
    content = (Path(s.log_dir) / "app.log").read_text()
    assert "should-be-suppressed" not in content
    assert "should-appear" in content


def test_configure_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    configure_logging(Settings())
    logger = get_logger()
    n1 = len(logger.handlers)
    configure_logging(Settings())
    n2 = len(logger.handlers)
    assert n1 == n2 == 1   # handlers never stack


def test_get_logger_returns_child_under_root(tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    configure_logging(Settings())
    root = get_logger()
    child = get_logger("orchestrator")
    assert child.name == "accord_ai.orchestrator"
    assert child.parent is root


def test_child_logger_output_reaches_file(tmp_path, monkeypatch):
    """A child logger's messages must land in app.log via the root handler."""
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    s = Settings()
    configure_logging(s)
    child = get_logger("extractor")
    child.warning("from-child")
    _flush(child)
    _flush(get_logger())
    content = (Path(s.log_dir) / "app.log").read_text()
    assert "from-child" in content
    assert "accord_ai.extractor" in content
