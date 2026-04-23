from pathlib import Path

import pytest

from accord_ai.config import Settings
from accord_ai.logging_config import PIIRedactionFilter, configure_logging, get_logger


def _flush_all(logger):
    for h in logger.handlers:
        h.flush()


def _log_and_read(tmp_path, monkeypatch, message, *, pii_on=True):
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("PII_REDACTION", "true" if pii_on else "false")
    s = Settings()
    configure_logging(s)
    logger = get_logger()
    logger.info(message)
    _flush_all(logger)
    return (Path(s.log_dir) / "app.log").read_text()


# --- standalone filter unit tests ---

def test_filter_redacts_email_in_string():
    f = PIIRedactionFilter()
    assert "a@b.com" not in f._redact("contact: a@b.com now")


def test_same_value_produces_same_hash():
    f = PIIRedactionFilter()
    assert f._redact("a@b.com") == f._redact("a@b.com")


def test_different_values_produce_different_hashes():
    f = PIIRedactionFilter()
    assert f._redact("a@b.com") != f._redact("c@d.com")


# --- end-to-end file output ---

def test_redacts_email_end_to_end(tmp_path, monkeypatch):
    content = _log_and_read(tmp_path, monkeypatch, "user a@b.com signed in")
    assert "a@b.com" not in content
    assert "<email:" in content


def test_redacts_ein(tmp_path, monkeypatch):
    content = _log_and_read(tmp_path, monkeypatch, "EIN 12-3456789 verified")
    assert "12-3456789" not in content
    assert "<ein:" in content


def test_redacts_ssn(tmp_path, monkeypatch):
    content = _log_and_read(tmp_path, monkeypatch, "ssn=123-45-6789 on record")
    assert "123-45-6789" not in content
    # P10.0.d: SSN uses a fixed tag (no hash). Short values hashed with
    # sha256-8 are rainbow-tableable; fixed tag = zero leak.
    assert "<ssn>" in content


def test_redacts_phone(tmp_path, monkeypatch):
    content = _log_and_read(tmp_path, monkeypatch, "call 555-123-4567 please")
    assert "555-123-4567" not in content
    assert "<phone:" in content


def test_redacts_vin(tmp_path, monkeypatch):
    content = _log_and_read(tmp_path, monkeypatch, "vehicle VIN=1FTFW1E50NFA12345")
    assert "1FTFW1E50NFA12345" not in content
    assert "<vin:" in content


def test_redacts_bearer_token(tmp_path, monkeypatch):
    content = _log_and_read(tmp_path, monkeypatch, "Authorization: Bearer abc.def-123_xyz")
    assert "abc.def-123_xyz" not in content
    # P10.0.d: bearer tokens get a fixed redaction that preserves the
    # "Bearer" prefix so the log line is still legibly an auth header.
    assert "Bearer <redacted>" in content


def test_non_pii_text_is_untouched(tmp_path, monkeypatch):
    content = _log_and_read(tmp_path, monkeypatch, "orchestrator ready")
    assert "orchestrator ready" in content
    assert "<email:" not in content


def test_disabled_when_setting_off(tmp_path, monkeypatch):
    content = _log_and_read(tmp_path, monkeypatch, "raw a@b.com", pii_on=False)
    assert "a@b.com" in content
    assert "<email:" not in content


def test_redaction_applies_to_printf_style_args(tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    s = Settings()
    configure_logging(s)
    logger = get_logger()
    logger.info("user %s signed in", "a@b.com")
    _flush_all(logger)
    content = (Path(s.log_dir) / "app.log").read_text()
    assert "a@b.com" not in content
    assert "<email:" in content


# --- regression tests for review findings ---

def test_phone_without_separators_not_matched(tmp_path, monkeypatch):
    """Bare 10-digit numbers (timestamps, IDs) must NOT be tagged as phones."""
    content = _log_and_read(tmp_path, monkeypatch, "tx 1710000000 done")
    assert "1710000000" in content
    assert "<phone:" not in content


def test_bearer_before_email_ordering(tmp_path, monkeypatch):
    """Bearer pattern must run before email — pattern-order regression guard."""
    content = _log_and_read(tmp_path, monkeypatch, "Authorization: Bearer xyz123")
    assert "Bearer <redacted>" in content
    assert "<email:" not in content


def test_ssn_not_confused_with_phone(tmp_path, monkeypatch):
    """SSN (3-2-4) and phone (3-3-4) must not clobber each other."""
    content = _log_and_read(tmp_path, monkeypatch, "ssn 123-45-6789")
    assert "<ssn>" in content
    assert "<phone:" not in content


def test_exception_traceback_redacted(tmp_path, monkeypatch):
    """Tracebacks go through formatException (after filter). Formatter-level pass must catch them."""
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    s = Settings()
    configure_logging(s)
    logger = get_logger()
    try:
        raise ValueError("user alice@example.com rejected")
    except ValueError:
        logger.exception("processing failed")
    _flush_all(logger)
    content = (Path(s.log_dir) / "app.log").read_text()
    assert "alice@example.com" not in content
    assert "<email:" in content


def test_extra_field_redacted(tmp_path, monkeypatch):
    """logger.info(..., extra={...}) values must also be redacted."""
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    s = Settings()
    configure_logging(s)
    logger = get_logger()
    logger.info("auth event", extra={"user_email": "alice@example.com"})
    _flush_all(logger)
    content = (Path(s.log_dir) / "app.log").read_text()
    assert "alice@example.com" not in content


# ---------------------------------------------------------------------------
# P10.0.d — new patterns ported from v3
# ---------------------------------------------------------------------------

def test_redact_drive_token():
    from accord_ai.logging_config import redact_pii_text
    out = redact_pii_text(
        "authorized with ya29.A0AfH6SMBxZ-very-long-token-here_42"
    )
    assert "ya29." not in out
    assert "<drive_token>" in out


def test_redact_bearer_preserves_prefix():
    from accord_ai.logging_config import redact_pii_text
    out = redact_pii_text(
        "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6..."
    )
    assert "eyJhbGci" not in out
    assert "Bearer <redacted>" in out


def test_redact_bearer_case_insensitive():
    from accord_ai.logging_config import redact_pii_text
    # Both capitalizations should redact; whatever case was used on input is fine.
    out = redact_pii_text("bearer SECRETTOKEN")
    assert "SECRETTOKEN" not in out


def test_redact_ssn_uses_fixed_tag_no_hash():
    """SSNs must not leak even via hash-correlation — fixed tag."""
    from accord_ai.logging_config import redact_pii_text
    out1 = redact_pii_text("SSN 123-45-6789")
    out2 = redact_pii_text("SSN 987-65-4321")
    assert out1 == out2     # both redact to the exact same string
    assert "<ssn>" in out1


def test_redact_dl_preserves_keyword():
    """DL pattern is context-gated — keep the 'DL:' keyword so operators
    can see something about the log context, but scrub the number."""
    from accord_ai.logging_config import redact_pii_text
    out = redact_pii_text("driver license: D12345678")
    assert "D12345678" not in out
    assert "driver license" in out.lower()
    assert "<dl:" in out


def test_redact_dl_alternate_keywords():
    from accord_ai.logging_config import redact_pii_text
    # DLN variant
    out = redact_pii_text("DLN AB123456")
    assert "AB123456" not in out
    # DL variant (with space to disambiguate from "DL" inside other words)
    out = redact_pii_text("DL D7654321")
    assert "D7654321" not in out


def test_redact_dob_mdy():
    from accord_ai.logging_config import redact_pii_text
    for date_str in ("3/15/1988", "03/15/1988", "12/31/2000"):
        out = redact_pii_text(f"born {date_str}")
        assert date_str not in out, f"failed for {date_str!r}: {out}"
        assert "<dob:" in out


def test_redact_dob_iso():
    from accord_ai.logging_config import redact_pii_text
    out = redact_pii_text("born 1988-03-15")
    assert "1988-03-15" not in out
    assert "<dob:" in out


def test_redact_address():
    from accord_ai.logging_config import redact_pii_text
    out = redact_pii_text("lives at 123 Main Street")
    assert "123 Main" not in out
    assert "<address:" in out


def test_redact_passport_context_gated():
    from accord_ai.logging_config import redact_pii_text
    out = redact_pii_text("passport: A12345678")
    assert "A12345678" not in out
    assert "passport" in out.lower()       # keyword preserved
    assert "<passport:" in out


def test_redact_routing_context_gated():
    from accord_ai.logging_config import redact_pii_text
    out = redact_pii_text("routing number: 021000021")
    assert "021000021" not in out
    assert "routing" in out.lower()
    assert "<routing:" in out


def test_redact_routing_not_matched_without_context():
    """Bare 9-digit numbers without 'routing' keyword must NOT match —
    prevents false positives on order IDs, etc."""
    from accord_ai.logging_config import redact_pii_text
    out = redact_pii_text("order id 021000021 was shipped")
    assert "021000021" in out


# ---------------------------------------------------------------------------
# Hint prefilter — performance behavior
# ---------------------------------------------------------------------------

def test_hint_prefilter_short_circuits_non_pii():
    from accord_ai.logging_config import _REDACT_STATS, redact_pii_text
    before = dict(_REDACT_STATS)
    redact_pii_text("no sensitive content here at all")
    after = _REDACT_STATS
    assert after["short_circuited"] == before["short_circuited"] + 1
    assert after["scanned"] == before["scanned"]


def test_hint_prefilter_scans_when_marker_present():
    from accord_ai.logging_config import _REDACT_STATS, redact_pii_text
    before = dict(_REDACT_STATS)
    redact_pii_text("email a@b.com")
    after = _REDACT_STATS
    assert after["scanned"] == before["scanned"] + 1


def test_empty_string_short_circuits():
    from accord_ai.logging_config import redact_pii_text
    assert redact_pii_text("") == ""
    assert redact_pii_text(None) is None   # defensive — not in hot path


# ---------------------------------------------------------------------------
# Module function is what callers should use for audit payloads etc.
# ---------------------------------------------------------------------------

def test_redact_pii_text_is_public_module_function():
    """Callers outside the logging pipeline (audit events, Drive URL logging)
    need to use this without instantiating a logging.Filter."""
    from accord_ai.logging_config import redact_pii_text
    assert callable(redact_pii_text)
    assert redact_pii_text("hello") == "hello"   # no-op on clean text


# ---------------------------------------------------------------------------
# Log timestamp NOT self-redacted — the bug v3's filter-only approach avoids
# ---------------------------------------------------------------------------

def test_log_timestamp_not_self_redacted_as_dob(tmp_path, monkeypatch):
    """With the ISO-DOB pattern, a formatter-level post-redaction pass
    would scrub the log's own '%(asctime)s' (e.g. '2026-04-18'). The v3
    architecture keeps redaction inside the filter, so timestamps survive.
    """
    import re as _re
    content = _log_and_read(tmp_path, monkeypatch, "some benign event")
    # Expect a real ISO date at the start of the line, not "<dob:...>".
    assert _re.search(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} ", content), (
        f"log timestamp was redacted (likely by a formatter-level pass): "
        f"{content!r}"
    )
