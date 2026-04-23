"""Logging config — structured logs, context injection, PII redaction.

P10.0.d — ported v3's production-tested 14-pattern PII filter:
  * Replaces v4's initial 6-pattern set
  * Adds Google Drive tokens (ya29.*), DL, DOB, address, passport, routing
  * Hint prefilter short-circuits ~70% of records before regex scan
  * Group-indexed redaction preserves context keywords while scrubbing values
  * Fixed-tag (no hash) redaction for pure secrets: bearer, ssn, drive_token

Redaction architecture mirrors v3 exactly:
  * PIIRedactionFilter renders record.getMessage() once, redacts both the
    message AND the traceback text (exc_text) in place, then sets
    record.msg = redacted + record.args = () so downstream formatters
    never see the raw content.
  * There is NO separate _PIIRedactingFormatter — a post-format scrub pass
    would redact the log's OWN timestamp (the ISO-DOB pattern matches
    `YYYY-MM-DD`). Keeping redaction inside the filter is correct-by-
    construction.
"""
from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Callable, Optional, Pattern, Tuple

from accord_ai.config import Settings
from accord_ai.request_context import ContextInjectionFilter

_ROOT_LOGGER_NAME = "accord_ai"

_STANDARD_LOGRECORD_ATTRS = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "asctime",
    # ContextInjectionFilter adds these — not 'extras' to be redacted
    "request_id", "tenant", "session_id",
})


# ---------------------------------------------------------------------------
# PII patterns (ported from v3 production — 14 patterns)
# ---------------------------------------------------------------------------
#
# group_index: 0 = replace full match; >0 = replace only that capture group
# so context keywords like "DL:" or "passport" stay legible in logs.
#
# Order matters: context-gated / longer patterns run before bare digits.

_ADDRESS_RE = re.compile(
    r"\b\d{1,6}\s+[A-Z][A-Za-z0-9. ]{2,40}"
    r"\s+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane"
    r"|Way|Ct|Court|Pl|Place|Pkwy|Parkway|Hwy|Highway|Ter|Terrace"
    r"|Cir|Circle)\b\.?"
)

_PII_PATTERNS: Tuple[Tuple[str, Pattern[str], int], ...] = (
    ("bearer",       re.compile(r"(Bearer\s+)[A-Za-z0-9._\-+/=]+", re.IGNORECASE), 0),
    ("drive_token",  re.compile(r"ya29\.[A-Za-z0-9_\-]+"), 0),
    ("email",        re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"), 0),
    ("ssn",          re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), 0),
    ("ein",          re.compile(r"\b\d{2}-\d{7}\b"), 0),
    ("phone",        re.compile(r"\(?\b\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"), 0),
    ("vin",          re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b"), 0),
    # Context-gated: keep the keyword, redact only the captured value.
    ("dl",           re.compile(r"(?i)(?:driver['\u2019]?s?\s*licen[cs]e|\bDLN\b|\bDL\b)[\s:#]*([A-Z0-9]{6,14})\b"), 1),
    ("dob",          re.compile(r"\b(0?[1-9]|1[0-2])[/\-](0?[1-9]|[12]\d|3[01])[/\-](19|20)\d{2}\b"), 0),
    ("dob",          re.compile(r"\b(19|20)\d{2}-(0?[1-9]|1[0-2])-(0?[1-9]|[12]\d|3[01])\b"), 0),
    ("address",      _ADDRESS_RE, 0),
    ("passport",     re.compile(r"(?i)\bpassport\b[\s:#]*([A-Z]?\d{8,9})\b"), 1),
    ("routing",      re.compile(r"(?i)\brouting\b(?:\s*(?:number|no\.?|#))?[\s:#]*(0\d{8})\b"), 1),
)


# Short-circuit prefilter — most log lines have no PII markers at all.
# In v3 production this skips ~70% of records (status messages, request-id
# lines, etc.). Cheap char/keyword scan before compiled regex.

_PII_HINT_CHARS = frozenset("0123456789@")
_PII_HINT_KEYWORDS: Tuple[str, ...] = (
    "Bearer", "bearer", "license", "Licen", "DLN", "DL ", "SSN", "EIN",
    "passport", "Passport", "routing", "Routing", "ya29.",
)


def _has_pii_hint(s: str) -> bool:
    if not s:
        return False
    if any(c in _PII_HINT_CHARS for c in s):
        return True
    return any(k in s for k in _PII_HINT_KEYWORDS)


# Diagnostic counters — ops can watch short_circuited/total ratio.
# Reset on process start. Not thread-atomic by design — approximate is fine.
_REDACT_STATS = {"total": 0, "short_circuited": 0, "scanned": 0}


def _hash8(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]


def redact_pii_text(message):
    """Apply all PII patterns. Public entrypoint — used by
    PIIRedactionFilter, audit payloads, Drive URL logging, any caller that
    writes user-visible strings.

    Fixed-tag replacements (no hash) for pure secrets:
      bearer      → "Bearer <redacted>"
      drive_token → "<drive_token>"
      ssn         → "<ssn>"

    Hashed-tag replacements for identifiers that may appear multiple times
    (so cross-line correlation stays possible without leaking the raw value):
      email, ein, phone, vin, address, dob → "<kind:hash8>"

    Group-gated for context-preserving patterns:
      dl, passport, routing → keyword preserved, captured value → "<kind:hash8>"
    """
    _REDACT_STATS["total"] += 1
    if message is None:
        _REDACT_STATS["short_circuited"] += 1
        return None
    if not message:
        _REDACT_STATS["short_circuited"] += 1
        return message
    if not _has_pii_hint(message):
        _REDACT_STATS["short_circuited"] += 1
        return message
    _REDACT_STATS["scanned"] += 1

    def _sub_factory(kind: str, group_index: int) -> Callable[[re.Match], str]:
        def _replace(m: re.Match) -> str:
            raw = m.group(0)
            if kind == "bearer":
                prefix = m.group(1)
                return f"{prefix}<redacted>"
            if kind == "ssn":
                return "<ssn>"
            if kind == "drive_token":
                return "<drive_token>"
            if group_index > 0:
                captured = m.group(group_index)
                return raw.replace(
                    captured, f"<{kind}:{_hash8(captured)}>", 1,
                )
            return f"<{kind}:{_hash8(raw)}>"
        return _replace

    out = message
    for kind, pattern, group_index in _PII_PATTERNS:
        out = pattern.sub(_sub_factory(kind, group_index), out)
    return out


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

class PIIRedactionFilter(logging.Filter):
    """Scrub known-sensitive substrings from rendered log messages.

    Renders record.getMessage() once, redacts the resulting string + the
    traceback text in place, and re-seats the redacted content as
    record.msg / record.exc_text so downstream formatters never see raw
    values. Also sweeps `extra={...}` fields.

    `_redact` is kept as a public method for backward-compat with existing
    tests; new code should call the module-level `redact_pii_text` directly.
    """

    def _redact(self, text: str) -> str:
        return redact_pii_text(text)

    def filter(self, record: logging.LogRecord) -> bool:
        # Render msg + args into a single string, then redact + re-seat.
        # This is v3's approach — safer than redacting msg and args
        # separately because it handles %-substitutions inside arg values.
        try:
            rendered = record.getMessage()
        except Exception:
            return True
        redacted = redact_pii_text(rendered)
        if redacted != rendered:
            record.msg = redacted
            record.args = ()

        # Pre-render exc_text so the formatter-side formatException path
        # won't re-render a RAW traceback. Redact whatever gets set.
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(
                record.exc_info,
            )
        if record.exc_text:
            record.exc_text = redact_pii_text(record.exc_text)

        # extra={...} fields land as attributes on the record.
        for k, v in list(record.__dict__.items()):
            if k not in _STANDARD_LOGRECORD_ATTRS and isinstance(v, str):
                setattr(record, k, redact_pii_text(v))
        return True


# ---------------------------------------------------------------------------
# configure_logging — filter-only architecture; no formatter subclass
# ---------------------------------------------------------------------------

def configure_logging(settings: Optional[Settings] = None) -> logging.Logger:
    """Configure the 'accord_ai' logger. Idempotent.

    Filter order: ContextInjectionFilter (adds request_id/tenant/session_id)
    runs BEFORE PIIRedactionFilter (scrubs msg/args/extras + exc_text).
    """
    if settings is None:
        settings = Settings()

    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(_ROOT_LOGGER_NAME)

    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    for f in list(logger.filters):
        logger.removeFilter(f)

    logger.setLevel(settings.log_level.upper())
    logger.propagate = False

    # Plain Formatter — no post-format redaction pass. The filter owns
    # redaction (msg + args + exc_text), and the formatter's default
    # behavior reads the already-redacted exc_text without re-rendering
    # the traceback.
    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Filters live on the handler, not the logger — so records from child
    # loggers that propagate up also get their context injected and redacted.
    # (Logger-level filters only run at the origin logger.)
    fh = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.addFilter(ContextInjectionFilter())
    if settings.pii_redaction:
        fh.addFilter(PIIRedactionFilter())
    logger.addHandler(fh)

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    if not name:
        return logging.getLogger(_ROOT_LOGGER_NAME)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
