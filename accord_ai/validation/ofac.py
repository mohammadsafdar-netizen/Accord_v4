"""OFAC SDN validator — local fuzzy screening against the US Treasury SDN list.

Port of accord_ai_v3/accord_ai/validation/ofac_sdn.py adapted for v4's
CustomerSubmission schema and the Validator protocol.

Scoring: token-set Jaccard similarity.
  score > 0.9  → error  (likely sanctions hit — must review)
  score > 0.6  → warning (possible match — review recommended)
  score <= 0.6 → clear

CSV columns (per OFAC spec, no header row):
  ent_num, SDN_Name, SDN_Type, Program, Title, Call_Sign, Vess_type,
  Tonnage, GRT, Vess_flag, Vess_owner, Remarks

SDN_Type values: "individual" / "entity" / "aircraft" / "vessel"
"""

from __future__ import annotations

import csv
import logging
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

import httpx

from .types import ValidationFinding, ValidationResult

_logger = logging.getLogger(__name__)

SDN_URL = "https://www.treasury.gov/ofac/downloads/sdn.csv"

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "ofac"
_CACHE_FILE = _CACHE_DIR / "sdn.csv"
_CACHE_TTL_SECONDS = 24 * 3600

# Minimum Jaccard score thresholds
_ERROR_THRESHOLD = 0.9
_WARNING_THRESHOLD = 0.6

_CORP_SUFFIXES = {
    "llc", "l.l.c.", "inc", "inc.", "incorporated", "corp", "corp.",
    "corporation", "co", "co.", "ltd", "ltd.", "limited", "gmbh", "ag",
    "sa", "s.a.", "sarl", "srl", "pllc", "lp", "l.p.", "llp", "plc",
    "company", "group", "holdings", "international", "intl", "global",
    "enterprises", "partners", "associates",
}
_MIN_TOKEN_LEN = 3


def _normalize(name: str) -> str:
    if not name:
        return ""
    s = name.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    tokens = [t for t in s.split() if t not in _CORP_SUFFIXES]
    return " ".join(tokens)


def _tokens_of(name: str) -> set[str]:
    return {t for t in _normalize(name).split() if len(t) >= _MIN_TOKEN_LEN}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


@dataclass
class _SDNEntry:
    ent_num: str
    name: str
    sdn_type: str
    program: str
    _normalized: str = field(default="", repr=False)
    _tokens: set[str] = field(default_factory=set, repr=False)


@dataclass
class _SDNMatch:
    name: str
    ent_num: str
    sdn_type: str
    program: str
    score: float
    match_type: str   # "exact" / "fuzzy"


class _SdnIndex:
    __slots__ = ("entries", "_token_to_idx", "_exact_to_idx")

    def __init__(self) -> None:
        self.entries: list[_SDNEntry] = []
        self._token_to_idx: dict[str, set[int]] = defaultdict(set)
        self._exact_to_idx: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def __bool__(self) -> bool:
        return bool(self.entries)

    def add(self, entry: _SDNEntry) -> None:
        idx = len(self.entries)
        self.entries.append(entry)
        for token in entry._tokens:
            self._token_to_idx[token].add(idx)
        if entry._normalized:
            self._exact_to_idx.setdefault(entry._normalized, idx)

    def exact(self, normalized_name: str) -> Optional[_SDNEntry]:
        idx = self._exact_to_idx.get(normalized_name)
        return self.entries[idx] if idx is not None else None

    def candidates(self, query_tokens: set[str]) -> list[_SDNEntry]:
        if not query_tokens:
            return []
        idxs: set[int] = set()
        for t in query_tokens:
            hit = self._token_to_idx.get(t)
            if hit:
                idxs.update(hit)
        return [self.entries[i] for i in idxs]


_INDEX_LOCK = threading.Lock()
_INDEX: _SdnIndex = _SdnIndex()
_INDEX_LOADED_AT: float = 0.0


def _cache_is_fresh() -> bool:
    if not _CACHE_FILE.exists():
        return False
    return (time.time() - _CACHE_FILE.stat().st_mtime) < _CACHE_TTL_SECONDS


def download_sdn(timeout: float = 30.0) -> bool:
    """Download the SDN CSV to the cache dir. Returns True on success."""
    global _CACHE_DIR, _CACHE_FILE
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        _CACHE_DIR = Path("/tmp/ofac_cache")
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE = _CACHE_DIR / "sdn.csv"

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            r = client.get(SDN_URL)
        if r.status_code != 200:
            _logger.warning("OFAC SDN download failed: HTTP %d", r.status_code)
            return False
        head = r.content[:50].decode("latin-1", errors="replace")
        if not head or not head[0].isdigit():
            _logger.warning("OFAC SDN response doesn't look like CSV: %r", head[:80])
            return False
        _CACHE_FILE.write_bytes(r.content)
        _logger.info("OFAC SDN downloaded (%s bytes)", f"{_CACHE_FILE.stat().st_size:,}")
        return True
    except Exception as exc:
        _logger.warning("OFAC SDN download error: %s", exc)
        return False


def _load_index(csv_path: Optional[Path] = None) -> _SdnIndex:
    path = csv_path or _CACHE_FILE
    index = _SdnIndex()
    with open(path, "r", encoding="latin-1", newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if len(row) < 4:
                continue
            clean = [c.strip().replace("-0-", "").strip() for c in row]
            name = clean[1]
            sdn_type = clean[2].lower() if clean[2] else ""
            program = clean[3]
            if not name:
                continue
            entry = _SDNEntry(
                ent_num=clean[0], name=name, sdn_type=sdn_type, program=program,
            )
            entry._normalized = _normalize(name)
            entry._tokens = _tokens_of(name)
            index.add(entry)
    _logger.info("OFAC SDN index loaded: %d entries", len(index))
    return index


def _ensure_index() -> bool:
    global _INDEX, _INDEX_LOADED_AT
    with _INDEX_LOCK:
        if _INDEX and (time.time() - _INDEX_LOADED_AT) < _CACHE_TTL_SECONDS:
            return True
        need_download = not _cache_is_fresh()

    if need_download:
        if not download_sdn() and not _CACHE_FILE.exists():
            return False

    try:
        new_index = _load_index()
    except Exception as exc:
        _logger.warning("OFAC SDN index load failed: %s", exc)
        return False

    if not new_index:
        return False

    with _INDEX_LOCK:
        _INDEX = new_index
        _INDEX_LOADED_AT = time.time()
    return True


def _check_name(
    name: str,
    entity_types: tuple[str, ...] = ("entity", "individual", ""),
) -> Optional[_SDNMatch]:
    """Screen a single name. Returns best match or None."""
    if not name or not name.strip():
        return None

    query_norm = _normalize(name)
    query_tokens = _tokens_of(name)
    if not query_tokens:
        return None

    with _INDEX_LOCK:
        index = _INDEX

    # Exact fast path
    exact_hit = index.exact(query_norm)
    if exact_hit is not None and (not entity_types or exact_hit.sdn_type in entity_types):
        return _SDNMatch(
            name=exact_hit.name, ent_num=exact_hit.ent_num,
            sdn_type=exact_hit.sdn_type, program=exact_hit.program,
            score=1.0, match_type="exact",
        )

    candidates = index.candidates(query_tokens)
    best: Optional[_SDNMatch] = None
    for entry in candidates:
        if entity_types and entry.sdn_type not in entity_types:
            continue
        if not entry._tokens:
            continue
        j = _jaccard(query_tokens, entry._tokens)
        if j >= _WARNING_THRESHOLD:
            if best is None or j > best.score:
                best = _SDNMatch(
                    name=entry.name, ent_num=entry.ent_num,
                    sdn_type=entry.sdn_type, program=entry.program,
                    score=j, match_type="fuzzy",
                )
    return best


def load_index_from_file(csv_path: Path) -> None:
    """Load index from a specific CSV path (used in tests with synthetic fixtures)."""
    global _INDEX, _INDEX_LOADED_AT
    new_index = _load_index(csv_path)
    with _INDEX_LOCK:
        _INDEX = new_index
        _INDEX_LOADED_AT = time.time()


class OFACValidator:
    """Screen business_name and contacts[].full_name against the OFAC SDN list."""

    name: str = "ofac"
    applicable_fields: List[str] = ["business_name", "contacts"]

    async def run(self, submission: Any) -> ValidationResult:
        import asyncio
        from functools import partial

        started = datetime.now(tz=timezone.utc)
        t0 = time.monotonic()
        findings: List[ValidationFinding] = []

        # Run blocking I/O in thread pool
        loop = asyncio.get_event_loop()
        try:
            ok = await loop.run_in_executor(None, _ensure_index)
        except Exception as exc:
            duration_ms = (time.monotonic() - t0) * 1000
            return ValidationResult(
                validator=self.name, ran_at=started,
                duration_ms=duration_ms, success=False,
                error=f"index load failed: {exc}",
            )

        if not ok:
            duration_ms = (time.monotonic() - t0) * 1000
            return ValidationResult(
                validator=self.name, ran_at=started,
                duration_ms=duration_ms, success=False,
                error="OFAC index unavailable (no cache file and download failed)",
            )

        sub_dict = submission.model_dump() if hasattr(submission, "model_dump") else {}
        names_to_check: list[tuple[str, str]] = []  # (field_path, name)

        biz_name = sub_dict.get("business_name") or ""
        if biz_name:
            names_to_check.append(("business_name", biz_name))

        for i, contact in enumerate(sub_dict.get("contacts") or []):
            if isinstance(contact, dict):
                fn = contact.get("full_name") or ""
            else:
                fn = getattr(contact, "full_name", None) or ""
            if fn:
                names_to_check.append((f"contacts[{i}].full_name", fn))

        for field_path, name in names_to_check:
            match = await loop.run_in_executor(None, partial(_check_name, name))
            if match is None:
                continue
            if match.score > _ERROR_THRESHOLD:
                findings.append(ValidationFinding(
                    validator=self.name,
                    field_path=field_path,
                    severity="error",
                    message=f"OFAC SDN match: '{match.name}' ({match.program})",
                    details={
                        "sdn_name": match.name, "ent_num": match.ent_num,
                        "program": match.program, "score": round(match.score, 3),
                        "match_type": match.match_type,
                    },
                ))
            elif match.score > _WARNING_THRESHOLD:
                findings.append(ValidationFinding(
                    validator=self.name,
                    field_path=field_path,
                    severity="warning",
                    message=f"Possible OFAC SDN match: '{match.name}' — review required",
                    details={
                        "sdn_name": match.name, "ent_num": match.ent_num,
                        "program": match.program, "score": round(match.score, 3),
                        "match_type": match.match_type,
                    },
                ))

        duration_ms = (time.monotonic() - t0) * 1000
        return ValidationResult(
            validator=self.name,
            ran_at=started,
            duration_ms=duration_ms,
            success=True,
            findings=findings,
        )
