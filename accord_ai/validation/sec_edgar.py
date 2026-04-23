"""SEC EDGAR public company validator — finalize-only.

Searches EDGAR for publicly traded companies matching the business name.
Hits are rare (<1% of commercial applicants) but rich when they occur.
No API key required; SEC mandates a descriptive User-Agent header.

Search: GET https://efts.sec.gov/LATEST/search-index?q="name"&forms=10-K
Filer:  GET https://data.sec.gov/submissions/CIK{cik:010d}.json

Rate limits: SEC allows 10 req/sec — not a concern for single submissions.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List, Optional

import httpx

from accord_ai.cache import ttl_cached
from accord_ai.validation.types import (
    PrefillPatch,
    ValidationFinding,
    ValidationResult,
)

_logger = logging.getLogger(__name__)

_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
_FILER_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_HTTP_TIMEOUT_S = 5.0
_CACHE_TTL_S = 24 * 3600.0  # 24h


@dataclass
class _EdgarMatch:
    cik: int
    entity_name: str


@dataclass
class _FilerData:
    sic: str
    sic_description: str
    state_of_incorporation: str
    fiscal_year_end: str


def _make_searcher(user_agent: str):
    headers = {"User-Agent": user_agent, "Accept": "application/json"}

    @ttl_cached(ttl_seconds=_CACHE_TTL_S, key=lambda name: name.strip().lower())
    async def _search(name: str) -> Optional[_EdgarMatch]:
        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S) as client:
                r = await client.get(
                    _SEARCH_URL,
                    params={"q": f'"{name.strip()}"', "forms": "10-K"},
                    headers=headers,
                )
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            _logger.debug("SEC EDGAR search failed for %r: %s", name, exc)
            return None

        hits = (data.get("hits") or {}).get("hits") or []
        if not hits:
            return None

        src = hits[0].get("_source") or {}
        # CIK is returned as a string like "0000789019"
        cik_raw = src.get("ciks") or []
        if not cik_raw:
            # Try direct field
            cik_raw = [src.get("cik") or src.get("file_num") or ""]
        cik_str = str(cik_raw[0]).strip().lstrip("0") if cik_raw else ""
        if not cik_str or not cik_str.isdigit():
            return None

        return _EdgarMatch(
            cik=int(cik_str),
            entity_name=src.get("entity_name") or src.get("display_names") or name,
        )

    @ttl_cached(ttl_seconds=_CACHE_TTL_S, key=lambda cik: cik)
    async def _filer(cik: int) -> Optional[_FilerData]:
        url = _FILER_URL.format(cik=cik)
        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S) as client:
                r = await client.get(url, headers=headers)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            _logger.debug("SEC EDGAR filer fetch failed for CIK %d: %s", cik, exc)
            return None

        return _FilerData(
            sic=str(data.get("sic") or ""),
            sic_description=data.get("sicDescription") or data.get("sic_description") or "",
            state_of_incorporation=data.get("stateOfIncorporation") or data.get("state") or "",
            fiscal_year_end=data.get("fiscalYearEnd") or "",
        )

    return _search, _filer


class SecEdgarValidator:
    """Search SEC EDGAR for public-company filings — finalize-only."""

    name: str = "sec_edgar"
    applicable_fields: tuple = ("business_name",)
    inline_eligible: bool = False

    def __init__(self, user_agent: str) -> None:
        self._user_agent = user_agent
        self._search, self._filer = _make_searcher(user_agent)

    async def run(self, submission: Any) -> ValidationResult:
        started = datetime.now(tz=timezone.utc)
        t0 = time.monotonic()
        findings: List[ValidationFinding] = []

        biz_name = getattr(submission, "business_name", None) or (
            submission.get("business_name") if hasattr(submission, "get") else None
        )
        if not biz_name:
            duration_ms = (time.monotonic() - t0) * 1000
            return ValidationResult(
                validator=self.name, ran_at=started,
                duration_ms=duration_ms, success=True, findings=[],
            )

        match = await self._search(biz_name)
        if match is None:
            duration_ms = (time.monotonic() - t0) * 1000
            return ValidationResult(
                validator=self.name, ran_at=started,
                duration_ms=duration_ms, success=True, findings=[],
            )

        filer = await self._filer(match.cik)
        details: dict = {"cik": match.cik, "entity_name": match.entity_name}
        if filer:
            details.update({
                "sic_code": filer.sic,
                "sic_description": filer.sic_description,
                "state_of_incorporation": filer.state_of_incorporation,
                "fiscal_year_end": filer.fiscal_year_end,
            })

        findings.append(ValidationFinding(
            validator=self.name,
            field_path="business_name",
            severity="info",
            message="SEC EDGAR record found (publicly traded)",
            details=details,
        ))

        duration_ms = (time.monotonic() - t0) * 1000
        return ValidationResult(
            validator=self.name, ran_at=started,
            duration_ms=duration_ms, success=True, findings=findings,
        )

    async def prefill(self, submission: Any, just_extracted: dict) -> Optional[PrefillPatch]:
        return None
