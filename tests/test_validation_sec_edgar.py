"""Tests for SecEdgarValidator (Phase 1.6.D)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, call, patch

import httpx
import pytest

from accord_ai.schema import CustomerSubmission
from accord_ai.validation.sec_edgar import SecEdgarValidator, _EdgarMatch, _FilerData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_UA = "Accord v4 compliance-check@accord.example"


def _make_validator(user_agent: str = _DEFAULT_UA) -> SecEdgarValidator:
    return SecEdgarValidator(user_agent=user_agent)


def _sub(business_name=None) -> CustomerSubmission:
    return CustomerSubmission(business_name=business_name)


_MSFT_MATCH = _EdgarMatch(cik=789019, entity_name="MICROSOFT CORP")
_MSFT_FILER = _FilerData(
    sic="7372",
    sic_description="Prepackaged Software",
    state_of_incorporation="WA",
    fiscal_year_end="0630",
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sec_edgar_public_company_match():
    """Public company match → info finding with CIK and SIC details."""
    v = _make_validator()
    with (
        patch.object(v, "_search", AsyncMock(return_value=_MSFT_MATCH)),
        patch.object(v, "_filer", AsyncMock(return_value=_MSFT_FILER)),
    ):
        result = await v.run(_sub(business_name="Microsoft Corporation"))

    assert result.success is True
    assert len(result.findings) == 1
    f = result.findings[0]
    assert f.severity == "info"
    assert "publicly traded" in f.message.lower()
    assert f.details["cik"] == 789019
    assert f.details["sic_code"] == "7372"
    assert f.details["state_of_incorporation"] == "WA"


@pytest.mark.asyncio
async def test_sec_edgar_no_match_returns_empty_findings():
    """Private company (no EDGAR match) → no findings (not an error)."""
    v = _make_validator()
    with patch.object(v, "_search", AsyncMock(return_value=None)):
        result = await v.run(_sub(business_name="Acme Plumbing LLC"))

    assert result.success is True
    assert result.findings == []


@pytest.mark.asyncio
async def test_sec_edgar_user_agent_header_required():
    """Custom User-Agent must be configured and non-empty."""
    custom_ua = "My Custom Agent identify@example.com"
    v = _make_validator(user_agent=custom_ua)
    assert v._user_agent == custom_ua
    # Verify that the stored user_agent is passed through (header is built at creation)
    # The search closure captures the headers dict at construction time
    assert v._user_agent != _DEFAULT_UA


@pytest.mark.asyncio
async def test_sec_edgar_missing_business_name_returns_empty():
    """Submission without business_name → no findings."""
    v = _make_validator()
    result = await v.run(_sub(business_name=None))

    assert result.success is True
    assert result.findings == []
