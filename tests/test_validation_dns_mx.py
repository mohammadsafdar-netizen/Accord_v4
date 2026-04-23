"""Tests for DnsMxValidator (Phase 1.6.D).

All DNS tests monkey-patch dns.resolver.resolve — no real network calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from accord_ai.schema import Contact, CustomerSubmission
from accord_ai.validation.dns_mx import DnsMxValidator, _MX_CACHE, _MX_LOCK, _check_mx_sync


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sub_with_email(email: str) -> CustomerSubmission:
    return CustomerSubmission(contacts=[Contact(email=email, full_name="Test User")])


def _patch_resolve(return_value):
    return patch("accord_ai.validation.dns_mx._check_mx_sync", return_value=return_value)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dns_mx_valid_domain_returns_no_finding():
    """Domain with MX records → no findings."""
    with _MX_LOCK:
        _MX_CACHE.clear()

    with _patch_resolve(return_value=True):
        result = await DnsMxValidator().run(_sub_with_email("user@example.com"))

    assert result.success is True
    assert result.findings == []


@pytest.mark.asyncio
async def test_dns_mx_nonexistent_domain_returns_error():
    """NXDOMAIN (domain doesn't exist) → error finding."""
    with _MX_LOCK:
        _MX_CACHE.clear()

    with _patch_resolve(return_value=None):  # None means NXDOMAIN from _check_mx_sync
        result = await DnsMxValidator().run(_sub_with_email("user@gmial.com"))

    assert result.success is True
    assert len(result.findings) == 1
    f = result.findings[0]
    assert f.severity == "error"
    assert "gmial.com" in f.message
    assert "does not exist" in f.message.lower()


@pytest.mark.asyncio
async def test_dns_mx_no_mx_records_returns_warning():
    """Domain exists but has no MX records → warning finding."""
    with _MX_LOCK:
        _MX_CACHE.clear()

    with _patch_resolve(return_value=False):  # False means NoAnswer
        result = await DnsMxValidator().run(_sub_with_email("user@no-mx-domain.test"))

    assert result.success is True
    assert len(result.findings) == 1
    f = result.findings[0]
    assert f.severity == "warning"
    assert "no MX records" in f.message


@pytest.mark.asyncio
async def test_dns_mx_no_emails_returns_no_findings():
    """Submission with no contact emails → no findings."""
    with _MX_LOCK:
        _MX_CACHE.clear()

    result = await DnsMxValidator().run(CustomerSubmission(business_name="Acme"))
    assert result.success is True
    assert result.findings == []


@pytest.mark.asyncio
async def test_dns_mx_top_level_email_checked():
    """Submission-level email (not just contacts) is also validated."""
    with _MX_LOCK:
        _MX_CACHE.clear()

    with _patch_resolve(return_value=None):
        sub = CustomerSubmission(email="admin@gmial.com")
        result = await DnsMxValidator().run(sub)

    assert result.success is True
    assert any("gmial.com" in f.message for f in result.findings)
