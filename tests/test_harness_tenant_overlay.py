"""Verify compose_harness_for_lobs supports per-tenant overlay.

The two-tier harness pattern from research 2026-04-24 (MULTITENANT_ARCHITECTURE.md):

- Tier 1: core.md (shared, engineering-controlled, coupled to adapter training)
- Tier 2: lobs/<lob>.md (per-active-LOB rules; v3's pattern)
- Tier 3: brokers/<tenant>.md (per-broker overlay; v4's multi-tenant extension)

Each tier can be absent without crashing. Cross-tenant isolation is enforced
by reading only the specified tenant's overlay.
"""
from pathlib import Path

import pytest

from accord_ai.extraction.harness_content.v3_harness_snapshot import (
    compose_harness_for_lobs,
)


@pytest.fixture(autouse=True)
def _cleanup_broker_files():
    """Remove any test broker harness files after each test."""
    broker_dir = (
        Path(__file__).resolve().parent.parent
        / "accord_ai/harness/brokers"
    )
    test_files = ["test_tenant_a.md", "test_tenant_b.md", "test_tenant_marker.md"]
    yield
    for f in test_files:
        p = broker_dir / f
        if p.exists():
            p.unlink()


def _write_broker_overlay(tenant: str, content: str) -> None:
    broker_dir = (
        Path(__file__).resolve().parent.parent
        / "accord_ai/harness/brokers"
    )
    broker_dir.mkdir(parents=True, exist_ok=True)
    (broker_dir / f"{tenant}.md").write_text(content)


def test_no_tenant_no_overlay_appears():
    """Without tenant, only core + LOB content is composed."""
    _write_broker_overlay("test_tenant_marker", "# BROKER MARKER XYZ")
    result = compose_harness_for_lobs(active_lobs=["commercial_auto"], tenant=None)
    assert "BROKER MARKER XYZ" not in result
    assert "Core Principles v6.1" in result


def test_tenant_overlay_appended_when_file_exists():
    """When tenant has an overlay file, content is appended."""
    _write_broker_overlay("test_tenant_marker", "# BROKER ACME\n- custom rule")
    result = compose_harness_for_lobs(
        active_lobs=["commercial_auto"],
        tenant="test_tenant_marker",
    )
    assert "BROKER ACME" in result
    assert "Core Principles v6.1" in result  # tier 1
    assert "Vehicle Count vs Employee Count" in result  # tier 2 (CA LOB)
    assert "custom rule" in result


def test_tenant_overlay_missing_silently_skipped():
    """Unknown tenant = no overlay, no crash."""
    result = compose_harness_for_lobs(
        active_lobs=["commercial_auto"],
        tenant="nonexistent_tenant_xyz",
    )
    assert "Core Principles v6.1" in result
    # No "broker" text appears (no overlay file)
    assert "BROKER" not in result


def test_tenant_overlay_empty_file_silently_skipped():
    """Empty overlay file = no effect (no extra whitespace)."""
    _write_broker_overlay("test_tenant_marker", "")
    result = compose_harness_for_lobs(
        active_lobs=["commercial_auto"],
        tenant="test_tenant_marker",
    )
    # Should not contain just-whitespace blocks from empty file
    assert "Core Principles v6.1" in result
    # No trailing junk
    assert not result.endswith("\n\n\n")


def test_tenants_have_isolated_overlays():
    """Tenant A's overlay never appears when tenant B is composed."""
    _write_broker_overlay("test_tenant_a", "# TENANT_A_MARKER")
    _write_broker_overlay("test_tenant_b", "# TENANT_B_MARKER")

    result_a = compose_harness_for_lobs(
        active_lobs=["commercial_auto"], tenant="test_tenant_a"
    )
    result_b = compose_harness_for_lobs(
        active_lobs=["commercial_auto"], tenant="test_tenant_b"
    )

    assert "TENANT_A_MARKER" in result_a
    assert "TENANT_B_MARKER" not in result_a
    assert "TENANT_B_MARKER" in result_b
    assert "TENANT_A_MARKER" not in result_b


def test_three_tier_composition_order():
    """Tier order: core → LOB → broker (core first, broker last)."""
    _write_broker_overlay("test_tenant_marker", "# LAST_MARKER")
    result = compose_harness_for_lobs(
        active_lobs=["commercial_auto"],
        tenant="test_tenant_marker",
    )

    core_pos = result.find("Core Principles v6.1")
    lob_pos = result.find("Vehicle Count vs Employee Count")
    broker_pos = result.find("LAST_MARKER")

    assert core_pos != -1 and lob_pos != -1 and broker_pos != -1
    assert core_pos < lob_pos < broker_pos, (
        f"Expected core → LOB → broker order; got {core_pos} / {lob_pos} / {broker_pos}"
    )


def test_tenant_overlay_without_lobs_still_works():
    """Can have a tenant overlay without any active LOBs."""
    _write_broker_overlay("test_tenant_marker", "# TENANT_ONLY")
    result = compose_harness_for_lobs(active_lobs=None, tenant="test_tenant_marker")
    assert "Core Principles v6.1" in result
    assert "TENANT_ONLY" in result
