"""Verify v3's curated harness content is correctly ported into v4.

Context: Step 25 research (2026-04-22) identified we were porting the WRONG
file (legacy harness.md v1.0 instead of active core.md v6.1), and missing
the LOB-composable pattern. This test suite confirms:

- base_harness.md is v3's core.md v6.1 (curated), not legacy v1.0
- lobs/commercial_auto.md and lobs/general_liability.md exist with v3 content
- V3_CORE_HARNESS loads from base_harness.md and contains v6.1 markers
- compose_harness_for_lobs composes core + active LOB correctly
- Cross-LOB isolation: a CA-only session never sees GL rules, and vice versa
- Legacy V3_HARNESS_FULL / V3_HARNESS_LIGHT still work (Step 25 matrix)
"""
from pathlib import Path

import pytest

from accord_ai.extraction.harness_content.v3_harness_snapshot import (
    V3_CA_LOB_HARNESS,
    V3_CORE_HARNESS,
    V3_GL_LOB_HARNESS,
    V3_HARNESS_FULL,
    V3_HARNESS_LIGHT,
    compose_harness_for_lobs,
)


HARNESS_DIR = Path(__file__).resolve().parent.parent / "accord_ai/harness"


def test_base_harness_is_core_v61_not_legacy():
    content = (HARNESS_DIR / "base_harness.md").read_text()
    # Curated v6.1 header — the marker we know differs from legacy v1.0
    assert "Core Principles v6.1" in content, (
        "base_harness.md should be v3's curated core.md v6.1, "
        "not the legacy harness.md v1.0"
    )
    # Sections characteristic of v6.1 curated content
    assert "Source Fidelity" in content
    assert "Entity & Attribute Routing" in content
    assert "Negation & Exclusion" in content
    assert "Schema Adherence" in content


def test_legacy_backup_preserved():
    legacy = HARNESS_DIR / "legacy_harness_v1.md"
    assert legacy.exists(), "legacy_harness_v1.md should be retained as backup"


def test_lob_harness_files_exist():
    assert (HARNESS_DIR / "lobs" / "commercial_auto.md").exists()
    assert (HARNESS_DIR / "lobs" / "general_liability.md").exists()


def test_commercial_auto_harness_has_v3_content():
    content = (HARNESS_DIR / "lobs" / "commercial_auto.md").read_text()
    assert "Commercial Auto" in content
    assert len(content) > 1000, "CA harness should be substantial (v3 ships v3.0)"


def test_general_liability_harness_has_v3_content():
    content = (HARNESS_DIR / "lobs" / "general_liability.md").read_text()
    assert "General Liability" in content
    assert len(content) > 500


def test_v3_core_harness_constant_populated():
    assert V3_CORE_HARNESS, "V3_CORE_HARNESS should be non-empty"
    assert "Core Principles v6.1" in V3_CORE_HARNESS


def test_v3_ca_lob_harness_constant_populated():
    assert V3_CA_LOB_HARNESS
    assert "Commercial Auto" in V3_CA_LOB_HARNESS


def test_v3_gl_lob_harness_constant_populated():
    assert V3_GL_LOB_HARNESS
    assert "General Liability" in V3_GL_LOB_HARNESS


def test_compose_harness_returns_core_only_when_no_lobs():
    result = compose_harness_for_lobs(None)
    assert "Core Principles v6.1" in result
    # CA-file distinctive section
    assert "Vehicle Count vs Employee Count" not in result
    # GL-file distinctive header (avoids false-positive on prose mentions)
    assert "General Liability Extraction Harness" not in result


def test_compose_harness_returns_core_only_when_empty_lob_list():
    result = compose_harness_for_lobs([])
    assert "Core Principles v6.1" in result
    assert "Vehicle Count vs Employee Count" not in result
    assert "General Liability Extraction Harness" not in result


def test_compose_harness_ca_includes_core_and_ca():
    result = compose_harness_for_lobs(["commercial_auto"])
    assert "Core Principles v6.1" in result
    assert "Vehicle Count vs Employee Count" in result
    # Cross-LOB isolation — GL file distinctive header should never appear
    # when only CA is active. (Prose mentions of "General Liability" can
    # legitimately appear in core.md or CA file as references.)
    assert "General Liability Extraction Harness" not in result


def test_compose_harness_gl_includes_core_and_gl():
    result = compose_harness_for_lobs(["general_liability"])
    assert "Core Principles v6.1" in result
    assert "General Liability Extraction Harness" in result
    assert "Vehicle Count vs Employee Count" not in result


def test_compose_harness_multi_lob_includes_all_requested():
    result = compose_harness_for_lobs(["commercial_auto", "general_liability"])
    assert "Core Principles v6.1" in result
    assert "Vehicle Count vs Employee Count" in result
    assert "General Liability Extraction Harness" in result


def test_compose_harness_unknown_lob_silently_skipped():
    """workers_comp has no LOB file yet; should return core only, no crash."""
    result = compose_harness_for_lobs(["workers_comp"])
    assert "Core Principles v6.1" in result
    assert "Vehicle Count vs Employee Count" not in result
    assert "General Liability Extraction Harness" not in result


def test_compose_harness_core_precedes_lob():
    """Core rules always first; LOB appends. Matters for prompt precedence."""
    result = compose_harness_for_lobs(["commercial_auto"])
    core_pos = result.find("Core Principles v6.1")
    ca_pos = result.find("Vehicle Count vs Employee Count")
    assert core_pos < ca_pos, "core.md content should precede LOB content"


def test_legacy_harness_constants_still_accessible():
    """Step 25 matrix comparisons rely on V3_HARNESS_FULL and V3_HARNESS_LIGHT."""
    assert V3_HARNESS_FULL.strip(), "V3_HARNESS_FULL still needed for matrix runs"
    assert V3_HARNESS_LIGHT.strip(), "V3_HARNESS_LIGHT still needed for matrix runs"


def test_experiment_harness_blocks_includes_core_modes():
    """The extractor's _HARNESS_BLOCKS dict has 'core', 'core_ca', 'core_gl' options."""
    from accord_ai.extraction.extractor import _HARNESS_BLOCKS

    assert "core" in _HARNESS_BLOCKS
    assert "core_ca" in _HARNESS_BLOCKS
    assert "core_gl" in _HARNESS_BLOCKS
    assert _HARNESS_BLOCKS["core"], "'core' block should be populated"
    # core_ca must contain CA-specific markers
    assert "Vehicle Count vs Employee Count" in _HARNESS_BLOCKS["core_ca"]
    # core_ca must NOT contain GL file header (cross-LOB isolation)
    assert "General Liability Extraction Harness" not in _HARNESS_BLOCKS["core_ca"]
    # core_gl must contain GL-specific markers
    assert "General Liability Extraction Harness" in _HARNESS_BLOCKS["core_gl"]
    assert "Vehicle Count vs Employee Count" not in _HARNESS_BLOCKS["core_gl"]
