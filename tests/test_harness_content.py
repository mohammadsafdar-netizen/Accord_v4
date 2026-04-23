"""V.3 — Verify base_harness.md and brokers/ directory exist.

These tests guard the file structure needed for the living harness system.
They will FAIL until base_harness.md is ported from v3 and brokers/ is created.
"""
from __future__ import annotations

import pathlib

HARNESS_DIR = pathlib.Path(__file__).parent.parent / "accord_ai" / "harness"


def test_base_harness_md_exists():
    """base_harness.md is the v3 harness ported verbatim — must exist and be non-empty."""
    p = HARNESS_DIR / "base_harness.md"
    assert p.exists(), f"Missing: {p}"
    content = p.read_text()
    assert len(content) > 500, "base_harness.md appears truncated or empty"


def test_brokers_directory_exists():
    """brokers/ is the per-broker overlay directory — must exist."""
    p = HARNESS_DIR / "brokers"
    assert p.is_dir(), f"Missing directory: {p}"
