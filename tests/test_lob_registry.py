"""Tests for the LOB plugin registry (Phase 1.7)."""
from __future__ import annotations

import pytest
import accord_ai.lobs.registry as _reg


# ---------------------------------------------------------------------------
# Fixture: clean up any lob_keys added during a test
# ---------------------------------------------------------------------------

@pytest.fixture()
def clean_registry():
    """Remove any test-registered keys from the module-level _REGISTRY."""
    keys_before = frozenset(_reg._REGISTRY.keys())
    yield
    for k in list(_reg._REGISTRY.keys()):
        if k not in keys_before:
            del _reg._REGISTRY[k]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_plugin(lob_key: str, fields=None):
    """Return a minimal LOBPlugin-conformant object."""
    from dataclasses import dataclass

    cf = fields or [("business_name", "required")]

    @dataclass(frozen=True)
    class FakePlugin:
        lob_key: str

        @property
        def critical_fields(self):
            return list(cf)

    return FakePlugin(lob_key=lob_key)


# ---------------------------------------------------------------------------
# 1. register a fake plugin
# ---------------------------------------------------------------------------

def test_register_plugin(clean_registry):
    plugin = _make_plugin("test_lob_xyz")
    _reg.register(plugin)
    assert "test_lob_xyz" in _reg.list_registered()


# ---------------------------------------------------------------------------
# 2. duplicate registration raises ValueError
# ---------------------------------------------------------------------------

def test_register_duplicate_raises():
    # "commercial_auto" is already registered by the package import
    with pytest.raises(ValueError, match="already registered"):
        _reg.register(_make_plugin("commercial_auto"))


# ---------------------------------------------------------------------------
# 3. non-protocol object raises TypeError
# ---------------------------------------------------------------------------

def test_register_non_plugin_raises():
    class Incomplete:
        pass  # no lob_key, no critical_fields

    with pytest.raises(TypeError):
        _reg.register(Incomplete())


# ---------------------------------------------------------------------------
# 4. get_critical_fields for a known LOB
# ---------------------------------------------------------------------------

def test_get_critical_fields_known_lob():
    from accord_ai.lobs.common import COMMON_CRITICAL
    fields = _reg.get_critical_fields("commercial_auto")
    assert isinstance(fields, list)
    assert len(fields) > len(COMMON_CRITICAL)
    # All common fields must be present
    for cf in COMMON_CRITICAL:
        assert cf in fields


# ---------------------------------------------------------------------------
# 5. unknown LOB falls back to COMMON_CRITICAL
# ---------------------------------------------------------------------------

def test_get_critical_fields_unknown_lob_falls_back_to_common():
    from accord_ai.lobs.common import COMMON_CRITICAL
    fields = _reg.get_critical_fields("nonexistent_lob_zzz")
    assert fields == list(COMMON_CRITICAL)


# ---------------------------------------------------------------------------
# 6. get_critical_fields returns a copy — caller mutations don't persist
# ---------------------------------------------------------------------------

def test_get_critical_fields_returns_list_copy():
    fields1 = _reg.get_critical_fields("commercial_auto")
    fields1.append(("injected", "should not persist"))
    fields2 = _reg.get_critical_fields("commercial_auto")
    assert ("injected", "should not persist") not in fields2


# ---------------------------------------------------------------------------
# 7. all three built-in LOBs are registered
# ---------------------------------------------------------------------------

def test_three_builtin_lobs_registered():
    registered = _reg.list_registered()
    assert "commercial_auto" in registered
    assert "general_liability" in registered
    assert "workers_comp" in registered
