"""LOB plugin registry — protocol + registration + dispatch.

Each LOB plugin module calls register(MyPlugin()) at import time.
The lobs/__init__.py ensures all builtins are imported on package import.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Protocol, Tuple, runtime_checkable


@runtime_checkable
class LOBPlugin(Protocol):
    """Structural contract every LOB plugin must satisfy.

    Attributes:
        lob_key: Discriminator string matching schema's lob field
                 (e.g. "commercial_auto").
        critical_fields: Merged (common + LOB-specific) list of
                         (v4_schema_path, human_reason) pairs.
    """
    lob_key: str
    critical_fields: List[Tuple[str, str]]


_REGISTRY: Dict[str, LOBPlugin] = {}


def register(plugin: LOBPlugin) -> None:
    """Register a LOB plugin. Raises TypeError if protocol unsatisfied,
    ValueError if the lob_key is already registered."""
    if not isinstance(plugin, LOBPlugin):
        raise TypeError(
            f"{type(plugin).__name__} does not implement LOBPlugin "
            f"(must have lob_key and critical_fields attributes)"
        )
    if plugin.lob_key in _REGISTRY:
        raise ValueError(f"LOB {plugin.lob_key!r} is already registered")
    _REGISTRY[plugin.lob_key] = plugin


def get_plugin(lob: str) -> Optional[LOBPlugin]:
    """Return the registered plugin for `lob`, or None if unknown."""
    return _REGISTRY.get(lob)


def get_critical_fields(lob: str) -> List[Tuple[str, str]]:
    """Return the merged critical-field list for `lob`.

    Unknown LOB falls back to COMMON_CRITICAL so the judge still gates
    on the LOB-agnostic baseline for future / unrecognised LOBs.
    """
    plugin = _REGISTRY.get(lob)
    if plugin is None:
        from accord_ai.lobs.common import COMMON_CRITICAL
        return list(COMMON_CRITICAL)
    return list(plugin.critical_fields)


def list_registered() -> List[str]:
    """Return sorted list of all registered LOB keys."""
    return sorted(_REGISTRY.keys())
