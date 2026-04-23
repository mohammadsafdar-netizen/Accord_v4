"""Prompt families live one-file-per-family under this package.
Versioned constants (SYSTEM_V1, SYSTEM_V2, etc.) keep old variants
reachable for A/B and rollback. Rendering goes through `render`.
"""
from accord_ai.llm.prompts.render import render

__all__ = ["render"]
