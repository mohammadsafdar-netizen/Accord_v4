"""Compact prompt-readable schema representation for v4's CustomerSubmission.

v3 hand-wrote a ~2k-token schema text and embedded it in the extraction system
prompt (prompts.py:_SCHEMA_SECTIONS). v4 historically dumped the full
Pydantic JSON schema (~26KB) into the USER message instead. That's a 13x token
difference and moves schema to the mutable turn slot, breaking prefix cache.

This module generates v3-style compact schema text from v4's Pydantic model,
so:
  * Schema is session-stable (prefix-cacheable)
  * Schema is small (fits with harness + state + user message under 16k)
  * Field names match v4's actual schema (generated, not hand-written)

Output format (v3-style):
    "business_name": str,
    "ein": "XX-XXXXXXX",
    "mailing_address": {line_one: str, line_two: str, ...},
    "lob_details" (discriminated: commercial_auto | general_liability | workers_comp): {
        "lob": "commercial_auto",
        "vehicles": [...],
        ...
    }

Ported conceptually from v3 accord_ai/extraction/prompts.py:_SCHEMA_SECTIONS,
adapted to v4's Pydantic model tree.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


# Human-readable type hints. Keys are Python/Pydantic type signals;
# values are concise strings inserted into the schema text.
_TYPE_HINTS = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "null": "null",
}


def _is_dict_schema(s: Dict[str, Any]) -> bool:
    return s.get("type") == "object" or "properties" in s


def _is_array_schema(s: Dict[str, Any]) -> bool:
    return s.get("type") == "array" or "items" in s


def _type_hint(prop_schema: Dict[str, Any]) -> str:
    """Produce a compact type hint for a Pydantic JSON schema fragment.

    Handles: simple types, anyOf (union/optional), enum literals, refs.
    Returns a string like "str", "int", "'option_a' | 'option_b'".
    """
    # Enums (Literal types): "enum": ["a", "b", "c"]
    if "enum" in prop_schema:
        options = prop_schema["enum"]
        quoted = [f'"{o}"' if isinstance(o, str) else str(o) for o in options]
        return " | ".join(quoted)

    # anyOf (Union / Optional)
    if "anyOf" in prop_schema:
        parts = []
        for opt in prop_schema["anyOf"]:
            if opt.get("type") == "null":
                continue  # Optional[T] → treat as just T
            parts.append(_type_hint(opt))
        if len(parts) == 1:
            return parts[0]
        return " | ".join(parts)

    # $ref — we expand these inline elsewhere; here just return a placeholder
    if "$ref" in prop_schema:
        return prop_schema["$ref"].split("/")[-1]

    # Simple type
    typ = prop_schema.get("type")
    if isinstance(typ, str):
        return _TYPE_HINTS.get(typ, typ)

    # Arrays
    if _is_array_schema(prop_schema):
        items = prop_schema.get("items", {})
        return f"[{_type_hint(items)}]"

    # Nested object — fall through
    return "obj"


def _resolve_ref(ref: str, defs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Resolve a $ref like "#/$defs/Address" against a $defs dict."""
    if not ref.startswith("#/$defs/"):
        return None
    name = ref.split("/")[-1]
    return defs.get(name)


def _render_object(
    schema: Dict[str, Any],
    defs: Dict[str, Any],
    indent: int = 2,
    max_depth: int = 4,
    current_depth: int = 0,
) -> str:
    """Render a JSON-schema object as compact prompt-readable text."""
    if current_depth >= max_depth:
        return "{...}"

    props = schema.get("properties", {})
    if not props:
        return "{...}"

    lines: List[str] = []
    pad = " " * (indent * (current_depth + 1))
    close_pad = " " * (indent * current_depth)

    for name, prop in props.items():
        rendered = _render_property(prop, defs, indent, max_depth, current_depth + 1)
        lines.append(f'{pad}"{name}": {rendered}')

    body = ",\n".join(lines)
    return "{\n" + body + "\n" + close_pad + "}"


def _render_property(
    prop: Dict[str, Any],
    defs: Dict[str, Any],
    indent: int,
    max_depth: int,
    current_depth: int,
) -> str:
    """Render a single property's value representation."""
    # $ref — look up and render inline
    if "$ref" in prop:
        target = _resolve_ref(prop["$ref"], defs)
        if target is not None and _is_dict_schema(target):
            return _render_object(target, defs, indent, max_depth, current_depth)
        if target is not None and "enum" in target:
            return _type_hint(target)
        return prop["$ref"].split("/")[-1]  # fallback: just the name

    # Arrays
    if _is_array_schema(prop):
        items = prop.get("items", {})
        if "$ref" in items:
            target = _resolve_ref(items["$ref"], defs)
            if target is not None and _is_dict_schema(target):
                item_body = _render_object(target, defs, indent, max_depth, current_depth)
                return f"[{item_body}]"
        return f"[{_type_hint(items)}]"

    # Nested object (inline properties)
    if _is_dict_schema(prop):
        return _render_object(prop, defs, indent, max_depth, current_depth)

    # anyOf / oneOf — handle discriminated unions, Optional, plain unions
    if "anyOf" in prop or "oneOf" in prop:
        variants = prop.get("anyOf", []) + prop.get("oneOf", [])

        # Discriminated union: nested oneOf inside anyOf (Pydantic pattern for
        # Union[A, B, C] with discriminator field)
        for variant in variants:
            if "oneOf" in variant:
                sub_variants = variant["oneOf"]
                rendered_variants = []
                for sv in sub_variants:
                    if "$ref" in sv:
                        target = _resolve_ref(sv["$ref"], defs)
                        if target is not None and _is_dict_schema(target):
                            rendered_variants.append(
                                _render_object(target, defs, indent, max_depth, current_depth)
                            )
                if rendered_variants:
                    nulls = any(v.get("type") == "null" for v in variants)
                    body = "\n" + (" " * (indent * current_depth)) + "| ".join(rendered_variants)
                    if nulls:
                        body += " | null"
                    return body

        # Plain anyOf with refs (no discriminator)
        refs = [opt for opt in variants if "$ref" in opt]
        nulls = [opt for opt in variants if opt.get("type") == "null"]
        if refs:
            target = _resolve_ref(refs[0]["$ref"], defs)
            if target is not None and _is_dict_schema(target):
                body = _render_object(target, defs, indent, max_depth, current_depth)
                if nulls:
                    return f"{body} | null"
                return body

    # Simple type hint
    return _type_hint(prop)


def build_schema_text(
    model_class: Any,
    *,
    exclude: Optional[Tuple[str, ...]] = None,
    max_depth: int = 4,
) -> str:
    """Build a compact prompt-readable schema text from a Pydantic model.

    Args:
        model_class: A Pydantic model class (e.g. CustomerSubmission).
        exclude: Top-level property names to omit (e.g. "conflicts" is
                 system-generated and not extracted by the LLM).
        max_depth: Maximum nesting depth before collapsing to "{...}".
                   v3 uses ~4 levels of nesting; lower values produce
                   smaller output at the cost of some field visibility.

    Returns: A string like `{"business_name": str, ...}`.
    """
    schema = model_class.model_json_schema()
    defs = schema.get("$defs", {})
    exclude_set = set(exclude or ())

    props = schema.get("properties", {})
    lines: List[str] = []
    pad = "  "

    for name, prop in props.items():
        if name in exclude_set:
            continue
        rendered = _render_property(prop, defs, indent=2, max_depth=max_depth, current_depth=1)
        lines.append(f'{pad}"{name}": {rendered}')

    return "{\n" + ",\n".join(lines) + "\n}"


def estimate_schema_tokens(text: str) -> int:
    """Rough token-count estimate (chars / 3.5 for typical JSON-ish text)."""
    return len(text) // 3 + 50
