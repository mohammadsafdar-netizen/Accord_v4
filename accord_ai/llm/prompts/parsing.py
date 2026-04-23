"""Shared helpers for parsing LLM output — v3's 7-strategy JSON repair.

LLMs emit structurally-broken JSON surprisingly often: markdown code fences,
thinking-mode preambles, trailing commas, truncation, embedded comments.
v3's parser cascades through 7 strategies before giving up. We port that here.

Strategy order (each runs on any candidate string; first success wins):
  1. Direct json.loads after fence/think strip
  2. Regex extraction of the outermost {...} block
  3. json_repair library (optional — pip install json_repair)
  4. ast.literal_eval (handles Python dict literals with single quotes)
  5. Trailing-comma removal
  6. Truncated JSON repair (balances braces for mid-generation cutoffs)
  7. Raise error_cls (v4 keeps this hard; v3 returned {} — v4 orchestrator owns retry)

Ported from v3's accord_ai/extraction/engine.py:141-220.
"""
from __future__ import annotations

import ast as _ast
import json as _json
import re
from typing import Any, Callable, Dict, Optional, Type

from pydantic import ValidationError as _ValidationError

from accord_ai.schema import CustomerSubmission as _CustomerSubmission


# Optional import — json_repair handles a wide range of malformed JSON.
# If not installed, strategy 3 is skipped and we fall through to 4.
try:
    from json_repair import repair_json as _repair_json
    _JSON_REPAIR_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent
    _repair_json = None
    _JSON_REPAIR_AVAILABLE = False


# --- Regex patterns --------------------------------------------------------

# Markdown code fence stripper:
#   ^\s*```(?:json)?\s*\n? ... \n?\s*```\s*$
_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$",
    re.DOTALL | re.IGNORECASE,
)

# Qwen3 thinking-mode block: <think>...</think>
# Matches greedily but non-overlapping. v3's engine.py strips these before parse.
_THINK_RE = re.compile(r"<think>[\s\S]*?</think>\s*", re.IGNORECASE)

# Finds the first {...} block. Used when fence-strip leaves non-JSON prose
# (FREE extraction mode, or thinking-mode leakage).
_FIRST_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

# Line-level comment stripper (//-style) — some LLMs insert JS-style comments.
_LINE_COMMENT_RE = re.compile(r"//.*?$", re.MULTILINE)

# Trailing comma before a closing brace/bracket.
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


# --- Public helpers --------------------------------------------------------


def strip_code_fences(text: str) -> str:
    """Remove surrounding ```...``` fences (with optional 'json' label).

    Idempotent — applying twice is the same as once.
    """
    m = _FENCE_RE.match(text)
    return m.group(1) if m else text


def strip_think_blocks(text: str) -> str:
    """Remove Qwen3-style <think>...</think> reasoning blocks.

    Qwen3 emits these even with chat_template_kwargs={"enable_thinking": False}
    in some configurations. Stripping them is cheap and safe.
    """
    return _THINK_RE.sub("", text)


def _try_parse(text: str) -> Optional[Dict[str, Any]]:
    """Single-pass parse with 7 strategies. Returns dict or None."""
    if not text or not text.strip():
        return None

    cleaned = text.strip()

    # Strategy 1: direct json.loads
    try:
        result = _json.loads(cleaned)
        if isinstance(result, dict):
            return result
        # Non-dict JSON — fall through
    except _json.JSONDecodeError:
        pass

    # Strategy 2: regex extraction of outermost {...}
    match = _FIRST_JSON_BLOCK_RE.search(cleaned)
    if match:
        candidate = _LINE_COMMENT_RE.sub("", match.group(0))

        # 2a: plain loads on the extracted block
        try:
            parsed = _json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except _json.JSONDecodeError:
            pass

        # Strategy 3: json_repair library
        if _JSON_REPAIR_AVAILABLE:
            try:
                repaired = _repair_json(candidate, return_objects=True)
                if isinstance(repaired, dict):
                    return repaired
            except Exception:  # pragma: no cover - defensive
                pass

        # Strategy 4: ast.literal_eval (handles single-quoted Python dicts)
        try:
            obj = _ast.literal_eval(candidate)
            if isinstance(obj, dict):
                return obj
        except (ValueError, SyntaxError, TypeError):
            pass

        # Strategy 5: trailing-comma removal
        no_trailing = _TRAILING_COMMA_RE.sub(r"\1", candidate)
        try:
            parsed = _json.loads(no_trailing)
            if isinstance(parsed, dict):
                return parsed
        except _json.JSONDecodeError:
            pass

    # Strategy 6: truncated JSON repair — balance braces + brackets
    if cleaned.lstrip().startswith("{"):
        balanced = _attempt_balance(cleaned)
        if balanced is not None:
            try:
                parsed = _json.loads(balanced)
                if isinstance(parsed, dict):
                    return parsed
            except _json.JSONDecodeError:
                pass
            # Also try json_repair on balanced version
            if _JSON_REPAIR_AVAILABLE:
                try:
                    repaired = _repair_json(balanced, return_objects=True)
                    if isinstance(repaired, dict):
                        return repaired
                except Exception:  # pragma: no cover
                    pass

    # Strategy 7: total failure — caller decides what to do
    return None


def _attempt_balance(text: str) -> Optional[str]:
    """Append the closing braces/brackets needed to balance an open JSON.

    Handles mid-generation truncation where the model cut off before
    emitting closing punctuation. Returns None if the imbalance looks
    too large to fix heuristically.
    """
    # Track brace/bracket depth, ignoring chars inside strings.
    opens_stack: list[str] = []
    in_string = False
    escaped = False
    pairs = {"}": "{", "]": "["}

    for ch in text:
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_string:
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ("{", "["):
            opens_stack.append(ch)
        elif ch in ("}", "]"):
            if opens_stack and opens_stack[-1] == pairs[ch]:
                opens_stack.pop()

    if not opens_stack:
        return None  # already balanced; different problem

    # Close in reverse order. If we're inside a string, we can't safely
    # repair — give up.
    if in_string:
        return None

    # Heuristic cap: don't try to close more than 10 unbalanced levels
    if len(opens_stack) > 10:
        return None

    suffix = "".join("}" if ch == "{" else "]" for ch in reversed(opens_stack))

    # Also trim any dangling trailing comma that would break json.loads
    trimmed = _TRAILING_COMMA_RE.sub(r"\1", text.rstrip())
    # If text ends mid-value (e.g. `"key":`), json.loads will still fail;
    # caller's json_repair fallback handles that case.
    return trimmed + suffix


# ---------------------------------------------------------------------------
# Shared: LLM text -> CustomerSubmission
# ---------------------------------------------------------------------------

# Type alias for an optional dict-level postprocess hook. Caller passes a
# function ``(delta_dict) -> delta_dict`` that runs after JSON-parse and
# before pydantic-validate.
_DeltaProcessor = Callable[[Dict[str, Any]], Dict[str, Any]]


def parse_submission_output(
    text: str,
    *,
    error_cls: Type[ValueError] = ValueError,
    postprocess: Optional[_DeltaProcessor] = None,
) -> _CustomerSubmission:
    """7-strategy parse → optional postprocess → pydantic validate.

    Strategies (each cascaded; first success wins):
      1. Direct json.loads
      2. Regex-extracted {...} + plain loads
      3. json_repair library (if installed)
      4. ast.literal_eval
      5. Trailing-comma removal + loads
      6. Truncated JSON repair (balance braces) + loads / json_repair
      7. Raise `error_cls` on total failure

    Raises `error_cls` (defaults to ValueError) with `from`-chained original
    exception on non-JSON output or schema-validation failure.
    """
    # Pre-clean: strip thinking blocks + fences (cheap, idempotent)
    cleaned = strip_think_blocks(text)
    cleaned = strip_code_fences(cleaned)

    data = _try_parse(cleaned)
    if data is None:
        raise error_cls("non-JSON output: all 7 parse strategies failed")

    if postprocess is not None:
        try:
            data = postprocess(data)
        except Exception as e:
            raise error_cls(f"postprocess failed: {e}") from e

    try:
        return _CustomerSubmission.model_validate(data)
    except _ValidationError as e:
        raise error_cls("schema validation failed") from e
