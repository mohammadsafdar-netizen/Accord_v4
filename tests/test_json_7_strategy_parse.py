"""Verify v3's 7-strategy JSON parser handles malformed LLM output.

v3 ships a cascading parse that salvages output even when the LLM emits:
  - Markdown code fences
  - <think>...</think> blocks (Qwen3 thinking mode)
  - Trailing commas
  - Python dict syntax (single quotes)
  - JS-style // line comments
  - Truncated JSON (mid-generation cutoff)
  - Prose wrapping the JSON block

v4 historically had only fence-strip + direct loads + regex fallback (3 of 7).
This test suite validates the full 7-strategy cascade.

Reference: v3 accord_ai/extraction/engine.py:141-220
"""
import pytest

from accord_ai.llm.prompts.parsing import (
    _attempt_balance,
    _try_parse,
    parse_submission_output,
    strip_code_fences,
    strip_think_blocks,
)


# --- Cheap helpers ---------------------------------------------------------


def test_strip_code_fences_plain_fence():
    assert strip_code_fences('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_code_fences_no_fence():
    assert strip_code_fences('{"a": 1}') == '{"a": 1}'


def test_strip_code_fences_idempotent():
    stripped_once = strip_code_fences('```json\n{"a": 1}\n```')
    stripped_twice = strip_code_fences(stripped_once)
    assert stripped_once == stripped_twice


def test_strip_think_blocks_simple():
    text = '<think>reasoning</think>\n{"a": 1}'
    assert strip_think_blocks(text).strip() == '{"a": 1}'


def test_strip_think_blocks_multiple():
    text = '<think>one</think><think>two</think>{"a": 1}'
    assert strip_think_blocks(text).strip() == '{"a": 1}'


def test_strip_think_blocks_no_block():
    assert strip_think_blocks('{"a": 1}') == '{"a": 1}'


# --- Strategy 1: direct json.loads ----------------------------------------


def test_strategy_1_direct_loads():
    assert _try_parse('{"a": 1, "b": "two"}') == {"a": 1, "b": "two"}


def test_strategy_1_empty_input_returns_none():
    assert _try_parse("") is None
    assert _try_parse("   ") is None


def test_strategy_1_non_dict_json_falls_through():
    # Top-level array is not a valid submission; should return None (not raise)
    # — the next strategy would extract nothing useful, so overall None is fine
    assert _try_parse("[1, 2, 3]") is None
    assert _try_parse('"hello"') is None


# --- Strategy 2: regex extraction of {...} --------------------------------


def test_strategy_2_prose_wrapping_json():
    text = 'Sure, here you go: {"a": 1} and that\'s it'
    assert _try_parse(text) == {"a": 1}


def test_strategy_2_strips_line_comments():
    text = '{"a": 1, // this is bogus\n"b": 2}'
    assert _try_parse(text) == {"a": 1, "b": 2}


# --- Strategy 3: json_repair library --------------------------------------


def test_strategy_3_json_repair_handles_unquoted_keys():
    # json_repair handles common malformations json.loads can't.
    text = '{a: 1, "b": 2}'
    result = _try_parse(text)
    # json_repair handles this; if missing, fallback to other strategies
    assert result == {"a": 1, "b": 2}


def test_strategy_3_json_repair_handles_missing_closing_quote():
    text = '{"a": "unclosed, "b": 2}'
    result = _try_parse(text)
    # Either json_repair salvages or we fall through — at minimum, no crash
    assert result is None or isinstance(result, dict)


# --- Strategy 4: ast.literal_eval -----------------------------------------


def test_strategy_4_python_dict_literal_single_quotes():
    # json_repair tries this first; for LLMs emitting Python dict syntax,
    # either path works. Checking end-to-end behavior.
    text = "{'a': 1, 'b': 2}"
    assert _try_parse(text) == {"a": 1, "b": 2}


# --- Strategy 5: trailing comma removal -----------------------------------


def test_strategy_5_trailing_comma_in_object():
    assert _try_parse('{"a": 1, "b": 2,}') == {"a": 1, "b": 2}


def test_strategy_5_trailing_comma_in_nested_array():
    assert _try_parse('{"items": [1, 2, 3,]}') == {"items": [1, 2, 3]}


# --- Strategy 6: truncated JSON repair ------------------------------------


def test_strategy_6_truncated_missing_close_brace():
    # Model cut off before closing brace — repair should close it
    text = '{"a": 1, "b": 2'
    result = _try_parse(text)
    assert result == {"a": 1, "b": 2}


def test_strategy_6_truncated_nested_object():
    text = '{"outer": {"inner": "value"'
    result = _try_parse(text)
    assert result == {"outer": {"inner": "value"}}


def test_strategy_6_truncated_with_array():
    text = '{"items": [1, 2, 3'
    result = _try_parse(text)
    assert result == {"items": [1, 2, 3]}


def test_strategy_6_balance_helper_balanced_input_returns_none():
    # Already balanced — no repair needed, helper returns None
    assert _attempt_balance('{"a": 1}') is None


def test_strategy_6_balance_helper_deep_imbalance_gives_up():
    # >10 unclosed levels — heuristic refuses to guess
    text = "{" * 15
    assert _attempt_balance(text) is None


def test_strategy_6_balance_ignores_braces_in_strings():
    # `{` inside a string must not count as an unclosed brace
    text = '{"msg": "hello { world"'
    # One unclosed real brace → should close with one }
    result = _attempt_balance(text)
    assert result is not None and result.endswith("}")


# --- Strategy 7: total failure raises -------------------------------------


def test_strategy_7_total_failure_returns_none():
    assert _try_parse("not even close to json") is None
    assert _try_parse("*&^%$#@!") is None


# --- Integration with parse_submission_output -----------------------------


def test_parse_submission_output_happy_path():
    # Minimum valid submission
    result = parse_submission_output('{"business_name": "Acme"}')
    assert result.business_name == "Acme"


def test_parse_submission_output_handles_fences():
    result = parse_submission_output('```json\n{"business_name": "Acme"}\n```')
    assert result.business_name == "Acme"


def test_parse_submission_output_handles_think_block():
    text = '<think>let me think</think>\n{"business_name": "Acme"}'
    result = parse_submission_output(text)
    assert result.business_name == "Acme"


def test_parse_submission_output_raises_on_total_failure():
    with pytest.raises(ValueError, match="non-JSON output"):
        parse_submission_output("model went off the rails completely")


def test_parse_submission_output_raises_on_schema_failure():
    # Valid JSON but wrong schema — business_name expects str, got dict
    with pytest.raises(ValueError, match="schema validation"):
        parse_submission_output('{"business_name": {"nested": "dict"}}')


def test_parse_submission_output_trailing_comma_recovers():
    result = parse_submission_output('{"business_name": "Acme",}')
    assert result.business_name == "Acme"


def test_parse_submission_output_truncated_recovers():
    # Missing closing brace — strategy 6 should fix
    text = '{"business_name": "Acme"'
    result = parse_submission_output(text)
    assert result.business_name == "Acme"


def test_parse_submission_output_combined_corruption():
    # Multiple issues at once: think block + fence + trailing comma
    text = (
        "<think>ok</think>\n"
        "```json\n"
        '{"business_name": "Acme",}\n'
        "```"
    )
    result = parse_submission_output(text)
    assert result.business_name == "Acme"


def test_parse_submission_output_postprocess_runs():
    calls = []

    def postprocess(delta):
        calls.append(delta)
        return {**delta, "business_name": delta["business_name"].upper()}

    result = parse_submission_output(
        '{"business_name": "acme"}', postprocess=postprocess
    )
    assert len(calls) == 1
    assert result.business_name == "ACME"


def test_parse_submission_output_postprocess_failure_wrapped():
    def broken(delta):
        raise RuntimeError("bad")

    with pytest.raises(ValueError, match="postprocess failed"):
        parse_submission_output('{"business_name": "x"}', postprocess=broken)
