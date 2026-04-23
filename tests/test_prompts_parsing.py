"""P6.0 — strip_code_fences helper (accord_ai/llm/prompts/parsing.py)."""
from accord_ai.llm.prompts.parsing import strip_code_fences


def test_plain_text_untouched():
    assert strip_code_fences('{"x": 1}') == '{"x": 1}'


def test_json_fenced_multiline_unwrapped():
    text = '```json\n{"x": 1}\n```'
    assert strip_code_fences(text) == '{"x": 1}'


def test_bare_fenced_unwrapped():
    text = '```\n{"x": 1}\n```'
    assert strip_code_fences(text) == '{"x": 1}'


def test_single_line_fenced_unwrapped():
    """Token-tight output: LLM emits ```json{"x":1}``` with no newlines."""
    assert strip_code_fences('```json{"x":1}```') == '{"x":1}'


def test_leading_and_trailing_whitespace_around_fences_unwrapped():
    text = '  \n```json\n{"x": 1}\n```  \n'
    assert strip_code_fences(text) == '{"x": 1}'


def test_unclosed_fence_left_alone():
    """Fail-closed: malformed input isn't guessed at."""
    text = '```json\n{"x": 1}'
    assert strip_code_fences(text) == text


def test_empty_string_returned_as_is():
    assert strip_code_fences("") == ""
