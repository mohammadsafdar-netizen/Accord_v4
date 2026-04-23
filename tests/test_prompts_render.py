"""7.b — strict-kwargs render helper."""
import pytest

from accord_ai.llm.prompts import render


def test_render_substitutes_matching_kwargs():
    assert render("Hello {name}", name="Alice") == "Hello Alice"


def test_render_multiple_placeholders():
    out = render("{greeting} {name}, age {age}", greeting="Hi", name="Alice", age="30")
    assert out == "Hi Alice, age 30"


def test_render_raises_when_kwarg_missing():
    with pytest.raises(ValueError) as exc_info:
        render("Hello {name}")
    assert "missing" in str(exc_info.value)
    assert "name" in str(exc_info.value)


def test_render_raises_when_kwarg_unexpected():
    with pytest.raises(ValueError) as exc_info:
        render("plain text", extra="nope")
    assert "unexpected" in str(exc_info.value)
    assert "extra" in str(exc_info.value)


def test_render_mentions_both_missing_and_unexpected():
    with pytest.raises(ValueError) as exc_info:
        render("Hello {name}", greeting="Hi")
    msg = str(exc_info.value)
    assert "missing" in msg and "name" in msg
    assert "unexpected" in msg and "greeting" in msg


def test_render_empty_template():
    assert render("") == ""


def test_render_no_placeholders_no_kwargs():
    assert render("plain text with no braces") == "plain text with no braces"


def test_render_duplicate_placeholder_counted_once():
    """{name} appearing twice is still one placeholder kwarg."""
    assert render("{name} and {name}", name="Alice") == "Alice and Alice"


def test_render_template_is_positional_only():
    """A placeholder named 'template' is legal — template arg is positional-only."""
    out = render("{template}", template="value")
    assert out == "value"
