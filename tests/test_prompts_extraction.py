"""7.b — extraction prompt family v1."""
from string import Formatter

from accord_ai.llm.prompts import extraction, render


def test_system_v1_is_nonempty_string():
    assert isinstance(extraction.SYSTEM_V1, str)
    assert len(extraction.SYSTEM_V1) > 50


def test_user_template_v1_declares_expected_placeholders():
    placeholders = {
        name for _, name, _, _ in Formatter().parse(extraction.USER_TEMPLATE_V1)
        if name
    }
    assert placeholders == {"schema", "current_state", "user_message"}


def test_user_template_v1_renders_with_correct_kwargs():
    out = render(
        extraction.USER_TEMPLATE_V1,
        schema="{...schema...}",
        current_state="{}",
        user_message="Our business is Acme Trucking.",
    )
    assert "Acme Trucking" in out
    assert "{...schema...}" in out
