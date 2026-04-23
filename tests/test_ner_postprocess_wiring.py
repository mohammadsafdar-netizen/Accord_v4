"""Verify the ner_postprocess flag wires validate_extraction_with_ner into
the extraction postprocess chain.

Background: v4's accord_ai/extraction/ner.py has validate_extraction_with_ner
already adapted to v4's schema (business_name at root, contacts[0].full_name),
with the learned safety rule ("single PERSON only" gate on Fix 2). Phase A
disabled it in production after a regression on multi-five-vehicle-fleet.

Research 2026-04-22 identified NER post-extraction validation as a
load-bearing v3 mechanism. This re-enables it behind a flag so the full v3-
port stack can measure its contribution (the single-PERSON gate now
prevents the Phase A regression).
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from accord_ai.extraction.extractor import Extractor
from accord_ai.schema import CustomerSubmission


@pytest.fixture
def mock_engine():
    eng = MagicMock()
    eng.generate = AsyncMock(return_value=MagicMock(text='{"business_name": "Acme Inc"}'))
    return eng


def _build(mock_engine, **kwargs):
    defaults = {
        "engine": mock_engine,
        "memory": None,
        "memory_enabled": False,
        "experiment_harness": "none",
        "extraction_mode": "xgrammar",
        "harness_position": "before",
        "ner_postprocess": False,
    }
    defaults.update(kwargs)
    return Extractor(**defaults)


@pytest.mark.asyncio
async def test_ner_postprocess_off_does_not_invoke_validator(mock_engine):
    """Default: NER validator is NOT called during extraction."""
    ext = _build(mock_engine, ner_postprocess=False)
    with patch(
        "accord_ai.extraction.ner.validate_extraction_with_ner"
    ) as mock_validate, patch(
        "accord_ai.extraction.ner.tag_entities"
    ) as mock_tag:
        await ext.extract(
            current_submission=CustomerSubmission(),
            user_message="hello",
        )
    assert not mock_validate.called
    assert not mock_tag.called


@pytest.mark.asyncio
async def test_ner_postprocess_on_invokes_validator(mock_engine):
    """When flag on, validator receives delta + ner_tags + current_state."""
    ext = _build(mock_engine, ner_postprocess=True)
    with patch(
        "accord_ai.extraction.ner.validate_extraction_with_ner",
        return_value={"business_name": "Acme Inc"},
    ) as mock_validate, patch(
        "accord_ai.extraction.ner.tag_entities",
        return_value={"persons": [], "orgs": ["Acme Inc"], "websites": []},
    ) as mock_tag:
        await ext.extract(
            current_submission=CustomerSubmission(),
            user_message="Our business is Acme Inc.",
        )
    mock_tag.assert_called_once_with("Our business is Acme Inc.")
    assert mock_validate.called
    # Validator receives delta (dict), ner_tags (dict), current_state kwarg
    call = mock_validate.call_args
    assert isinstance(call.args[0], dict)  # delta
    assert isinstance(call.args[1], dict)  # ner_tags
    assert "current_state" in call.kwargs


@pytest.mark.asyncio
async def test_ner_postprocess_failure_silently_swallowed(mock_engine):
    """If NER (e.g. spaCy model missing) raises, extraction still completes."""
    ext = _build(mock_engine, ner_postprocess=True)
    with patch(
        "accord_ai.extraction.ner.tag_entities",
        side_effect=RuntimeError("spacy model missing"),
    ):
        # Extraction should complete without raising
        result = await ext.extract(
            current_submission=CustomerSubmission(),
            user_message="hello",
        )
    # Result should still be the parsed delta (as CustomerSubmission)
    assert isinstance(result, CustomerSubmission)


def test_config_ner_postprocess_default_off():
    """Default off until eval confirms productive composition with harness stack."""
    from accord_ai.config import Settings

    s = Settings()
    assert s.ner_postprocess is False


def test_config_ner_postprocess_accepts_true():
    from accord_ai.config import Settings

    s = Settings(ner_postprocess=True)
    assert s.ner_postprocess is True
