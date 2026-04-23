"""9.b — Extractor tests. FakeEngine-driven, no network."""
import pytest

from accord_ai.core.diff import apply_diff
from accord_ai.extraction.extractor import (
    Extractor,
    ExtractionOutputError,
    _adaptive_max_tokens,
    _build_corrections_block,
)
from accord_ai.feedback.memory import CorrectionMemory, CorrectionMemoryEntry
from accord_ai.llm.fake_engine import FakeEngine
from accord_ai.schema import CustomerSubmission
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Adaptive max_tokens — Phase A step 1 (ported from v3 runner.py:368-372)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("msg_len,expected", [
    # Short branch (<30): 512 tokens
    (0,   512),
    (1,   512),
    (29,  512),
    # Default branch (30-1500 inclusive): 2048 tokens
    (30,   2048),
    (100,  2048),
    (500,  2048),
    (1500, 2048),
    # Long branch (>1500): 4096 tokens
    (1501, 4096),
    (3000, 4096),
    (10000, 4096),
])
def test_adaptive_max_tokens_boundaries(msg_len, expected):
    msg = "x" * msg_len
    assert _adaptive_max_tokens(msg) == expected


def test_adaptive_max_tokens_short_empty_string():
    assert _adaptive_max_tokens("") == 512


def test_adaptive_max_tokens_at_short_boundary_stays_short():
    """Exactly 29 chars → short; 30 chars → default."""
    assert _adaptive_max_tokens("x" * 29) == 512
    assert _adaptive_max_tokens("x" * 30) == 2048


def test_adaptive_max_tokens_at_long_boundary_stays_default():
    """Exactly 1500 chars → default; 1501 chars → long."""
    assert _adaptive_max_tokens("x" * 1500) == 2048
    assert _adaptive_max_tokens("x" * 1501) == 4096


@pytest.mark.asyncio
async def test_extractor_sends_adaptive_max_tokens():
    """Extractor passes the adapted budget (not a fixed value) through the
    engine call. Verifies the wiring end-to-end."""
    engine = FakeEngine([{"business_name": "Acme"}])
    extractor = Extractor(engine)
    # Long message → expect 4096
    long_msg = "x" * 2000
    await extractor.extract(
        user_message=long_msg,
        current_submission=CustomerSubmission(),
    )
    assert engine.last_call.max_tokens == 4096


@pytest.mark.asyncio
async def test_extractor_short_message_gets_512_budget():
    engine = FakeEngine([{"business_name": "Acme"}])
    extractor = Extractor(engine)
    await extractor.extract(
        user_message="hi",    # 2 chars
        current_submission=CustomerSubmission(),
    )
    assert engine.last_call.max_tokens == 512


@pytest.mark.asyncio
async def test_extract_sends_guided_json_customer_submission_schema():
    """Structured-output guard: the extractor must pass the
    CustomerSubmission schema as json_schema so vLLM's xgrammar constrains
    output to valid JSON. Without this, malformed output drops through to
    ExtractionOutputError → 502."""
    engine = FakeEngine([{"business_name": "Acme"}])
    extractor = Extractor(engine)
    await extractor.extract(
        user_message="hi",
        current_submission=CustomerSubmission(),
    )
    schema = engine.last_call.json_schema
    assert schema is not None
    # Sanity-check it's actually the CustomerSubmission schema:
    # the top-level `properties` must include the root fields.
    assert "business_name" in schema.get("properties", {})
    assert "lob_details" in schema.get("properties", {})


# --- Error type ---

def test_extraction_output_error_is_valueerror():
    assert issubclass(ExtractionOutputError, ValueError)


# --- Happy path ---

@pytest.mark.asyncio
async def test_extract_returns_parsed_customer_submission():
    engine = FakeEngine([{"business_name": "Acme"}])
    extractor = Extractor(engine)

    diff = await extractor.extract(
        user_message="We are Acme.",
        current_submission=CustomerSubmission(),
    )
    assert isinstance(diff, CustomerSubmission)
    assert diff.business_name == "Acme"


@pytest.mark.asyncio
async def test_extract_result_has_diff_semantics_via_model_fields_set():
    """Only LLM-set fields are in model_fields_set — makes apply_diff right."""
    engine = FakeEngine([{"business_name": "Acme", "ein": "12-3456789"}])
    extractor = Extractor(engine)

    diff = await extractor.extract(
        user_message="Acme, EIN 12-3456789",
        current_submission=CustomerSubmission(),
    )
    assert "business_name" in diff.model_fields_set
    assert "ein" in diff.model_fields_set
    # Fields the LLM didn't mention stay out of model_fields_set
    assert "phone" not in diff.model_fields_set


# --- End-to-end via apply_diff ---

@pytest.mark.asyncio
async def test_extracted_diff_merges_cleanly_via_apply_diff():
    """Full turn shape: extract → apply_diff → merged state."""
    current = CustomerSubmission(business_name="Acme")
    engine = FakeEngine([{"ein": "12-3456789", "email": "ops@acme.com"}])
    extractor = Extractor(engine)

    diff = await extractor.extract(
        user_message="EIN 12-3456789, email ops@acme.com",
        current_submission=current,
    )
    merged = apply_diff(current, diff)
    assert merged.business_name == "Acme"      # preserved — not in diff
    assert merged.ein == "12-3456789"
    assert merged.email == "ops@acme.com"


# --- Markdown fences handled ---

@pytest.mark.asyncio
async def test_extract_strips_markdown_code_fences():
    engine = FakeEngine(['```json\n{"business_name": "Acme"}\n```'])
    extractor = Extractor(engine)
    diff = await extractor.extract(
        user_message="Acme",
        current_submission=CustomerSubmission(),
    )
    assert diff.business_name == "Acme"


# --- Error paths ---

@pytest.mark.asyncio
async def test_extract_first_json_block_fallback():
    """FREE mode: prose before/after JSON is tolerated via first-block fallback."""
    engine = FakeEngine(['Here is the data: {"business_name": "Acme"} Hope that helps!'])
    extractor = Extractor(engine)
    diff = await extractor.extract(
        user_message="We are Acme.",
        current_submission=CustomerSubmission(),
    )
    assert diff.business_name == "Acme"


@pytest.mark.asyncio
async def test_extract_raises_on_non_json_output():
    engine = FakeEngine(["this is not JSON"])
    extractor = Extractor(engine)
    with pytest.raises(ExtractionOutputError, match="non-JSON"):
        await extractor.extract(
            user_message="msg",
            current_submission=CustomerSubmission(),
        )


@pytest.mark.asyncio
async def test_extract_raises_on_schema_invalid_json():
    """Schema is extra='ignore' so unknown keys don't trigger validation
    failure — use a primitive-type mismatch that pydantic can't coerce.
    lob_details is a discriminated-union object; passing a bare string
    violates the union shape and is the exact failure mode seen live."""
    engine = FakeEngine([{"lob_details": "commercial_auto"}])
    extractor = Extractor(engine)
    with pytest.raises(ExtractionOutputError, match="schema validation"):
        await extractor.extract(
            user_message="msg",
            current_submission=CustomerSubmission(),
        )


@pytest.mark.asyncio
async def test_extract_propagates_engine_exceptions_unwrapped():
    class _BoomEngine:
        async def generate(self, messages, *, temperature=0.0, max_tokens=4096, json_schema=None):
            raise RuntimeError("provider down")

    with pytest.raises(RuntimeError, match="provider down"):
        await Extractor(_BoomEngine()).extract(
            user_message="msg",
            current_submission=CustomerSubmission(),
        )


# --- Prompt composition ---

@pytest.mark.asyncio
async def test_extract_sends_system_then_user_messages():
    engine = FakeEngine([{"business_name": "Acme"}])
    await Extractor(engine).extract(
        user_message="msg",
        current_submission=CustomerSubmission(),
    )
    messages = engine.last_messages
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


@pytest.mark.asyncio
async def test_extract_prompt_includes_user_message_verbatim():
    engine = FakeEngine([{"business_name": "Acme"}])
    await Extractor(engine).extract(
        user_message="distinctive-marker-abc-42",
        current_submission=CustomerSubmission(),
    )
    assert "distinctive-marker-abc-42" in engine.last_messages[1]["content"]


@pytest.mark.asyncio
async def test_extract_prompt_includes_current_submission():
    engine = FakeEngine([{"ein": "12-3456789"}])
    await Extractor(engine).extract(
        user_message="msg",
        current_submission=CustomerSubmission(business_name="Existing-Acme"),
    )
    assert "Existing-Acme" in engine.last_messages[1]["content"]


@pytest.mark.asyncio
async def test_extract_prompt_includes_schema_reference():
    """Schema JSON injected via {schema} placeholder."""
    engine = FakeEngine([{"business_name": "x"}])
    await Extractor(engine).extract(
        user_message="msg",
        current_submission=CustomerSubmission(),
    )
    # CustomerSubmission appears as the schema's title
    assert "CustomerSubmission" in engine.last_messages[1]["content"]


# ---------------------------------------------------------------------------
# Phase 2.4 — Correction memory injection
# ---------------------------------------------------------------------------


def _make_entry(field: str = "business_name", wrong: str = "Acme", correct: str = "Acme LLC") -> CorrectionMemoryEntry:
    return CorrectionMemoryEntry(
        field_path=field,
        wrong_value=wrong,
        correct_value=correct,
        explanation=None,
        created_at=datetime.now(timezone.utc),
    )


class _StubMemory:
    """Memory stub that returns a fixed list of entries."""
    def __init__(self, entries):
        self._entries = entries
        self.called_with = []

    def get_relevant(self, tenant, **kwargs):
        self.called_with.append(tenant)
        return self._entries


@pytest.mark.asyncio
async def test_extractor_injects_corrections_when_memory_enabled(monkeypatch):
    """Prompt spy: 'RECENT CORRECTIONS' header and entries appear in user content."""
    from accord_ai import request_context
    monkeypatch.setattr(request_context, "get_tenant", lambda: "acme")

    engine = FakeEngine([{"business_name": "Acme LLC"}])
    stub = _StubMemory([_make_entry("ein", "123", "12-3456789")])
    extractor = Extractor(engine, memory=stub, memory_enabled=True)
    await extractor.extract(user_message="our EIN is 12-3456789", current_submission=CustomerSubmission())

    user_content = engine.last_messages[1]["content"]
    assert "RECENT CORRECTIONS" in user_content
    assert "ein" in user_content
    assert stub.called_with == ["acme"]


@pytest.mark.asyncio
async def test_extractor_omits_block_when_no_corrections(monkeypatch):
    """Empty memory → no header, existing prompt shape preserved."""
    from accord_ai import request_context
    monkeypatch.setattr(request_context, "get_tenant", lambda: "acme")

    engine = FakeEngine([{"business_name": "Acme"}])
    stub = _StubMemory([])  # no entries
    extractor = Extractor(engine, memory=stub, memory_enabled=True)
    await extractor.extract(user_message="hello", current_submission=CustomerSubmission())

    user_content = engine.last_messages[1]["content"]
    assert "RECENT CORRECTIONS" not in user_content
    assert "hello" in user_content


@pytest.mark.asyncio
async def test_extractor_omits_block_when_memory_disabled(monkeypatch):
    """memory_enabled=False → corrections block skipped even with entries."""
    from accord_ai import request_context
    monkeypatch.setattr(request_context, "get_tenant", lambda: "acme")

    engine = FakeEngine([{"business_name": "Acme"}])
    stub = _StubMemory([_make_entry()])
    extractor = Extractor(engine, memory=stub, memory_enabled=False)
    await extractor.extract(user_message="hello", current_submission=CustomerSubmission())

    user_content = engine.last_messages[1]["content"]
    assert "RECENT CORRECTIONS" not in user_content


@pytest.mark.asyncio
async def test_extractor_survives_memory_query_failure(monkeypatch):
    """Memory raises → extractor still extracts (block skipped, warning logged)."""
    import logging
    from accord_ai import request_context
    monkeypatch.setattr(request_context, "get_tenant", lambda: "acme")

    class _BoomMemory:
        def get_relevant(self, **kwargs):
            raise RuntimeError("db unreachable")

    engine = FakeEngine([{"business_name": "Acme"}])
    extractor = Extractor(engine, memory=_BoomMemory(), memory_enabled=True)

    diff = await extractor.extract(user_message="hello", current_submission=CustomerSubmission())
    assert diff.business_name == "Acme"  # extraction succeeded
    user_content = engine.last_messages[1]["content"]
    assert "RECENT CORRECTIONS" not in user_content


# ---------------------------------------------------------------------------
# Phase 2.4 — Entry formatting
# ---------------------------------------------------------------------------


def test_entry_as_prompt_line_truncates_long_values():
    """Values > 40 chars are truncated with ellipsis."""
    entry = CorrectionMemoryEntry(
        field_path="business_name",
        wrong_value="A" * 50,
        correct_value="B" * 50,
        explanation=None,
        created_at=datetime.now(timezone.utc),
    )
    line = entry.as_prompt_line()
    assert "…" in line
    # Each value slot is max 40 chars → truncated to 39 + ellipsis
    wrong_part = line.split("was ")[1].split(" →")[0]
    assert len(wrong_part) <= 40


def test_entry_as_prompt_line_includes_explanation_when_present():
    entry = CorrectionMemoryEntry(
        field_path="ein",
        wrong_value="old",
        correct_value="new",
        explanation="Typo in EIN",
        created_at=datetime.now(timezone.utc),
    )
    line = entry.as_prompt_line()
    assert "Typo in EIN" in line
    assert "(" in line and ")" in line


# ---------------------------------------------------------------------------
# Phase 3.3 — ExtractionContext rendering (4 tests)
# ---------------------------------------------------------------------------

from accord_ai.extraction.context import EMPTY_CONTEXT, ExtractionContext


@pytest.mark.asyncio
async def test_extractor_omits_context_block_when_empty():
    """EMPTY_CONTEXT → no FLOW/FOCUS/QUESTION headers in user content."""
    engine = FakeEngine([{"business_name": "Acme"}])
    extractor = Extractor(engine)
    await extractor.extract(
        user_message="We are Acme.",
        current_submission=CustomerSubmission(),
        context=EMPTY_CONTEXT,
    )
    user_content = engine.last_messages[1]["content"]
    assert "FLOW:" not in user_content
    assert "FOCUS FIELDS:" not in user_content
    assert "QUESTION ASKED:" not in user_content


@pytest.mark.asyncio
async def test_extractor_renders_focus_fields():
    """Context with expected_fields → 'FOCUS FIELDS:' appears in user content."""
    engine = FakeEngine([{"ein": "12-3456789"}])
    extractor = Extractor(engine)
    ctx = ExtractionContext(
        current_flow="business_identity",
        expected_fields=("ein", "entity_type"),
    )
    await extractor.extract(
        user_message="Our EIN is 12-3456789.",
        current_submission=CustomerSubmission(),
        context=ctx,
    )
    user_content = engine.last_messages[1]["content"]
    assert "FLOW: business_identity" in user_content
    assert "FOCUS FIELDS: ein, entity_type" in user_content


@pytest.mark.asyncio
async def test_extractor_renders_question_text():
    """Context with question_text → 'QUESTION ASKED:' appears in user content."""
    engine = FakeEngine([{"business_name": "Acme"}])
    extractor = Extractor(engine)
    ctx = ExtractionContext(
        current_flow="greet",
        question_text="What is your business name?",
    )
    await extractor.extract(
        user_message="Acme LLC",
        current_submission=CustomerSubmission(),
        context=ctx,
    )
    user_content = engine.last_messages[1]["content"]
    assert "QUESTION ASKED: What is your business name?" in user_content
    assert "also extract any other fields the user provides" in user_content


@pytest.mark.asyncio
async def test_extractor_with_both_corrections_and_context(monkeypatch):
    """Corrections block + context block both appear, user message is labeled."""
    from accord_ai import request_context
    monkeypatch.setattr(request_context, "get_tenant", lambda: "acme")

    engine = FakeEngine([{"ein": "12-3456789"}])
    stub = _StubMemory([_make_entry("ein", "bad-ein", "12-3456789")])
    extractor = Extractor(engine, memory=stub, memory_enabled=True)
    ctx = ExtractionContext(
        current_flow="business_identity",
        expected_fields=("ein",),
        question_text="What is your EIN?",
    )
    await extractor.extract(
        user_message="EIN is 12-3456789",
        current_submission=CustomerSubmission(),
        context=ctx,
    )
    user_content = engine.last_messages[1]["content"]
    assert "RECENT CORRECTIONS" in user_content
    assert "FLOW: business_identity" in user_content
    assert "FOCUS FIELDS: ein" in user_content
    assert "USER MESSAGE: EIN is 12-3456789" in user_content
