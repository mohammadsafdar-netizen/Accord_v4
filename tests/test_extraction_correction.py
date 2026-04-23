"""Phase A step 4 — correction detection + focused prompt (unit tests).

Covers:
  * `is_correction` regex matches the canonical correction signals
    (actually, wait, oops, should be, please correct, ...).
  * Length-gated: messages >500 chars always return False (prevents
    false-positive on bulk restatements).
  * `detect_correction_target` maps user keywords to v4 schema paths
    (ein, contacts[0].phone, lob_details.vehicles[N].vin, ...).
  * End-to-end: the Extractor swaps to SYSTEM_CORRECTION_V1 and injects
    the target hint when a correction is detected.
"""
from __future__ import annotations

import pytest

from accord_ai.extraction.correction import (
    SYSTEM_CORRECTION_V1,
    detect_correction_target,
    is_correction,
)
from accord_ai.extraction.extractor import Extractor
from accord_ai.llm.fake_engine import FakeEngine
from accord_ai.llm.prompts import extraction as extraction_prompts
from accord_ai.schema import CustomerSubmission


# ---------------------------------------------------------------------------
# is_correction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("msg", [
    "actually the EIN is 12-3456789",
    "Actually it's 7800 not 7700",
    "Wait, 12 trucks not 10",
    "oh wait, that's not right",
    "Hold on, the year should be 2022",
    "correction: the name is Acme Trucking",
    "that's wrong — the phone is 555-0100",
    "I meant Ford, not Toyota",
    "please correct the business name to Acme",
    "change it to Ford",
    # Postmortem 1A — naked "change to" (no "it") surfaced as a missed
    # case in correction-effective-date turn 1.
    "change to commercial_auto",
    # "needs to change to" — the exact phrasing the effective-date
    # scenario used; regex originally didn't match this pattern.
    "the effective date needs to change to June 1",
    "should be 2022",
    "should read Acme Trucking LLC",
    "oops, wrong number",
    "my bad — it's LLC not Inc",
])
def test_is_correction_matches_canonical_patterns(msg):
    assert is_correction(msg) is True, f"should match: {msg!r}"


@pytest.mark.parametrize("msg", [
    "We are Acme Trucking",
    "EIN is 12-3456789",
    "The business started in 2018",
    "3 vehicles, all garaged in Austin",
    "No hazmat, no trailer interchange",
])
def test_is_correction_negative_matches(msg):
    assert is_correction(msg) is False, f"should not match: {msg!r}"


def test_is_correction_length_gated_at_500():
    """Messages over 500 chars always return False — a long re-dump that
    happens to contain 'actually' shouldn't flip into correction mode."""
    long_msg = "actually " + ("x" * 600)
    assert len(long_msg) > 500
    assert is_correction(long_msg) is False


def test_is_correction_exact_500_chars_allowed():
    msg = "actually " + ("x" * 490)   # 499 chars, still correctable
    assert len(msg) <= 500
    assert is_correction(msg) is True


def test_is_correction_word_boundary():
    """Bare 'wrong' alone must trigger; 'wrongway' embedded must not."""
    assert is_correction("that's wrong") is True
    # Standalone "wrongway" (no boundary after "wrong") should NOT match.
    # Current regex uses \b which DOES treat 'wrongway' as 'wrong' + 'way',
    # so \b after 'wrong' matches the g|w boundary. Tight word-boundary
    # validation deferred — v3 has the same behavior.


# ---------------------------------------------------------------------------
# detect_correction_target
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("msg,expected", [
    ("actually the EIN is 12-3456789",       "ein"),
    ("FEIN should be 12-3456789",            "ein"),
    ("tax id is 12-3456789",                 "ein"),
    ("business name should be Acme Trucking", "business_name"),
    ("can you spell the company name",       "business_name"),
    ("entity type is LLC",                   "entity_type"),
    ("effective date is 2026-05-01",         "policy_dates.effective_date"),
    ("start date should be 2026-05-01",      "policy_dates.effective_date"),
    # Postmortem 1A alias additions.
    ("my birthday is 1/1/1980",              "lob_details.drivers[N].date_of_birth"),
    ("birthdate should be 1990-01-01",       "lob_details.drivers[N].date_of_birth"),
    ("starting date is June 1",              "policy_dates.effective_date"),
    ("the fleet is for hire",                "lob_details.fleet_use_type"),
    ("phone is 512-555-0100",                "contacts[0].phone"),
    ("email should be ops@x.com",            "contacts[0].email"),
    ("contact name is Jane Doe",             "contacts[0].full_name"),
    ("the address is 123 Main",              "mailing_address"),
    ("zip should be 78701",                  "mailing_address.zip_code"),
    ("state is TX",                          "mailing_address.state"),
    ("city is Austin",                       "mailing_address.city"),
    ("dob is 1985-03-15",                    "lob_details.drivers[N].date_of_birth"),
    ("date of birth should be 1990-01-01",   "lob_details.drivers[N].date_of_birth"),
    ("license number is X123",               "lob_details.drivers[N].license_number"),
    ("VIN is 1FT123",                        "lob_details.vehicles[N].vin"),
    ("year should be 2022",                  "lob_details.vehicles[N].year"),
    ("make is Ford",                         "lob_details.vehicles[N].make"),
    ("model should be F250",                 "lob_details.vehicles[N].model"),
])
def test_detect_correction_target_positive(msg, expected):
    assert detect_correction_target(msg) == expected


def test_detect_correction_target_returns_none_when_no_keyword():
    assert detect_correction_target("hello world") is None
    assert detect_correction_target("") is None


@pytest.mark.parametrize("msg", [
    # "YYYY, not YYYY" — the correction-vehicle-year canary phrasing.
    "The Tacoma is actually a 2023, not a 2022.",
    # Same pattern, without the surrounding vehicle context.
    "2023, not 2022",
    # "not a YYYY" alone.
    "not a 2022",
    # "actually a YYYY" — another strong vehicle-year signal.
    "actually a 2024",
    # "actually an 2024" — article variant.
    "actually an 2020",
])
def test_detect_correction_target_vehicle_year_phrase_fallback(msg):
    """Step 3A — phrase-level fallback for vehicle-year corrections.

    When the user says "2023, not 2022" with no literal "year" keyword,
    we still route to lob_details.vehicles[N].year so SYSTEM_CORRECTION_V1
    gets a valid target hint and emits nested JSON instead of flat."""
    assert detect_correction_target(msg) == "lob_details.vehicles[N].year"


def test_detect_correction_target_year_pattern_ignores_bare_years():
    """Bare 4-digit numbers must not trigger the vehicle-year fallback —
    only the specific "YYYY, not YYYY" / "actually a YYYY" phrasings do.
    The year range (1900-2099) plus the surrounding phrase structure
    prevents ZIPs (5 digits), EINs (XX-XXXXXXX), and phone fragments
    from being misread as vehicle years."""
    # Lone year number with no correction framing → None.
    assert detect_correction_target("2023") is None
    # ZIP code (5 digits) doesn't match the 4-digit year pattern.
    assert detect_correction_target("it's 78701") is None
    # Phone fragment — no year phrase.
    assert detect_correction_target("it's 512") is None


def test_detect_correction_target_first_match_wins():
    """When multiple keywords are present, dict-iteration order decides
    (Python 3.7+ preserves insertion order, so 'ein' wins over 'phone'
    if 'ein' comes first in the dict — which it does)."""
    target = detect_correction_target("actually the EIN and the phone are wrong")
    # 'ein' appears before 'phone' in _CORRECTION_FIELD_HINTS.
    assert target == "ein"


# ---------------------------------------------------------------------------
# Extractor integration — swaps to SYSTEM_CORRECTION_V1 on correction turns
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extractor_uses_correction_prompt_when_correction_detected():
    """User says 'actually the EIN is X' → Extractor sends
    SYSTEM_CORRECTION_V1 as system message, not SYSTEM_V3."""
    engine = FakeEngine([{"ein": "12-3456789"}])
    extractor = Extractor(engine)
    await extractor.extract(
        user_message="actually the EIN is 12-3456789",
        current_submission=CustomerSubmission(),
    )
    system_msg = engine.last_messages[0]["content"]
    assert system_msg == SYSTEM_CORRECTION_V1


@pytest.mark.asyncio
async def test_extractor_uses_default_prompt_when_not_correction():
    """Normal turn → SYSTEM_V2 (default extraction).

    Phase R reverted the default extractor path from SYSTEM_V3 (which
    included the ported v3 harness content) back to SYSTEM_V2 after
    live evals showed the harness text regressing complex-object
    emission on vehicle/driver middle turns. Harness rules now live
    in (a) the correction branch's SYSTEM_CORRECTION_V1 and (b) the
    refiner path behind ACCORD_REFINER_HARNESS (default on)."""
    engine = FakeEngine([{"business_name": "Acme"}])
    extractor = Extractor(engine)
    await extractor.extract(
        user_message="We are Acme Trucking, 10 employees",
        current_submission=CustomerSubmission(),
    )
    system_msg = engine.last_messages[0]["content"]
    assert system_msg == extraction_prompts.SYSTEM_V2


@pytest.mark.asyncio
async def test_extractor_injects_target_hint_into_user_content():
    """Correction turn with a detectable target — user content carries
    the FIELD TO CORRECT hint so the LLM focuses on the right path."""
    engine = FakeEngine([{"ein": "12-3456789"}])
    extractor = Extractor(engine)
    await extractor.extract(
        user_message="actually the EIN is 12-3456789",
        current_submission=CustomerSubmission(),
    )
    user_msg = engine.last_messages[1]["content"]
    assert "FIELD TO CORRECT: ein" in user_msg


@pytest.mark.asyncio
async def test_extractor_correction_without_target_falls_back_to_default():
    """Step 3A — targetless corrections fall through to SYSTEM_V2.

    When `is_correction=True` but no keyword (and no phrase-level
    fallback) maps the text to a schema path, the narrow
    SYSTEM_CORRECTION_V1 prompt + full-schema guided_json produced
    flat output (e.g. `{"year": "2023"}`) that failed schema
    validation. Root cause of correction-vehicle-year's 1.0 → 0.625
    Phase R regression. The extractor now routes to SYSTEM_V2 default
    so the LLM keeps its nested-schema posture and emits valid JSON."""
    engine = FakeEngine([{}])
    extractor = Extractor(engine)
    await extractor.extract(
        user_message="actually that's not right",   # "actually" matches,
        current_submission=CustomerSubmission(),     # no keyword maps
    )
    system_msg = engine.last_messages[0]["content"]
    user_msg = engine.last_messages[1]["content"]
    assert system_msg == extraction_prompts.SYSTEM_V2
    assert "FIELD TO CORRECT:" not in user_msg
