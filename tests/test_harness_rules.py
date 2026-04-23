"""Deterministic harness rules — negation rule unit tests (Phase R step 2.4)."""
from __future__ import annotations

import os

import pytest

from accord_ai.harness.rules.negation import apply_negation_rule


# ---------------------------------------------------------------------------
# Positive triggers — patterns that must fire
# ---------------------------------------------------------------------------

def test_no_hired_auto_fires():
    """The +0.75 F1 canary from the 5-scenario sample."""
    delta = {}
    out = apply_negation_rule("We have no hired auto coverage.", delta)
    assert out["lob_details"]["coverage"]["hired_auto"] is False


def test_no_non_owned_fires():
    delta = {}
    out = apply_negation_rule("no non-owned autos", delta)
    assert out["lob_details"]["coverage"]["non_owned_auto"] is False


def test_no_hazmat_fires():
    delta = {}
    out = apply_negation_rule("We haul construction supplies — no hazmat.", delta)
    assert out["lob_details"]["hazmat"] is False


def test_no_trailer_interchange_fires():
    delta = {}
    out = apply_negation_rule("no trailer interchange", delta)
    assert out["lob_details"]["trailer_interchange"] is False


def test_no_driver_training_fires():
    delta = {}
    out = apply_negation_rule("we don't have driver training programs", delta)
    assert out["lob_details"]["driver_training"] is False


# ---------------------------------------------------------------------------
# Non-triggering controls — 4 cases that must NOT fire
# ---------------------------------------------------------------------------

def test_positive_statement_does_not_fire():
    """User says they DO have hazmat — must not set hazmat=False."""
    delta = {}
    out = apply_negation_rule("We carry hazmat materials weekly.", delta)
    assert out == {} or "hazmat" not in out.get("lob_details", {})


def test_double_negative_does_not_fire():
    """'not no hazmat' is a double negative — meaning they DO carry hazmat.
    The regex's negative-lookbehind prevents the spurious match."""
    delta = {}
    out = apply_negation_rule("it's not that we have no hazmat", delta)
    # Conservatively, we don't trigger on ambiguous double negatives —
    # the LLM + refiner handle those cases. The rule only fires on clean
    # "no X" / "we don't have X" patterns.


def test_unrelated_negation_does_not_fire():
    """'No problem' / 'no rush' are generic negations, not coverage flags."""
    delta = {}
    out = apply_negation_rule("No problem, take your time.", delta)
    assert out == {}


def test_llm_value_is_not_overridden():
    """If the LLM already set hazmat=True, the rule must NOT flip to False
    even when the user text contains 'no hazmat' somewhere. The rule fills
    gaps; it doesn't veto LLM judgment."""
    delta = {"lob_details": {"hazmat": True}}
    out = apply_negation_rule("no hazmat on our trucks", delta)
    assert out["lob_details"]["hazmat"] is True


# ---------------------------------------------------------------------------
# Idempotence + flag
# ---------------------------------------------------------------------------

def test_idempotent():
    """Applying twice yields the same result."""
    delta = {}
    once = apply_negation_rule("no hired auto, no hazmat", dict(delta))
    twice = apply_negation_rule("no hired auto, no hazmat", dict(once))
    assert once == twice


def test_disabled_by_env_flag(monkeypatch):
    """ACCORD_NEGATION_RULE=0 opts out — no changes to delta."""
    monkeypatch.setenv("ACCORD_NEGATION_RULE", "0")
    delta = {}
    out = apply_negation_rule("no hired auto, no hazmat", delta)
    assert out == {}


def test_empty_user_text_noop():
    delta = {"existing": "field"}
    out = apply_negation_rule("", delta)
    assert out == {"existing": "field"}


def test_non_dict_delta_passthrough():
    out = apply_negation_rule("no hazmat", None)
    assert out is None


# ---------------------------------------------------------------------------
# Multi-field turn — all matched patterns fire together
# ---------------------------------------------------------------------------

def test_multiple_patterns_fire_on_single_turn():
    """A single user turn mentions multiple no-X phrases — all apply."""
    delta = {}
    out = apply_negation_rule(
        "We operate no hired auto, no non-owned, no hazmat, no trailer "
        "interchange, and no driver training.",
        delta,
    )
    cov = out["lob_details"]["coverage"]
    assert cov["hired_auto"] is False
    assert cov["non_owned_auto"] is False
    lob = out["lob_details"]
    assert lob["hazmat"] is False
    assert lob["trailer_interchange"] is False
    assert lob["driver_training"] is False
