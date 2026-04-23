"""Correction-turn detection + focused prompt (Phase A step 4).

Ported from accord_ai_v3/extraction/prompts.py::is_correction and
build_correction_prompt. When the user posts "actually it's 7800 not
7700" or "the EIN should be 12-3456789," we want the LLM to extract
ONLY the corrected field — not re-extract the whole submission. v3's
approach: detect correction → swap to a smaller focused system prompt
that emphasizes "output only changed fields; {} if nothing."

v4 adaptations:
  * v3 paths → v4 paths in the field-hint map (ein, not named_insured.tax_id)
  * Uses v4's existing SYSTEM_V3 harness content as the base, with the
    correction directive layered on top — preserves the LOB routing
    discipline while adding the correction posture.
"""
from __future__ import annotations

import re
from typing import Dict, Optional

from accord_ai.llm.prompts.harness import HARNESS_RULES


# ---------------------------------------------------------------------------
# Correction detection — regex ported verbatim from v3
# ---------------------------------------------------------------------------

_CORRECTION_RE = re.compile(
    r"\b(actually|correction|wrong|mistake"
    # "change to" / "change it to" / "change that to" / "change this to"
    r"|change\s+(?:it\s+|that\s+|this\s+)?to"
    r"|i\s+meant"
    r"|that(?:'s|\s+is)\s+(?:not\s+right|incorrect|wrong)"
    r"|please\s+(?:correct|fix|change|update)"
    # "needs to change to" surfaced as a missed case in postmortem 1A
    r"|needs?\s+to\s+(?:change|be)"
    r"|should\s+(?:be|read|say)"
    r"|oops|my\s+bad"
    r"|^(?:oh\s+)?wait\b|^hold\s+on\b)\b",
    re.IGNORECASE,
)

# Length cutoff — a "correction" longer than 500 chars is almost always
# a bulk re-dump that happens to include the word "actually" somewhere.
# v3's threshold. Prevents the focused prompt from firing on a full
# restatement of the intake.
_CORRECTION_MAX_LEN = 500


def is_correction(message: str) -> bool:
    """True if the message looks like a user correction of a prior field.

    Long messages (>500 chars) always return False — see module docstring.
    """
    if len(message) > _CORRECTION_MAX_LEN:
        return False
    return bool(_CORRECTION_RE.search(message))


# ---------------------------------------------------------------------------
# Field targeting — keyword → v4 schema path
# ---------------------------------------------------------------------------

# Remapped from v3's _CORRECTION_FIELD_HINTS. The `[N]` placeholder
# communicates "any index" to the LLM; pydantic's diff merge handles
# which index actually needs to update.
#
# Deliberately omitted from this map (v3's decision): "llc", "corporation",
# "s-corp" — they false-positive inside business names ("Johnson LLC"
# triggering entity_type correction). Users correcting entity type
# say "entity type" explicitly.
_CORRECTION_FIELD_HINTS: Dict[str, str] = {
    "ein":           "ein",
    "fein":          "ein",
    "tax id":        "ein",
    "tax_id":        "ein",
    "business name": "business_name",
    "company name":  "business_name",
    "spell":         "business_name",
    "entity type":   "entity_type",
    "entity_type":   "entity_type",
    "effective date": "policy_dates.effective_date",
    "start date":    "policy_dates.effective_date",
    # Postmortem 1A additions — natural-language variants the v3 harness
    # never had to write down because v3's detector is pattern-based
    # against the same user surface.
    "starting date": "policy_dates.effective_date",
    "phone":         "contacts[0].phone",
    "email":         "contacts[0].email",
    "contact name":  "contacts[0].full_name",
    "address":       "mailing_address",
    "zip":           "mailing_address.zip_code",
    "state":         "mailing_address.state",
    "city":          "mailing_address.city",
    "dob":           "lob_details.drivers[N].date_of_birth",
    "date of birth": "lob_details.drivers[N].date_of_birth",
    # "birthday" / "birthdate" are how broker conversations actually
    # phrase date-of-birth corrections ("my birthday is March 12 not
    # March 21"). Postmortem 1A: correction-driver-dob turn 2 returned
    # empty extraction because these words weren't mapped.
    "birthday":      "lob_details.drivers[N].date_of_birth",
    "birthdate":     "lob_details.drivers[N].date_of_birth",
    "license":       "lob_details.drivers[N].license_number",
    "vin":           "lob_details.vehicles[N].vin",
    "year":          "lob_details.vehicles[N].year",
    "make":          "lob_details.vehicles[N].make",
    "model":         "lob_details.vehicles[N].model",
    "fleet":         "lob_details.fleet_use_type",
}


# Vehicle-year correction patterns. Step 3A — the scenario
# correction-vehicle-year uses "actually a 2023, not a 2022" which has
# no literal "year" keyword in the text. Without a target hint,
# SYSTEM_CORRECTION_V1 emits narrow {"year": "2023"} that fails the
# nested schema validation (year must live at
# lob_details.vehicles[0].year). Two phrasings are strong enough
# signals on their own to mean vehicle model-year:
#   * "<YEAR>, not <YEAR>"  (e.g. "2023, not 2022")
#   * "actually a <YEAR>"   (e.g. "actually a 2023")
# 1900-2099 bound avoids matching phone area codes / ZIPs / EINs.
_VEHICLE_YEAR_CORRECTION_RE = re.compile(
    r"(?:"
    r"(?:19|20)\d{2}\s*,?\s*not\s+(?:a\s+)?(?:19|20)\d{2}"
    r"|actually\s+(?:a\s+|an\s+)?(?:19|20)\d{2}"
    r"|not\s+(?:a\s+)?(?:19|20)\d{2}"
    r")",
    re.IGNORECASE,
)


def detect_correction_target(message: str) -> Optional[str]:
    """Return the v4 schema path the correction seems to target, or None.

    Returns the bare path (e.g. ``"ein"`` or ``"lob_details.vehicles[N].vin"``);
    caller composes this into the prompt as needed.
    """
    msg_lower = message.lower()
    for keyword, field_path in _CORRECTION_FIELD_HINTS.items():
        if keyword in msg_lower:
            return field_path
    # Phrase-level fallback — "YYYY, not YYYY" / "actually a YYYY" signal
    # vehicle model-year corrections even without the word "year".
    if _VEHICLE_YEAR_CORRECTION_RE.search(message):
        return "lob_details.vehicles[N].year"
    return None


# ---------------------------------------------------------------------------
# Focused system prompt
# ---------------------------------------------------------------------------

# The correction-turn system prompt. Same harness rules as SYSTEM_V3, but
# with the default extraction posture swapped for a correction posture:
# "return ONLY changed fields; empty {} if nothing identifiable."
SYSTEM_CORRECTION_V1 = (
    "You are an insurance-intake CORRECTION engine. The user is fixing "
    "a previously-extracted field — do NOT re-extract the whole "
    "submission.\n\n"
    "Rules:\n"
    "1. Extract ONLY the field(s) being corrected. Leave every other "
    "field absent from your output — pydantic's diff merge preserves "
    "unchanged fields.\n"
    "2. If you cannot determine what is being corrected, output {} "
    "(empty object). An empty diff is always safer than a wrong diff.\n"
    "3. Use the SAME nested JSON structure as the schema — nested "
    "objects stay nested; list indices stay consistent with what the "
    "submission already has.\n"
    "4. Do not invent. If the user says 'actually the EIN is 12-3456789' "
    "output only `{\"ein\": \"12-3456789\"}`. Do not add fields they "
    "didn't mention.\n\n" + HARNESS_RULES
)
