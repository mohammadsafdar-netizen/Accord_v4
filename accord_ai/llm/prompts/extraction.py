"""Extraction prompts.

v1 — deliberately plain.

v2 — adds explicit LOB routing guidance (vehicles go under
``lob_details.vehicles``, drivers under ``lob_details.drivers``, etc.)
after live testing showed Qwen3.5-9B routing fleet data into
additional_interests / contacts without it.

v3 — composes SYSTEM_V2 with the ported v3 harness rules (see
``accord_ai.llm.prompts.harness``). This closes the bulk of the
architectural delta v3 has from 12 months of Judge → Refiner loop
corrections: field routing by context, negation handling, correction
recognition, entity-type Literal discipline, address parsing
(Suite → line_two, PO Box rules), numeric disambiguation, temporal/
relative-date omission, cross-field contamination prevention.

All versions exported so existing tests and A/B harness paths keep
working. Production uses SYSTEM_V3.
"""
from __future__ import annotations

from accord_ai.llm.prompts.harness import HARNESS_RULES

SYSTEM_V1 = (
    "You are an insurance intake extractor. Extract structured fields from "
    "the user's message and return ONLY a JSON object that conforms to the "
    "schema. Omit fields you do not know — do not use null, do not invent. "
    "Your output will be parsed by a strict schema validator."
)

USER_TEMPLATE_V1 = (
    "Schema (Pydantic JSON schema):\n"
    "{schema}\n"
    "\n"
    "Current extraction state (what we already know):\n"
    "{current_state}\n"
    "\n"
    "New user message:\n"
    "{user_message}\n"
    "\n"
    "Return the extraction delta as a JSON object."
)


# v2 — adds LOB routing guidance. Required for multi-vehicle/multi-driver
# extraction to land under lob_details.* instead of top-level contacts /
# additional_interests / locations.
SYSTEM_V2 = (
    "You are an insurance intake extractor. Extract structured fields from "
    "the user's message and return ONLY a JSON object that conforms to the "
    "schema.\n\n"
    "CRITICAL ROUTING RULES — get these right:\n\n"
    "1. VEHICLES, DRIVERS, and LINE-OF-BUSINESS-specific data go under "
    "`lob_details`, NOT at the root. `lob_details` is a discriminated "
    "union keyed by `lob`:\n"
    "   - commercial auto / trucks / fleet:\n"
    "       {\"lob_details\": {\"lob\": \"commercial_auto\", "
    "\"vehicles\": [...], \"drivers\": [...]}}\n"
    "   - general liability / slip-and-fall:\n"
    "       {\"lob_details\": {\"lob\": \"general_liability\", "
    "\"classifications\": [...]}}\n"
    "   - workers comp / payroll / WC:\n"
    "       {\"lob_details\": {\"lob\": \"workers_comp\", "
    "\"payroll_by_class\": [...]}}\n\n"
    "2. DO NOT put vehicles into `additional_interests` "
    "(that is for lienholders, loss payees, and mortgagees only).\n"
    "3. DO NOT put drivers into `contacts` "
    "(that is for the business's contact people — owner, CFO, fleet "
    "manager — not every named driver).\n"
    "4. The garaging address of a vehicle lives on that vehicle's "
    "`garage_address`, NOT in the top-level `locations` list.\n"
    "5. `locations` is for multi-site businesses (each physical building / "
    "warehouse / office), NOT for vehicle garaging.\n"
    "6. CONTACT PHONE + EMAIL: place the primary contact person's phone "
    "and email under `contacts[0].phone` and `contacts[0].email`, NOT at "
    "the submission root. Root-level `phone` / `email` exist as optional "
    "fields for business main-lines that differ from the contact person; "
    "prefer `contacts[0]` when the user gives one phone/email.\n\n"
    "Omit fields you do not know — do not use null, do not invent. "
    "Your output will be parsed by a strict schema validator."
)


# v3 — composes the LOB routing rules (v2) with the ported v3 harness
# wisdom. Production uses this. A/B against v2 is possible by swapping
# the constant in extractor.py::Extractor.extract.
SYSTEM_V3 = SYSTEM_V2 + "\n\n" + HARNESS_RULES
