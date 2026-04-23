"""LOB-agnostic critical fields — imported by all LOB plugins.

These are the fields every ACORD 125 front-page needs + the policy
header all forms share. Sourced from v3 CA and GL plugins where both
lists overlap.
"""
from __future__ import annotations

from typing import List, Tuple

CriticalField = Tuple[str, str]

COMMON_CRITICAL: List[CriticalField] = [
    ("business_name",                "business_name is required"),
    ("mailing_address.line_one",     "mailing_address.line_one is required"),
    ("mailing_address.city",         "mailing_address.city is required"),
    ("mailing_address.state",        "mailing_address.state is required"),
    ("mailing_address.zip_code",     "mailing_address.zip_code is required"),
    ("entity_type",                  "entity_type is required (corporation/partnership/llc/...)"),
    ("ein",                          "ein (tax_id) is required"),
    ("business_start_date",          "business_start_date is required"),
    ("policy_dates.effective_date",  "policy_dates.effective_date is required"),
    ("policy_status",                "policy_status is required (new/renewal/rewrite)"),
]
