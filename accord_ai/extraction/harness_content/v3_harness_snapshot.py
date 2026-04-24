"""Harness content snapshots for extraction prompts.

Three variants available:
  - V3_HARNESS_FULL: LEGACY v1.0 content (was v3's harness.md; kept for Step 25 A/B/C/D matrix comparisons).
  - V3_HARNESS_LIGHT: 5 highest-signal rules from the legacy content.
  - V3_CORE_HARNESS: CURRENT v3 core.md v6.1 (curated). Loaded from accord_ai/harness/base_harness.md.
  - V3_CA_LOB_HARNESS: v3 lobs/commercial_auto.md (v3.0). Loaded from accord_ai/harness/lobs/commercial_auto.md.
  - V3_GL_LOB_HARNESS: v3 lobs/general_liability.md (v1.0). Loaded from accord_ai/harness/lobs/general_liability.md.

The v3 eval (Step 25 research, 2026-04-22) identified:
  - harness.md was legacy v1.1; core.md v6.1 is the actual v3 production harness
  - LOB-specific harness is composed per active LOB, not a single monolithic file
  - Harness is concatenated INSIDE the system message (at the END, after schema)
    not passed as a separate system message

Helper: compose_harness_for_lobs(active_lobs) returns the composed core + LOB text
v3 would use for a given active-LOB set.
"""
from pathlib import Path

_HARNESS_DIR = Path(__file__).resolve().parent.parent.parent / "harness"


def _read(relative: str) -> str:
    path = _HARNESS_DIR / relative
    if not path.exists():
        return ""
    return path.read_text()


# Current curated content (loaded from files on disk; edits to those files
# are picked up on next process restart, not at every call — these are
# module-level constants deliberately).
V3_CORE_HARNESS = _read("base_harness.md")
V3_CA_LOB_HARNESS = _read("lobs/commercial_auto.md")
V3_GL_LOB_HARNESS = _read("lobs/general_liability.md")

_LOB_HARNESS_MAP = {
    "commercial_auto": V3_CA_LOB_HARNESS,
    "general_liability": V3_GL_LOB_HARNESS,
    # workers_comp intentionally omitted — v3 didn't ship a workers_comp.md
}


def compose_harness_for_lobs(
    active_lobs: list[str] | None = None,
    tenant: str | None = None,
) -> str:
    """Compose harness text from three tiers (v3 + v4 multi-tenant extension).

    Tier 1 — core.md (universal principles)
    Tier 2 — lobs/<lob>.md (per-active-LOB principles; v3 pattern)
    Tier 3 — brokers/<tenant>.md (per-broker overlay; v4 addition)

    The two-tier split between core.md (shared, engineering-controlled) and
    brokers/<tenant>.md (per-tenant, broker admin-controlled) follows the
    2026-04-24 multi-tenant research recommendation:
      - core.md is coupled to adapter training — changes require retrain
      - tenant overlay is freely editable — frequent changes safe

    Never-active LOBs and missing tenant overlays are silently excluded.
    Cross-LOB rule interference prevention (v3's hard-learned lesson).

    Args:
        active_lobs: LOB keys ("commercial_auto", "general_liability").
        tenant: Broker slug for per-tenant overlay (optional).

    Returns: Composed harness text with sections separated by blank lines.
    """
    from pathlib import Path as _Path

    parts = [V3_CORE_HARNESS.strip()] if V3_CORE_HARNESS.strip() else []

    if active_lobs:
        for lob in active_lobs:
            lob_content = _LOB_HARNESS_MAP.get(lob, "").strip()
            if lob_content:
                parts.append(lob_content)

    if tenant:
        tenant_path = _HARNESS_DIR / "brokers" / f"{tenant}.md"
        if tenant_path.exists():
            tenant_overlay = tenant_path.read_text().strip()
            if tenant_overlay:
                parts.append(tenant_overlay)

    if not parts:
        return ""
    return "\n\n".join(parts) + "\n"


# Legacy v1.0 content retained for matrix comparisons ONLY.
# Production path uses V3_CORE_HARNESS + compose_harness_for_lobs().
V3_HARNESS_FULL = """
# Extraction Harness v1.0

This document contains the extraction model's living rules. It is injected into every extraction prompt.

---

## Field Routing

Every value in a user message belongs to exactly ONE field. Determine the field by context, not by the value format alone.

- Numbers near "truck", "vehicle", "fleet", "rig" = `vehicle_count`. Numbers near "employee", "staff", "worker", "people", "of us" = `employees.total`. Never assume.
- When a number appears without context, do NOT extract it. Omit it and let the system re-ask.
- Dollar amounts near "premium" = `prior_insurance.premium`. Near "revenue"/"gross" = `annual_revenue`. Near "payroll" = `annual_payroll`. Near "value" with vehicle context = `cost_new` or `stated_amount`.
- An address on its own turn belongs to `mailing_address` unless the conversation is in the vehicle section (then it's `garaging_address`) or lienholder section (then it's the lienholder's address). A PO Box is NEVER a garaging address.

## Negation & Qualifiers

The negation word modifies the CLOSEST boolean concept.

- "no hazmat" = `hazmat: false`
- "no dash cams but yes driver training" = `dash_cam: false`, `driver_training: true`
- "we do NOT carry hazmat" = `hazmat: false`
- "used to not have GPS, now we do" = `telematics: true` (temporal: "now" wins)
- "cannot say we don't have driver training" = `driver_training: true` (double negative = positive)
- "No hired or non-owned auto" = `hired_auto: false` AND `non_owned_auto: false`

Never infer a boolean from context. "We haul fuel" does NOT imply `hazmat: true` unless the user explicitly mentions hazmat/hazardous/dangerous goods.

Silence ≠ false. `false` means the topic was discussed and denied; omission means it was never raised.

## Correction Recognition

When a user corrects a previous value, extract ONLY the corrected field.

- "actually it's 7800 not 7700" = correction to a number field. Extract the NEW value (7800).
- "we're an S-Corp not an LLC" = correction to `entity_type`. Extract `subchapter_s`. Do NOT touch `business_name`.
- "wait, 12 trucks not 10" = correction to `vehicle_count`. Extract 12.
- The word "actually", "wait", "correction", "wrong", "not X but Y", "should be" all signal corrections.
- On a correction turn: extract ONLY changed fields. Do NOT re-extract unchanged data.

## Entity Type Resolution

Priority order (highest wins):
1. Legal suffix in business name: "Inc"/"Corp" = corporation, "LLC" = llc, "LP" = partnership, "LLP" = limited_liability_partnership
2. Explicit user statement: "we're a corporation", "S-Corp"
3. NEVER infer from business description. "sole owner" does not mean entity_type = "individual" unless user explicitly says "sole proprietorship."

"We are an S-Corp" means entity_type = "subchapter_s". It does NOT mean business_name = "an S-Corp".

## Address Parsing

- "Suite", "Ste", "Unit", "Apt", "#" should go into `line_two`, not appended to `line_one`.
- Compound city names: "Kansas City" is ONE city. State comes from context, not the city name.
- "Same address" for garaging = copy from the FIRST vehicle's garaging address or business mailing address.
- PO Box = lienholder or mailing only. NEVER a garaging address.

## Numeric Disambiguation

When multiple numbers appear in one message, each belongs to a specific field based on surrounding words:

- XX-XXXXXXX (with hyphen, 9 digits total) = `ein` / FEIN. Strip dashes before storing; validate exactly 9 digits.
- 5-6 digits alone = could be ZIP or NAICS. "ZIP" or address context = `zip_code`. "NAICS" or "code" = `naics_code`.
- 10 digits or (XXX) XXX-XXXX = `phone`. Strip all formatting characters, keep 10 digits only.
- 4-digit year = `business_start_date` or vehicle `year` depending on context
- Dollar amounts ($X,XXX or Xk or X million) = route by context keyword

## VIN Normalization

- VIN is always exactly 17 characters, uppercase, alphanumeric (no I, O, Q).
- Strip spaces, dashes, and formatting before storing.
- Do NOT invent, guess, or pad VINs. If the user provides fewer than 17 characters, store what was given verbatim.
- "VIN: 1HGBH41JXMN109186" → `"1HGBH41JXMN109186"` (stripped, uppercase)

## Temporal Context

- "2 years ago" = DO NOT fabricate a specific date. Extract "approximately 2 years ago" as description, leave date blank.
- "started in 2018" = `business_start_date: "2018"` (year only, do not add month/day)
- "policy starts July 1" = `effective_date: "07/01/YYYY"` with current or next year
- "expires next March" = approximate; extract the month but be explicit about uncertainty
- NEVER auto-calculate expiration from effective date
- The model has NO knowledge of the current date. "Next month", "end of this month" → OMIT the date entirely.

## Source Fidelity

Extract only what is explicitly in the current user message. Never infer, assume, default, or pull from prior state.

- **Omit, never default.** If a value is not in the text, omit the field — not `null`, `""`, `0`, or a placeholder.
- **Never fabricate** DOBs, license numbers, phones, dates, EINs, VINs, or dollar amounts.
- **Corrections override.** "Actually 2022" after "2021" stores `2022`.
- **List items must be verbatim.** Every attribute of a list item must come from the text; otherwise drop the entire item.

## Lists & Collections

- **Identity key required.** Before creating a list item the text must supply at least one identity field:
  - `vehicles[N]`: VIN OR (year + make) OR (year + make + model)
  - `drivers[N]`: full_name OR license_number
  - `loss_history[N]`: occurrence_date OR description
  - `prior_insurance[N]`: carrier_name
- **No implicit items.** "We have 5 trucks" does NOT create 5 empty vehicle objects. Use `vehicle_count: 5`.
- **List preservation.** If nothing new belongs on an existing list, omit the key — returning `[]` would wipe prior items.

## Cross-Field Contamination Prevention

Fields must NEVER leak values between unrelated sections:

- `effective_date` is the NEW policy start date. NEVER copy it to `business_start_date`.
- `business_start_date` comes ONLY from "started in YYYY", "founded YYYY", "since YYYY".
- Prior carrier's `expiration_date` goes ONLY in `prior_insurance.expiration_date`, NOT in `policy_dates.expiration_date`.
- `annual_revenue` and `annual_payroll` are different fields.
- Lienholder address is NEVER the vehicle garaging address or the business mailing address.

## Prior Insurance

When user mentions "renewing from X", "current carrier is X", "switching from X":
- Create a `prior_insurance` entry with `carrier_name`
- If premium mentioned: `prior_insurance.premium`
- Set `prior_policy_status: "renewal"` or `"rewrite"` accordingly

## Loss History

- "No claims" / "clean record" = do NOT create a loss_history entry. Leave the array empty.
- Only create entries when the user describes actual incidents with details.
- "2 years ago" in loss context = store as description context, do NOT fabricate a specific date.

---
"""

V3_HARNESS_LIGHT = """
# Extraction Rules (focused)

The following rules address the most common extraction errors. Apply them to every turn.

## 1. Negation — always explicit

- "no hazmat" → `hazmat: false`
- "no dash cams" → `dash_cam: false`
- "no GPS/telematics" → `telematics: false`
- "no driver training" → `driver_training: false`
- "no hired auto" → `hired_auto: false`
- "no non-owned auto" → `non_owned_auto: false`
- "No hired or non-owned auto" → BOTH `hired_auto: false` AND `non_owned_auto: false`
- Silence ≠ false. Only set `false` when explicitly denied. When not mentioned, omit the field.

## 2. EIN / FEIN format

- Format: XX-XXXXXXX (hyphen after 2 digits). Store as exactly 9 digits without the hyphen in `ein`.
- "12-3456789" → `ein: "123456789"` (strip hyphen, validate 9 digits)
- Do NOT fabricate or pad. If fewer than 9 digits given, store verbatim.

## 3. Date extraction rules

- NEVER fabricate dates. "2 years ago", "next month", "end of this month" → OMIT the date field entirely.
- "started in 2018" → `business_start_date: "2018"` (year only, no month/day)
- "policy starts July 1" → `effective_date: "07/01"` (omit year if not stated)
- NEVER copy `effective_date` to `business_start_date` or vice versa.
- NEVER auto-calculate expiration date from effective date.

## 4. VIN normalization

- VIN is exactly 17 uppercase alphanumeric characters (no I, O, Q).
- Strip spaces and dashes before storing: "1FT-FW1E 42NEC25162" → `"1FTFW1E42NEC25162"`
- Do NOT invent, guess, or pad VINs that are shorter than 17 characters.

## 5. Phone normalization

- Strip ALL formatting: parentheses, dashes, spaces, dots.
- Store as 10 digits only: "(555) 867-5309" → `"5558675309"`
- Area code is required. Do NOT store 7-digit numbers.

---
"""
