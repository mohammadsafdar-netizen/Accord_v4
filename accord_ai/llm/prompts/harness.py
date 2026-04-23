"""Harness principles — ported from v3's living harness document
(``accord_ai_v3/harness/harness.md``, v1.1, 1130-check coverage).

v3's harness is the distilled output of a 12-month Judge → Refiner loop:
every repeated correction became a rule. Without these rules, a fresh
extractor prompt re-learns the same failure modes scenario by scenario.
We port the wisdom verbatim, translated for v4's schema:

  v3 path                       →  v4 path
  ──────────────────────────────────────────────────────────────
  business.tax_id               →  ein
  business.employees.total      →  full_time_employees
  business.employee_count       →  full_time_employees
  business.naics                →  naics_code
  business.contact_name / phone →  contacts[0].full_name / phone
  policy.effective_date         →  policy_dates.effective_date
  policy.expiration_date        →  policy_dates.expiration_date
  policy.status                 →  policy_status
  vehicles[N].garaging_address  →  lob_details.vehicles[N].garage_address
  vehicles[N].*                 →  lob_details.vehicles[N].*
  drivers[N].*                  →  lob_details.drivers[N].*
  auto_info.hazmat              →  lob_details.hazmat
  auto_info.use_type            →  lob_details.vehicles[0].use_type
  prior_insurance.premium       →  prior_insurance[N].premium_amount

Field-schema notes:
  * ``entity_type`` Literal values in v4: corporation, partnership, llc,
    individual, subchapter_s, joint_venture, not_for_profit, trust.
    'sole_proprietorship' is NOT valid — use 'individual'.
    'limited_liability_partnership' is NOT in v4 — use 'partnership'.
  * ``dash_cam`` / ``telematics`` do not exist in v4's CA schema; the
    judge doesn't ask for them. Omit if mentioned — the responder will
    not re-ask for what the schema can't hold.

Stays under ~2000 tokens when rendered. Shared between extractor and
refiner prompts so both get the same wisdom without duplication.
"""
from __future__ import annotations


HARNESS_RULES = (
    "## HARNESS — EXTRACTION WISDOM\n"
    "(Ported from v3's living-harness document — 12 months of refiner "
    "corrections distilled into rules.)\n"
    "\n"
    "### Field Routing\n"
    "Every value belongs to exactly ONE field. Route by context, not by value format alone.\n"
    "\n"
    "- Numbers near 'truck'/'vehicle'/'fleet'/'rig' → `lob_details.vehicle_count`. "
    "Numbers near 'employee'/'staff'/'worker'/'people'/'of us' → `full_time_employees` "
    "(unless user distinguishes part-time; then split between "
    "`full_time_employees` and `part_time_employees`).\n"
    "- A number with NO context → OMIT. Do not extract. Let the responder re-ask.\n"
    "- Dollar amounts: near 'premium' → `prior_insurance[N].premium_amount`. "
    "Near 'revenue'/'gross'/'sales' → `annual_revenue`. "
    "Near 'payroll' → `annual_payroll`. "
    "Near 'value'+vehicle context → `lob_details.vehicles[N].cost_new` or `stated_amount`.\n"
    "- A lone address → `mailing_address` unless the turn is in the vehicle/fleet "
    "context (then → `lob_details.vehicles[N].garage_address`) or the lienholder "
    "context (then → `additional_interests[N].address`). A PO Box is NEVER a "
    "garage address.\n"
    "\n"
    "### Negation & Qualifiers\n"
    "The negation word modifies the CLOSEST boolean concept.\n"
    "\n"
    "- 'no hazmat' → `lob_details.hazmat: false`\n"
    "- 'we do NOT carry hazmat' → `lob_details.hazmat: false`\n"
    "- 'not true that we carry hazmat' → `lob_details.hazmat: false` (double-neg = false)\n"
    "- 'cannot say we don't have driver training' → `lob_details.driver_training: true` "
    "(double negative = positive)\n"
    "- 'used to not have GPS, now we do' → temporal; 'now' wins. Omit fields v4's "
    "schema doesn't hold (dash_cam, telematics are not in CommercialAutoDetails).\n"
    "- NEVER infer a boolean from context. 'We haul fuel' does NOT imply "
    "`hazmat: true` unless the user explicitly mentions hazmat/hazardous/dangerous.\n"
    "\n"
    "### Correction Recognition\n"
    "When a user corrects a previous value, extract ONLY the corrected field.\n"
    "\n"
    "- Signals: 'actually', 'wait', 'correction', 'wrong', 'not X but Y', "
    "'should be'.\n"
    "- 'actually it's 7800 not 7700' → corrected address/number field. Extract 7800 only.\n"
    "- 'we're an S-Corp not an LLC' → `entity_type: subchapter_s`. Do NOT touch business_name.\n"
    "- 'wait, 12 trucks not 10' → `lob_details.vehicle_count: 12`. Do NOT re-extract drivers.\n"
    "- On a correction turn: extract ONLY the changed fields. Leave unchanged data absent from the diff.\n"
    "\n"
    "### Entity Type Resolution\n"
    "Priority (highest wins):\n"
    "1. Legal suffix in business name: 'Inc'/'Corp' → corporation; 'LLC' → llc; "
    "'LP'/'LLP' → partnership (v4 has no limited_liability_partnership — use partnership).\n"
    "2. Explicit user statement: 'we're a corporation', 'S-Corp' → subchapter_s.\n"
    "3. 'sole proprietorship' or 'sole owner' → individual (NOT sole_proprietorship — "
    "that value is NOT in the Literal enum).\n"
    "4. NEVER infer from business description alone. 'sole owner' without the word "
    "'proprietor' does not set entity_type.\n"
    "\n"
    "'We are an S-Corp' means `entity_type: subchapter_s`. It does NOT mean "
    "`business_name: 'an S-Corp'`. The phrase describes legal structure, not company name.\n"
    "\n"
    "### Address Parsing\n"
    "- 'Suite'/'Ste'/'Unit'/'Apt'/'#' → `line_two`, NOT appended to `line_one`.\n"
    "- Compound city names: 'Kansas City' is ONE city. State comes from context. "
    "'Kansas City MO' and 'Kansas City KS' are different cities.\n"
    "- 'Same address' for garaging → copy from the FIRST vehicle's garage_address "
    "or business mailing_address.\n"
    "- PO Box → additional_interests (lienholder) or mailing_address only. NEVER a "
    "vehicle garage_address.\n"
    "\n"
    "### Numeric Disambiguation\n"
    "Multiple numbers in one message — route by surrounding words:\n"
    "\n"
    "- XX-XXXXXXX (9 digits, one hyphen) → `ein`\n"
    "- 5-6 digits alone: 'ZIP' or address context → `mailing_address.zip_code`. "
    "'NAICS' or 'code' → `naics_code`. 'SIC' → `sic_code`.\n"
    "- 10 digits or (XXX) XXX-XXXX → `phone` (root) OR `contacts[0].phone`.\n"
    "- 4-digit year: business founded context → `business_start_date`; "
    "vehicle context → `lob_details.vehicles[N].year`.\n"
    "- Dollar amounts — route by the context keyword, not the amount alone.\n"
    "\n"
    "### Temporal & Relative Dates\n"
    "The model has NO knowledge of the current date. Handle accordingly:\n"
    "\n"
    "- 'Next month'/'end of this month'/'in two weeks' → OMIT the date. Do NOT fabricate.\n"
    "- 'Started 8 years ago' → OMIT `business_start_date`. Extract only if user gives a year: "
    "'started in 2018' → `business_start_date: '2018-01-01'`.\n"
    "- 'Policy starts July 1' → `policy_dates.effective_date: 'YYYY-07-01'` ONLY if the "
    "user specifies the year. Otherwise OMIT.\n"
    "- '2 years ago' for a loss → put in `loss_history[N].description` as context; "
    "leave `date_of_loss` blank.\n"
    "- NEVER auto-calculate `policy_dates.expiration_date` from effective_date.\n"
    "- NEVER output a past date for `policy_dates.effective_date` unless the user explicitly provides a past date.\n"
    "\n"
    "### Prior Insurance\n"
    "When user mentions 'renewing from X', 'current carrier is X', 'switching from X':\n"
    "\n"
    "- Create a `prior_insurance[N]` entry with `carrier_name`.\n"
    "- If premium mentioned → `prior_insurance[N].premium_amount`.\n"
    "- If expiration mentioned → `prior_insurance[N].expiration_date`.\n"
    "- Set `policy_status: 'renewal'` or `'rewrite'` accordingly.\n"
    "- The PRIOR carrier's expiration is NOT the new `policy_dates.expiration_date`.\n"
    "\n"
    "### Loss History\n"
    "- 'No claims' / 'clean record' / 'no losses' → LEAVE loss_history EMPTY. Do not create entries.\n"
    "- Only create entries when the user describes actual incidents with details.\n"
    "- Each entry needs: `description` (required), `date_of_loss` or timeframe, "
    "`amount_paid` if stated, `claim_status` (open/closed).\n"
    "- '2 years ago' → description context only; leave `date_of_loss` blank.\n"
    "\n"
    "### Cross-Field Contamination Prevention\n"
    "Fields must NEVER leak values between unrelated sections:\n"
    "\n"
    "- `policy_dates.effective_date` is the NEW policy start. NEVER copy to `business_start_date`.\n"
    "- `business_start_date` comes ONLY from 'started in YYYY'/'founded YYYY'/'since YYYY'. "
    "It is NEVER the same as effective_date unless user explicitly says so.\n"
    "- Prior carrier's expiration_date → ONLY `prior_insurance[N].expiration_date`, NEVER "
    "`policy_dates.expiration_date`.\n"
    "- `annual_revenue` and `annual_payroll` are different fields. '$2.5M revenue, $800K payroll' = "
    "two distinct values.\n"
    "- Lienholder address → `additional_interests[N].address`, NEVER a vehicle's "
    "`garage_address` or the business `mailing_address`.\n"
    "- `lob_details.coverage.hired_auto` / `non_owned_auto` live on coverage, NEVER "
    "on the top level of `lob_details`.\n"
    "- `prior_insurance` lives at the submission root, NEVER inside `lob_details` "
    "(except for WorkersCompDetails which has its own `prior_insurance` list — the "
    "root array is preferred for non-WC lines).\n"
)
