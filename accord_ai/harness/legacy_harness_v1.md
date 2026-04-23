# Extraction Harness v1.0

This document is the extraction model's living constitution. It is injected into every extraction prompt. The Judge and Refiner models update it automatically when extraction errors are detected.

**Rules:** This document must stay under 2000 tokens. Sections are REFINED, never just appended. When a new principle subsumes an older one, the older one is removed.

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
- "we do NOT carry hazmat" = `hazmat: false` (NOT modifies "carry hazmat")
- "not true that we carry hazmat" = `hazmat: false` (double negation = false)
- "used to not have GPS, now we do" = `telematics: true` (temporal: "now" wins)
- "cannot say we don't have driver training" = `driver_training: true` (double negative = positive)

Never infer a boolean from context. "We haul fuel" does NOT imply `hazmat: true` unless the user explicitly mentions hazmat/hazardous/dangerous goods.

## Correction Recognition

When a user corrects a previous value, extract ONLY the corrected field.

- "actually it's 7800 not 7700" = correction to an address/number field. Extract the NEW value (7800).
- "we're an S-Corp not an LLC" = correction to `entity_type`. Extract `subchapter_s`. Do NOT touch `business_name`.
- "wait, 12 trucks not 10" = correction to `vehicle_count`. Extract 12.
- The word "actually", "wait", "correction", "wrong", "not X but Y", "should be" all signal corrections.
- On a correction turn: extract ONLY changed fields. Do NOT re-extract unchanged data.

## Entity Type Resolution

Priority order (highest wins):
1. Legal suffix in business name: "Inc"/"Corp" = corporation, "LLC" = llc, "LP" = partnership, "LLP" = limited_liability_partnership
2. Explicit user statement: "we're a corporation", "S-Corp"
3. NEVER infer from business description. "sole owner" does not mean entity_type = "individual" unless user explicitly says "sole proprietorship."

"We are an S-Corp" means entity_type = "subchapter_s". It does NOT mean business_name = "an S-Corp". The phrase describes the legal structure, not the company name.

## Address Parsing

- "Suite", "Ste", "Unit", "Apt", "#" should go into `line_two`, not appended to `line_one`.
- Compound city names: "Kansas City" is ONE city (not Kansas + City). State comes from context, not the city name. Kansas City MO and Kansas City KS are different cities.
- "Same address" for garaging = copy from the FIRST vehicle's garaging address or business mailing address.
- PO Box = lienholder or mailing only. NEVER a garaging address.

## Numeric Disambiguation

When multiple numbers appear in one message, each belongs to a specific field based on surrounding words:

- XX-XXXXXXX (with hyphen, 9 digits) = `tax_id` / FEIN
- 5-6 digits alone = could be ZIP or NAICS. "ZIP" or address context = `zip_code`. "NAICS" or "code" = `naics`.
- 10 digits or (XXX) XXX-XXXX = `phone`
- 4-digit year = `business_start_date` or vehicle `year` depending on context
- Dollar amounts ($X,XXX or Xk or X million) = route by context keyword

## Temporal Context

- "2 years ago" = DO NOT fabricate a specific date. Extract "approximately 2 years ago" as description, leave date blank.
- "started in 2018" = `business_start_date: "2018"` (year only, do not add month/day)
- "policy starts July 1" = `effective_date: "07/01/YYYY"` with current or next year
- "expires next March" = approximate, extract the month but be explicit about uncertainty
- NEVER auto-calculate expiration from effective date

## Prior Insurance

When user mentions "renewing from X", "current carrier is X", "switching from X":
- Create a `prior_insurance` entry with `carrier_name`
- If premium mentioned: `prior_insurance.premium`
- If expiration mentioned: `prior_insurance.expiration_date`
- Set `policy.status: "renewal"` or `"rewrite"` accordingly
- The PRIOR carrier's expiration is NOT the new policy's expiration_date

## Loss History

- "No claims" / "clean record" = do NOT create a loss_history entry. Leave the array empty.
- Only create entries when the user describes actual incidents with details.
- Each entry needs: description (required), date or timeframe, amount if stated, status (open/closed).
- "2 years ago" in loss context = store as description context, do NOT fabricate "01/01/2024".

## Cross-Field Contamination Prevention

Fields must NEVER leak values between unrelated sections:

- `effective_date` is the NEW policy start date. NEVER copy it to `business_start_date`.
- `business_start_date` comes ONLY from "started in YYYY", "founded YYYY", "since YYYY". It is NEVER the same as effective_date unless user explicitly says so.
- Prior carrier's `expiration_date` goes ONLY in `prior_insurance.expiration_date`, NOT in `policy.expiration_date`.
- `annual_revenue` and `annual_payroll` are different fields. "$2.5M revenue, $800K payroll" = two separate values.
- Lienholder address is NEVER the vehicle garaging address or the business mailing address.

## Relative & Vague Dates

The model has NO knowledge of the current date. Handle accordingly:

- "Next month", "end of this month", "in two weeks" → OMIT the date entirely. Do NOT fabricate.
- "Started 8 years ago" → OMIT specific date. Extract only if user says a year: "started in 2018".
- "Policy starts July 1" → extract as "07/01" but do NOT guess the year unless user specifies.
- "2 years ago" for loss history → put in description as context, leave occurrence_date blank.
- NEVER output a date in the past for effective_date unless user explicitly provides a past date.

---

*Last updated: v1.1 — added cross-contamination prevention and relative date handling from Round 4 extreme testing (1,130 checks, 91.6% cumulative).*
