# Extraction Harness — Core Principles v6.1 (curated)

Universal rules applied to every extraction, regardless of LOB. LOB-specific principles live in separate files appended based on active LOBs.

## 1. Source Fidelity

Extract only what is explicitly in the current user message. Never infer, assume, default, or pull from prior state.

- **Omit, never default.** If a value is not in the text, omit the field — not `null`, `""`, `0`, or a placeholder.
- **Never fabricate** DOBs, license numbers, phones, dates, EINs, VINs, or dollar amounts.
- **Corrections override.** "Actually 2022" after "2021" stores `2022`.
- **List items must be verbatim.** Every attribute of a list item must come from the text; otherwise drop the entire item.

## 2. Entity & Attribute Routing

Every value belongs to exactly ONE field. Route by semantic context and the entity it modifies.

- **Semantic isolation.** A name under "Contact" is a contact, not a driver. "Years in business" is not a claim count. A number under "employees" is not a vehicle count.
- **Immediate-context binding.** A numerical or descriptive attribute binds to the coverage or entity mentioned in its same sentence. A deductible with no coverage context attaches to the most recently mentioned coverage or is omitted.
- **Component granularity.** Sub-parts of a limit ("per person", "per accident") each go into their own schema key — never a single combined field.
- **Per-vehicle attribution.** Attributes stated alongside a specific vehicle belong on that `vehicles[N]` object, not on a fleet-wide field.
- **Contact role vs. job title.** If a name is provided for a contact, look for explicit job titles (e.g., "CEO", "President", "Agent"). Do not infer or use generic terms like "role" if not present in the schema. If only a name is given, use `named_insured.contact.full_name` and omit `job_title`.

## 3. Lists & Collections

- **Identity key required.** Before creating a list item the text must supply at least one identity field:
  - `vehicles[N]`: VIN OR (year + make) OR (year + make + model)
  - `drivers[N]`: full_name OR license_number
  - `loss_history[N]`: occurrence_date OR description
  - `prior_insurance[N]`: carrier_name
- **No implicit items.** "We have 5 trucks" does NOT create 5 empty vehicle objects.
- **List preservation.** If nothing new belongs on an existing list, omit the key — returning `[]` would wipe prior items.
- **State codes are arrays.** "NE IA MO KS CO" → `["NE","IA","MO","KS","CO"]`.

## 4. Negation & Exclusion

| User said | Emit |
|---|---|
| Silence (topic never raised) | Omit the field |
| Explicit negation ("no X", "without X", "don't have X") | `false` |
| Explicit affirmation ("yes X", "we have X") | `true` |

- **Silence ≠ false.** `false` means the topic was discussed and denied; omission means it was never raised.
- **Scope correctly.** "No claims" negates `loss_history`, not the entire policy. "No hazmat" negates `operations.hazmat.hazmat`.
- **Combined negations split.** "No hired or non-owned auto" → `false` on BOTH `operations.hired_non_owned.hired.hired_auto` AND `operations.hired_non_owned.non_owned.non_owned_auto`.
- **Common boolean targets and their paths:**
  - `operations.hazmat.hazmat` — "no hazmat", "no dangerous goods"
  - `operations.safety.dash_cam` — "no dash cameras"
  - `operations.safety.telematics` — "no GPS tracking", "no telematics"
  - `operations.safety.driver_training` — "no formal training program"
  - `operations.trailer_interchange` — "no interchange agreement"
  - `operations.hired_non_owned.hired.hired_auto` — "don't rent vehicles"
  - `operations.hired_non_owned.non_owned.non_owned_auto` — "employees don't use personal cars"
  - `operations.fleet_use.for_hire` — "we don't haul for others"

**Negation few-shot examples** (input → exact JSON output):

Example 1: "No hired auto, no non-owned, no hazmat"
```json
{"operations": {"hired_non_owned": {"hired": {"hired_auto": false}, "non_owned": {"non_owned_auto": false}}, "hazmat": {"hazmat": false}}}
```

Example 2: "We don't carry hazmat, no GPS tracking, but we do have dash cams and driver training"
```json
{"operations": {"hazmat": {"hazmat": false}, "safety": {"telematics": false, "dash_cam": true, "driver_training": true}}}
```

Example 3: "No trailer interchange, we don't haul for others"
```json
{"operations": {"trailer_interchange": false, "fleet_use": {"for_hire": false}}}
```

## 5. Types & Formats

- Integers for counts. Booleans for yes/no. Strings for names/IDs. Dates as `MM/DD/YYYY`.
- Two-digit years → 4 digits by context. Policy dates → future ("26"→"2026"). DOBs → past ("78"→"1978").
- Currency → integer dollars, no `$` or commas. "$1.2M"→`1200000`.
- Phone → `(XXX) XXX-XXXX`. EIN → `XX-XXXXXXX`. State codes → 2-letter USPS.
- Entity type → only emit when a legal form is explicitly stated (LLC, Inc, Corp, Partnership, Sole Proprietor, S-Corp).

## 6. Insurance Products vs Business Nature

- **Requests** ("I need GL", "Commercial Auto for…") → coverage fields.
- **Descriptions** ("We're a plumbing contractor") → `named_insured.nature_of_business`.
- **Never** put an insurance product name into `nature_of_business` or `business_name`.

## 7. Schema Adherence

- **Strict path adherence.** Always use the exact field paths provided in the v3 Schema Reference. Do not invent new paths, nest fields incorrectly, or use deprecated/incorrect names.
- **Avoid invalid fields.** If the user provides information that does not map to a valid schema path (e.g., "contact role"), omit the field. Do not attempt to map it to an incorrect field or create a new one.
- **Correct field names.** Pay close attention to field names like `carrier_name` (not `carrier`) and `full_name` (not `contact_name` or `role` for a person's name).
- **Dict vs. List.** Remember that `coverages` is a dictionary keyed by coverage type, not a list. Access limits and deductibles using paths like `coverages.liability.csl_limit` or `coverages.physical_damage.collision_deductible`, not `coverages[N]`.
- **No Cargo Coverage.** The v3 schema does not include fields for cargo coverage. Do not attempt to extract or store cargo-related limits or details. Omit these if mentioned.