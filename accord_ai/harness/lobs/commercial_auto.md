# Extraction Harness v3.0 (curated)

LOB-specific principles for Commercial Auto extraction. Appended to core harness when Commercial Auto is detected.

## Vehicle Count vs Employee Count

- Numbers near "truck", "vehicle", "fleet", "rig", "van" → `operations.vehicle_count`
- Numbers near "employee", "staff", "worker", "of us" → `named_insured.employees.total`
- "just me and my brother" → 2 people (colloquial counting)
- Ambiguous number with no context → omit
- **DO NOT create a `trailer_count` field.** Trailers are implicitly handled by `operations.vehicle_count` and `vehicles[N].body_type`.

## Use Type Classification

- "delivery" / "last-mile" / "courier" → `operations.fleet_use.use_type: "delivery"`
- "long haul" / "cross-country" / "interstate" → `"long_haul"`
- "local service" / "customer sites" → `"service"`
- "haul other people's freight" → `"for_hire"` + `operations.fleet_use.for_hire: true`
- "move our own stuff" → `"commercial"` + `for_hire: false`
- "mostly regional" → `operations.fleet_use.use_type: "regional"` (This is a distinct enum value from "local", "intermediate", and "long_haul").

## Hazmat & Trailer Interchange

- `operations.hazmat.hazmat: true` ONLY when user explicitly says "hazmat", "hazardous materials", "dangerous goods".
- "We haul fuel" alone is NOT enough — user must explicitly acknowledge hazmat.
- `operations.hazmat.hazmat: false` should be set when user explicitly states "No hazmat" or similar negations.
- `operations.trailer_interchange: true` ONLY when user says "interchange agreement".
- "We have trailers" ≠ trailer interchange.
- Explicit negation of trailer interchange (e.g., "No trailer interchange") should set `operations.trailer_interchange: false`.

## Hired vs Non-Owned Auto

- `operations.hired_non_owned.hired.hired_auto: true` = business RENTS or BORROWS vehicles.
- `operations.hired_non_owned.non_owned.non_owned_auto: true` = EMPLOYEES use PERSONAL vehicles for business.
- Explicit negations like "No hired autos" or "No non-owned" MUST be captured by setting the corresponding boolean to `false`.
- "No hired" alone → set `operations.hired_non_owned.hired.hired_auto: false`, do NOT infer `non_owned`.

## Vehicle Entries

- Create an entry for EACH vehicle described, regardless of the total number. Do not impose an arbitrary limit on the number of `vehicles` entries.
- Create an entry ONLY with specific year/make/model. Never partial.
- VIN is 17 characters. Do NOT invent VINs.
- `vehicles[N].garaging_address` is where vehicle is parked overnight. Never a PO Box.
- Do NOT fabricate `vehicles[N].cost_new`, `vehicles[N].annual_mileage`, or `vehicles[N].gvw`.

## Driver Entries

- Create entry ONLY when text supplies full name OR license number.
- "We have 5 drivers" alone → `operations.driver_count: 5`, NO driver entries.
- Clean-record → `drivers[N].mvr_status: "clean"`, omit violations array.
- Do NOT assign policy `effective_date` to driver `hire_date`.

## Coverages

- "$1M CSL" → `coverages.liability.limit_type: "csl"`, `coverages.liability.csl_limit: 1000000`
- Split "100/300/100" → `coverages.liability.limit_type: "split"`, `coverages.liability.bi_per_person: 100000`, `coverages.liability.bi_per_accident: 300000`, `coverages.liability.pd_per_accident: 100000`
- Physical damage: `coverages.physical_damage.comprehensive: true` + `coverages.physical_damage.comprehensive_deductible` when comp requested; same for collision.
- Single deductible for both → set BOTH `coverages.physical_damage.comprehensive_deductible` AND `coverages.physical_damage.collision_deductible`.
- Medical payments, towing, rental are FLAT SCALARS: `coverages.med_pay_limit`, `coverages.towing_limit`, `coverages.rental_reimbursement_limit`.
- UM/UIM: `coverages.um_uim.bi_per_person`, `coverages.um_uim.bi_per_accident`.
- General Liability terms ("per occurrence", "aggregate") belong to a different LOB — do NOT map to auto coverage fields.

## Loss History

- `loss_type` enum: `auto_accident`, `cargo_damage`, `collision`, `comprehensive`, `property_damage`, `bodily_injury`, `theft`, `vandalism`, `weather`, `fire`, `other`. Always emit alongside free-text `description`.
- "No claims" / "clean record" → emit nothing for `loss_history`.

## Operations Territory & Radius

- `operations.territory.states_of_operation` should be populated with 2-letter USPS state codes mentioned by the user.
- `operations.territory.radius_of_operations` is an ENUM with values: `local`, `intermediate`, `long_haul`.
- Infer `radius_of_operations` based on stated radius or common understanding:
    - `local`: Typically up to 250 miles.
    - `intermediate`: Typically 250-500 miles.
    - `long_haul`: Typically over 500 miles.
- When a mileage is provided, map it to the closest enum value. For "About 400 mile radius", this falls into the `intermediate` range.
- **DO NOT** create a `radius_miles` field. If a specific mileage is given but does not fit neatly into the enum categories, prioritize the descriptive term (e.g., "regional") or omit the radius if no clear mapping exists. If the user provides both a descriptive term and a mileage that conflict, use the descriptive term.
- When both a descriptive term (e.g., "regional") and a mileage (e.g., "400 miles") are provided, and the mileage clearly maps to an enum value, use the enum value derived from the mileage. For example, "regional" and "400 miles" should both result in `operations.territory.radius_of_operations: "intermediate"`.

## Safety Features

- `operations.safety.dash_cam: true` when user mentions dash cameras or "dash cam".
- `operations.safety.telematics: true` when user mentions telematics, onboard diagnostics, or similar data-logging systems.
- `operations.safety.driver_training: true` when user mentions formal driver training programs.
- `operations.safety.gps_tracking: true` when user explicitly mentions GPS tracking for vehicles. This is distinct from telematics, which may or may not include GPS.

## Common refiner mistakes to avoid

1. **DO NOT write** `coverages.physical_damage.cargo` or
   `coverages.cargo.*` — cargo coverage is not in the v3 extraction
   schema. There is nowhere for a `cargo_limit` to land; do not write
   principles that reference cargo coverage paths.

2. **DO NOT write** `coverages[0].limit` — coverages is a DICT, not a
   LIST. The correct paths are `coverages.liability.csl_limit`,
   `coverages.physical_damage.comprehensive_deductible`,
   `coverages.med_pay_limit` (flat int), `coverages.towing_limit` (flat int),
   `coverages.um_uim.bi_per_accident`, etc.

3. **DO NOT write** `prior_insurance[N].carrier` — the field is
   `carrier_name`, not `carrier`.

4. **DO NOT write** `business.*` — the top-level is
   `named_insured.*`, not `business.*`. `business.contact_name` is
   NOT a valid path; use `named_insured.contact.full_name`.

5. **DO NOT write** `auto_info.*` — use `operations.*` instead.
   `auto_info.hazmat` does not exist;
   `operations.hazmat.hazmat` does.

6. **DO NOT write** new top-level keys that aren't listed above. If a
   user mentions something (cargo haul types, fleet details, safety
   programs) and no structured field exists, the best option is to
   OMIT the rule rather than invent a path that the converter will
   silently drop.

7. **DO NOT write** `coverages.general_liability.limit` — general
   liability is a different LOB. Auto liability is
   `coverages.liability.csl_limit`.

Any principle that references a path not in this schema is likely to
make things worse, not better. If you're unsure whether a path exists,
OMIT the rule rather than guess.