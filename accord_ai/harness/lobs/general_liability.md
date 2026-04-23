# General Liability Extraction Harness v1.0

LOB-specific principles for General Liability extraction. Appended to core harness when GL is detected.

## Operations Description

- `operations_description` is a free-text summary of what the business DOES, not a list of products/services.
- Extract full sentences describing the business activity: "commercial construction general contractor doing ground-up buildings"
- Do NOT extract financial data or employee counts into operations_description

## Products vs Services Distinction

- `products_sold: true` ONLY when user mentions selling physical goods
- `products_manufactured: true` ONLY when user mentions manufacturing (much higher risk than retail/distribution)
- `services_provided` = free-text description of services offered
- "We build houses" → products_manufactured: true (construction = manufacturing for GL purposes)
- "We sell clothing" → products_sold: true, products_manufactured: false
- "We repair computers" → services_provided, NOT products fields

## Subcontractor Usage

- `subcontractor_usage: true` when user mentions using subs, hiring contractors, outsourcing specialties
- "We do electrical work ourselves and sub out plumbing" = subcontractor_usage: true
- Track subcontractor type in the operations description

## Premises Percentage

- `work_on_premises_pct` is the percentage of work done AT the customer's location (not the insured's own location)
- "We work on client sites 80% of the time" → `work_on_premises_pct: "80"`
- If not stated, OMIT (do not guess)

## Class Codes

- GL class codes are different from NAICS. Do NOT conflate them.
- `classification.class_code` = ISO GL class code (5-digit)
- Premium basis: "revenue" (sales), "payroll", "area" (sqft), or "units"
- Construction/contractors usually use payroll basis
- Retail/hospitality usually use revenue basis

## Coverage Limits

- Each Occurrence vs General Aggregate: different fields, usually ratio is 1:2 (e.g., $1M/$2M)
- Products-Completed Ops Aggregate is separate from General Aggregate
- Personal & Advertising Injury is a SEPARATE limit from bodily injury
- Damage to Rented Premises is a SEPARATE limit, usually much smaller (e.g., $100K)
- Medical Expense limit is per-person, usually $5K-$10K

## Financial Basis

- `annual_gross_receipts` = total revenue from all sources (GL rating basis)
- NEVER confuse with `annual_payroll` — these are different fields
- "Revenue $2.5M, payroll $800K" → annual_gross_receipts: "2500000", annual_payroll: "800000"

## Contractual Requirements

- `additional_insured: true` when user mentions needing to name others as additional insured
- `certificate_holder` = free-text name when user mentions who needs certificates
- `contractual_requirements` = free-text summary of contract-mandated insurance terms

## Safety Programs

- `safety_program: true` ONLY when user mentions a FORMAL program (written, documented, OSHA-compliant)
- Informal "we try to be safe" is NOT a formal program → false
- `quality_control: true` when user mentions QC procedures, inspections, testing
