"""9.d — Explainer. Pure function; tests assert on exact or substring output."""
from accord_ai.conversation.explainer import explain
from accord_ai.schema import CustomerSubmission


# --- Empty / minimal ---

def test_empty_submission_returns_sentinel():
    assert explain(CustomerSubmission()) == "(empty submission)"


def test_business_name_only():
    out = explain(CustomerSubmission(business_name="Acme"))
    assert out == "Business: Acme"


def test_dba_alone_renders_as_dba_prefix():
    out = explain(CustomerSubmission(dba="Acme Trading"))
    assert out == "DBA: Acme Trading"


def test_business_name_and_dba_combined():
    out = explain(CustomerSubmission(business_name="Acme Corp", dba="Acme"))
    assert "Business: Acme Corp (DBA: Acme)" in out


# --- Business info ---

def test_full_business_info_block():
    sub = CustomerSubmission(
        business_name="Acme Trucking",
        ein="12-3456789",
        email="ops@acme.com",
        phone="555-123-4567",
        business_address={"line_one": "123 Main St", "city": "Detroit",
                          "state": "MI", "zip_code": "48201"},
    )
    out = explain(sub)
    assert "Business: Acme Trucking" in out
    assert "Address: 123 Main St, Detroit, MI 48201" in out
    assert "EIN: 12-3456789" in out
    assert "Email: ops@acme.com" in out
    assert "Phone: 555-123-4567" in out


def test_mailing_address_separate_from_business():
    sub = CustomerSubmission(
        business_name="Acme",
        business_address={"city": "Detroit"},
        mailing_address={"city": "Ann Arbor"},
    )
    out = explain(sub)
    assert "Address: Detroit" in out
    assert "Mailing: Ann Arbor" in out


# --- Policy dates ---

def test_policy_dates_rendered():
    sub = CustomerSubmission(
        business_name="Acme",
        policy_dates={"effective_date": "2026-05-01", "expiration_date": "2027-05-01"},
    )
    assert "Policy: 2026-05-01 to 2027-05-01" in explain(sub)


def test_partial_policy_dates_shows_not_set():
    sub = CustomerSubmission(
        business_name="Acme",
        policy_dates={"effective_date": "2026-05-01"},
    )
    assert "Policy: 2026-05-01 to (not set)" in explain(sub)


# --- Commercial Auto ---

def test_commercial_auto_section_header():
    sub = CustomerSubmission(
        business_name="Acme", lob_details={"lob": "commercial_auto"},
    )
    assert "Line of Business: Commercial Auto" in explain(sub)


def test_commercial_auto_drivers_with_full_detail():
    sub = CustomerSubmission(
        business_name="Acme",
        lob_details={"lob": "commercial_auto", "drivers": [
            {"first_name": "Alice", "last_name": "Nguyen",
             "date_of_birth": "1985-03-21",
             "license_number": "ABC123", "license_state": "MI",
             "years_experience": 10},
            {"first_name": "Bob"},
        ]},
    )
    out = explain(sub)
    assert "Drivers (2):" in out
    assert "Alice Nguyen" in out
    assert "DOB 1985-03-21" in out
    assert "License MI-ABC123" in out
    assert "10y exp" in out
    assert "Bob" in out


def test_commercial_auto_vehicles_with_vin_and_garage():
    sub = CustomerSubmission(
        business_name="Acme",
        lob_details={"lob": "commercial_auto", "vehicles": [
            {"year": 2022, "make": "Ford", "model": "F-150",
             "vin": "1FTFW1E50NFA12345",
             "garage_address": {"city": "Warren", "state": "MI"}},
        ]},
    )
    out = explain(sub)
    assert "Vehicles (1):" in out
    assert "2022 Ford F-150" in out
    assert "VIN 1FTFW1E50NFA12345" in out
    assert "garaged at Warren, MI" in out


def test_commercial_auto_coverage_currency_formatting():
    sub = CustomerSubmission(
        business_name="Acme",
        lob_details={"lob": "commercial_auto", "coverage": {
            "liability_limit_csl": 1_000_000,
            "comp_deductible": 500,
            "coll_deductible": 1000,
            "hired_auto": True,
        }},
    )
    out = explain(sub)
    assert "Liability CSL: $1,000,000" in out
    assert "Comp deductible: $500" in out
    assert "Coll deductible: $1,000" in out
    assert "Hired auto: yes" in out


def test_commercial_auto_hazmat_and_radius():
    sub = CustomerSubmission(
        business_name="Acme",
        lob_details={"lob": "commercial_auto",
                     "radius_of_operations": "long_haul", "hazmat": True},
    )
    out = explain(sub)
    assert "Radius of operations: long_haul" in out
    assert "Hazmat: yes" in out


def test_driver_count_declared_without_drivers_list():
    sub = CustomerSubmission(
        business_name="Acme",
        lob_details={"lob": "commercial_auto", "driver_count": 5},
    )
    out = explain(sub)
    assert "Drivers declared: 5" in out


# --- General Liability ---

def test_general_liability_with_classifications():
    sub = CustomerSubmission(
        business_name="Acme",
        lob_details={"lob": "general_liability", "employee_count": 25,
                     "classifications": [
                         {"class_code": "91580", "naics_code": "484110",
                          "annual_gross_receipts": 2_500_000},
                     ]},
    )
    out = explain(sub)
    assert "Line of Business: General Liability" in out
    assert "Employees: 25" in out
    assert "Classifications (1):" in out
    assert "class 91580" in out
    assert "NAICS 484110" in out
    assert "receipts $2,500,000" in out


def test_general_liability_coverage_claims_made_basis():
    sub = CustomerSubmission(
        business_name="Acme",
        lob_details={"lob": "general_liability", "coverage": {
            "each_occurrence_limit": 1_000_000,
            "general_aggregate_limit": 2_000_000,
            "claims_made_basis": True,
        }},
    )
    out = explain(sub)
    assert "Each occurrence: $1,000,000" in out
    assert "General aggregate: $2,000,000" in out
    assert "Basis: claims-made" in out


# --- Workers Comp ---

def test_workers_comp_full_shape():
    sub = CustomerSubmission(
        business_name="Acme",
        lob_details={"lob": "workers_comp",
                     "experience_mod": 0.95,
                     "owner_exclusion": True,
                     "waiver_of_subrogation": False,
                     "payroll_by_class": [
                         {"class_code": "8810", "description": "Clerical",
                          "payroll": 500_000, "employee_count": 5, "state": "MI"},
                     ],
                     "coverage": {
                         "employers_liability_per_accident": 1_000_000,
                     }},
    )
    out = explain(sub)
    assert "Line of Business: Workers Compensation" in out
    assert "Experience mod: 0.95" in out
    assert "Owner exclusion: yes" in out
    assert "Waiver of subrogation: no" in out
    assert "class 8810 Clerical (MI) — $500,000, 5 empl" in out
    assert "Per accident: $1,000,000" in out


# --- Additional interests + loss history ---

def test_additional_interests_with_addresses():
    sub = CustomerSubmission(
        business_name="Acme",
        additional_interests=[
            {"name": "Ally Bank", "role": "lienholder",
             "address": {"city": "Detroit", "state": "MI"}},
            {"name": "Landlord LLC", "role": "additional_insured"},
        ],
    )
    out = explain(sub)
    assert "Additional Interests (2):" in out
    assert "Ally Bank (lienholder) at Detroit, MI" in out
    assert "Landlord LLC (additional_insured)" in out


def test_loss_history_entry_with_all_fields():
    sub = CustomerSubmission(
        business_name="Acme",
        loss_history=[
            {"date_of_loss": "2024-03-15", "type_of_loss": "collision",
             "amount_paid": 8500, "claim_status": "closed",
             "description": "rear-ended at intersection"},
        ],
    )
    out = explain(sub)
    assert "Loss History (1):" in out
    assert "2024-03-15: collision, paid $8,500, closed" in out
    assert "rear-ended at intersection" in out


# --- Section ordering + absence ---

def test_sections_separated_by_blank_lines():
    sub = CustomerSubmission(
        business_name="Acme",
        policy_dates={"effective_date": "2026-05-01", "expiration_date": "2027-05-01"},
        lob_details={"lob": "commercial_auto"},
    )
    out = explain(sub)
    # Three sections → two blank-line separators
    assert out.count("\n\n") >= 2


def test_partial_submission_omits_missing_sections():
    """Only business block present — no policy/lob/ai/loss noise."""
    out = explain(CustomerSubmission(business_name="Acme"))
    assert out == "Business: Acme"
    assert "Policy" not in out
    assert "Line of Business" not in out
    assert "Additional" not in out
    assert "Loss" not in out


def test_full_commercial_auto_submission_end_to_end():
    """Production-shape submission — every branch exercised."""
    sub = CustomerSubmission(
        business_name="Acme Trucking", dba="Acme Haulers",
        business_address={"line_one": "123 Main St", "city": "Detroit",
                          "state": "MI", "zip_code": "48201"},
        ein="12-3456789", email="ops@acme.com", phone="555-123-4567",
        policy_dates={"effective_date": "2026-05-01",
                      "expiration_date": "2027-05-01"},
        additional_interests=[
            {"name": "Ally Bank", "role": "lienholder"},
        ],
        loss_history=[
            {"date_of_loss": "2024-03-15", "type_of_loss": "collision",
             "amount_paid": 8500},
        ],
        lob_details={"lob": "commercial_auto",
                     "radius_of_operations": "long_haul", "hazmat": False,
                     "drivers": [{"first_name": "Alice", "last_name": "Nguyen"}],
                     "vehicles": [{"year": 2022, "make": "Ford", "model": "F-150",
                                   "vin": "1FTFW1E50NFA12345"}],
                     "coverage": {"liability_limit_csl": 1_000_000,
                                  "comp_deductible": 500}},
    )
    out = explain(sub)
    # Every section present, in order. Position-based because the LOB section
    # has its own internal "\n\n" separators between Drivers/Vehicles/Coverage.
    business_pos = out.find("Business: Acme Trucking")
    policy_pos = out.find("Policy: 2026-05-01")
    lob_pos = out.find("Line of Business: Commercial Auto")
    ai_pos = out.find("Additional Interests (1):")
    loss_pos = out.find("Loss History (1):")

    assert business_pos == 0
    assert 0 < policy_pos < lob_pos < ai_pos < loss_pos
    # Nested content rendered within the LOB block
    assert "Drivers (1):" in out
    assert "Alice Nguyen" in out
    assert "Vehicles (1):" in out
    assert "1FTFW1E50NFA12345" in out
    assert "Liability CSL: $1,000,000" in out
