"""Accord AI domain schema (Phase 10.S.1 — full v3 parity, approach B).

All models inherit _StrictModel (str_strip_whitespace=True,
validate_assignment=True, extra="ignore"). Optional fields default to
None so existing constructions like CustomerSubmission(business_name="X")
keep working.

extra="ignore" is deliberate. We're consuming LLM output where the model
places fields at plausible-but-wrong schema levels (e.g. ``hired_auto``
on ``CommercialAutoDetails`` when the schema puts it under
``CommercialAutoDetails.coverage``; ``prior_insurance`` at root when
only WorkersCompDetails has it). Under extra="forbid", one stray field
rejects the entire 50+ field extraction — observed as F1=0.0 on the
``bulk-*`` scenarios. With extra="ignore", the misplaced fields are
silently dropped and the rest of the extraction lands. Schema-drift
protection is covered by tests/test_schema_v3_port.py which constructs
models with explicit kwargs (still raises on unknown typing).

Types are real wherever meaningful (date / int / Decimal / Literal) rather
than v3's everything-is-string legacy. Extractor prompt rendering (via
CustomerSubmission.model_json_schema()) picks up the richer types
automatically — no prompt-side change needed here.

Cull record (v3 → v4, approach B): fax, deposit_amount, estimated_premium,
Driver.ssn_or_tax_id, Driver.full_name, Driver.license_type,
Vehicle.modified_equipment, BusinessInfo.experience_mod (moved to
WorkersCompDetails), BusinessInfo root-level phone/email (folded into
Contact), CyberInfo. See tests/test_schema_v3_port.py for regression locks.
"""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Annotated, Any, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        # "ignore" (NOT "forbid") — see module docstring. LLM output
        # places fields at plausible-but-wrong schema levels; forbid
        # rejected the whole submission, ignore keeps the valid fields.
        extra="ignore",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


# ---------------------------------------------------------------------------
# Enrichment conflicts (stored alongside the submission, not extracted from LLM)
# ---------------------------------------------------------------------------

class FieldConflict(_StrictModel):
    """Records a disagreement between user-supplied data and enriched data.

    Created by inline enrichers (e.g. NHTSA vPIC, Zippopotam) when what the
    user said differs from what an authoritative source says. Surfaced at the
    finalize review screen — never silently overwritten inline.

    `resolved` is flipped to True when the user confirms or the controller
    accepts the enriched value. Only unresolved conflicts block ready_to_finalize.
    """
    field_path: str
    user_value: Any
    enriched_value: Any
    source: str                                         # "nhtsa_vpic", "zippopotam", …
    created_at: datetime = Field(default_factory=lambda: datetime.now())
    resolved: bool = False


# ---------------------------------------------------------------------------
# Shared enums
# ---------------------------------------------------------------------------

EntityType = Literal[
    "corporation", "partnership", "llc", "individual",
    "subchapter_s", "joint_venture", "not_for_profit", "trust",
]

PolicyStatus = Literal["new", "renewal", "rewrite"]
BillingPlan  = Literal["direct", "agency"]
PaymentPlan  = Literal["monthly", "quarterly", "semi_annual", "annual", "one_time"]
Sex          = Literal["M", "F"]
MaritalStatus = Literal["S", "M", "W", "D", "P"]    # Single/Married/Widowed/Divorced/Partner
MvrStatus    = Literal["clean", "minor_violations", "major_violations", "suspended"]
DriverRelationship = Literal["employee", "owner", "family", "other"]
VehicleUseType = Literal["service", "commercial", "retail", "pleasure"]
VehicleType  = Literal["private_passenger", "commercial", "special"]


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class Address(_StrictModel):
    line_one: Optional[str] = None
    line_two: Optional[str] = None
    city:     Optional[str] = None
    state:    Optional[str] = None          # 2-letter state code
    zip_code: Optional[str] = None
    county:   Optional[str] = None


class Contact(_StrictModel):
    """A person-level contact attached to a business, producer, or interest."""
    full_name: Optional[str] = None
    phone:     Optional[str] = None
    email:     Optional[str] = None
    role:      Optional[str] = None         # e.g. "CFO", "Fleet Manager"


class Producer(_StrictModel):
    """Agency / broker filling the submission."""
    agency_name:               Optional[str] = None
    contact_name:              Optional[str] = None
    # Name of the person signing the app on behalf of the agency — distinct
    # from contact_name (day-to-day contact). Maps to
    # Producer_AuthorizedRepresentative_FullName_A on ACORD 125/126/127/131/160.
    authorized_representative: Optional[str] = None
    phone:                     Optional[str] = None
    email:                     Optional[str] = None
    mailing_address:           Optional[Address] = None
    # Carrier-internal identifier the broker uses for the insured.
    producer_code:             Optional[str] = None
    # Industry-standard NPN (NAIC/NIPR) — distinct from producer_code. Maps
    # to Producer_NationalIdentifier_A on ACORD 125/126/127/130/131/137/160.
    national_producer_number:  Optional[str] = None
    # State producer license (e.g. "TX-987654").
    license_number:            Optional[str] = None


class Violation(_StrictModel):
    """A single driver violation (maps to ACORD 127 AccidentConviction_* slots).

    Field is `occurred_on` not `date` to avoid shadowing datetime.date in the
    class scope — with `from __future__ import annotations` that shadow makes
    Pydantic mis-resolve the field's own type.
    """
    occurred_on:    Optional[date] = None
    type:           Optional[str] = None        # "accident", "conviction", "suspension"
    description:    Optional[str] = None
    location_city:  Optional[str] = None
    location_state: Optional[str] = None


class Driver(_StrictModel):
    first_name:       Optional[str] = None
    middle_initial:   Optional[str] = None
    last_name:        Optional[str] = None
    date_of_birth:    Optional[date] = None
    sex:              Optional[Sex] = None
    marital_status:   Optional[MaritalStatus] = None
    license_number:     Optional[str] = None
    license_state:      Optional[str] = None
    licensed_year:      Optional[int] = None       # 4-digit year first licensed
    license_expiration: Optional[date] = None
    years_experience: Optional[int] = None
    hire_date:        Optional[date] = None
    occupation:       Optional[str] = None
    relationship:     Optional[DriverRelationship] = None
    vehicle_assigned: Optional[int] = None       # index into vehicles[]
    pct_use:          Optional[Decimal] = None   # percentage 0-100
    mvr_status:       Optional[MvrStatus] = None
    mailing_address:  Optional[Address] = None
    violations:       List[Violation] = Field(default_factory=list)


class Vehicle(_StrictModel):
    year:                    Optional[int] = None
    make:                    Optional[str] = None
    model:                   Optional[str] = None
    vin:                     Optional[str] = None
    serial_number:           Optional[str] = None   # non-road equipment (ACORD 160)
    body_type:               Optional[str] = None
    vehicle_type:            Optional[VehicleType] = None
    gvw:                     Optional[int] = None   # gross vehicle weight, lbs
    cost_new:                Optional[Decimal] = None
    stated_amount:           Optional[Decimal] = None
    use_type:                Optional[VehicleUseType] = None
    radius_of_travel:        Optional[int] = None   # miles
    farthest_zone:           Optional[str] = None   # zone code
    territory:               Optional[str] = None
    class_code:              Optional[str] = None
    annual_mileage:          Optional[int] = None
    seating_capacity:        Optional[int] = None
    registration_state:      Optional[str] = None   # 2-letter
    garage_address:          Optional[Address] = None
    # Per-vehicle coverage selections (P10.S.10a) — matches ACORD 127/129
    # Vehicle_Coverage_* widget family. Canonical location for per-vehicle
    # deductibles and selection state (collision/comprehensive deductibles
    # live on coverage.*, not on Vehicle directly — deduplicated in P10.0.f.2).
    coverage:                Optional[VehicleCoverage] = None


class AdditionalInterest(_StrictModel):
    name:             Optional[str] = None
    address:          Optional[Address] = None
    role:             Optional[str] = None       # "loss_payee", "additional_insured", "mortgagee"
    interest_type:    Optional[str] = None       # free-form detail
    loan_number:      Optional[str] = None
    reference_number: Optional[str] = None


class LossHistory(_StrictModel):
    date_of_loss:        Optional[date] = None
    type_of_loss:        Optional[str] = None
    amount_paid:         Optional[Decimal] = None
    description:         Optional[str] = None
    claim_status:        Optional[str] = None    # "open", "closed", "subrogating"
    claim_number:        Optional[str] = None
    insurer_name:        Optional[str] = None
    loss_location_state: Optional[str] = None    # 2-letter


class PriorInsurance(_StrictModel):
    """Carrier/policy history — maps to ACORD 130 PriorCoverage_* slots."""
    carrier_name:    Optional[str] = None
    policy_number:   Optional[str] = None
    effective_date:  Optional[date] = None
    expiration_date: Optional[date] = None
    premium_amount:  Optional[Decimal] = None


class Location(_StrictModel):
    """Premises / property location — used by GL + WC for operations + rating."""
    address:                Optional[Address] = None
    description:            Optional[str] = None
    annual_gross_receipts:  Optional[Decimal] = None
    annual_payroll:         Optional[Decimal] = None
    full_time_employees:    Optional[int] = None
    part_time_employees:    Optional[int] = None
    building_occupancy:     Optional[str] = None


class Classification(_StrictModel):
    class_code:             Optional[str] = None
    description:            Optional[str] = None
    naics_code:             Optional[str] = None
    annual_gross_receipts:  Optional[Decimal] = None
    annual_payroll:         Optional[Decimal] = None


class PayrollByClass(_StrictModel):
    class_code:     Optional[str] = None
    description:    Optional[str] = None
    payroll:        Optional[Decimal] = None
    employee_count: Optional[int] = None
    state:          Optional[str] = None   # 2-letter


class PolicyDates(_StrictModel):
    effective_date:  Optional[date] = None
    expiration_date: Optional[date] = None


# ---------------------------------------------------------------------------
# Per-LOB coverages
# ---------------------------------------------------------------------------

class VehicleCoverage(_StrictModel):
    """Per-vehicle coverage selections and deductibles (P10.S.10a).

    Distinct from CommercialAutoCoverage, which holds policy-wide limits
    (e.g. BI/PD/CSL). This block controls what each individual vehicle
    carries — collision on truck #1 but not #2, different deductibles
    per unit, etc.

    Fields match ACORD 127/129's Vehicle_Coverage_* widget family.
    """
    liability:                       Optional[bool]    = None
    collision:                       Optional[bool]    = None
    comprehensive:                   Optional[bool]    = None
    collision_deductible_amount:     Optional[Decimal] = None
    comprehensive_deductible_amount: Optional[Decimal] = None
    medical_payments:                Optional[bool]    = None
    uninsured_motorists:             Optional[bool]    = None
    underinsured_motorists:          Optional[bool]    = None
    towing_labour:                   Optional[bool]    = None
    rental_reimbursement:            Optional[bool]    = None


class CommercialAutoCoverage(_StrictModel):
    liability_limit_csl:  Optional[int] = None
    bi_per_person:        Optional[int] = None
    bi_per_accident:      Optional[int] = None
    pd_per_accident:      Optional[int] = None
    uim_limit:            Optional[int] = None
    medpay_limit:         Optional[int] = None
    comp_deductible:      Optional[int] = None
    coll_deductible:      Optional[int] = None
    hired_auto:           Optional[bool] = None
    non_owned_auto:       Optional[bool] = None


class GeneralLiabilityCoverage(_StrictModel):
    each_occurrence_limit:              Optional[int] = None
    general_aggregate_limit:            Optional[int] = None
    products_ops_aggregate_limit:       Optional[int] = None
    personal_advertising_injury_limit:  Optional[int] = None
    damage_to_rented_premises_limit:    Optional[int] = None
    medical_expense_limit:              Optional[int] = None
    deductible:                         Optional[int] = None
    claims_made_basis:                  Optional[bool] = None


class WorkersCompCoverage(_StrictModel):
    employers_liability_per_accident: Optional[int] = None
    employers_liability_per_employee: Optional[int] = None
    employers_liability_per_policy:   Optional[int] = None


# ---------------------------------------------------------------------------
# Per-LOB details
# ---------------------------------------------------------------------------

class CommercialAutoDetails(_StrictModel):
    lob: Literal["commercial_auto"] = "commercial_auto"
    drivers:              List[Driver]  = Field(default_factory=list)
    vehicles:             List[Vehicle] = Field(default_factory=list)
    driver_count:         Optional[int] = None
    vehicle_count:        Optional[int] = None
    radius_of_operations: Optional[str] = None
    hazmat:               Optional[bool] = None
    coverage:             Optional[CommercialAutoCoverage] = None

    # --- P10.0.g.8 additions: v3 CA-critical fields previously absent ---
    # Sourced from accord_ai_v3/lobs/commercial_auto/__init__.py's
    # get_required_fields()['critical'] list. `fleet_use_type` is the
    # operations-wide default use_type (v4 already models per-vehicle
    # Vehicle.use_type; this is the broker-level "fleet is used for X"
    # summary that appears on ACORD 125's operations block).
    fleet_use_type:       Optional[VehicleUseType] = None
    fleet_for_hire:       Optional[bool] = None
    states_of_operation:  List[str] = Field(default_factory=list)  # 2-letter state codes
    trailer_interchange:  Optional[bool] = None
    driver_training:      Optional[bool] = None

    # FMCSA DOT number — present for licensed trucking carriers
    fmcsa_dot_number:     Optional[str] = None


class GeneralLiabilityDetails(_StrictModel):
    lob: Literal["general_liability"] = "general_liability"
    employee_count:   Optional[int] = None
    classifications:  List[Classification] = Field(default_factory=list)
    coverage:         Optional[GeneralLiabilityCoverage] = None


class WorkersCompDetails(_StrictModel):
    lob: Literal["workers_comp"] = "workers_comp"
    # Experience modifier is a rate (0.95 / 1.05 style), not a dollar amount
    # — float is the right type and avoids Decimal/float equality surprises.
    experience_mod:        Optional[float] = None
    owner_exclusion:       Optional[bool] = None
    waiver_of_subrogation: Optional[bool] = None
    payroll_by_class:      List[PayrollByClass]  = Field(default_factory=list)
    prior_insurance:       List[PriorInsurance]  = Field(default_factory=list)   # up to 5 slots on ACORD 130
    coverage:              Optional[WorkersCompCoverage] = None


LobDetails = Annotated[
    Union[CommercialAutoDetails, GeneralLiabilityDetails, WorkersCompDetails],
    Field(discriminator="lob"),
]


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

class CustomerSubmission(_StrictModel):
    # Identity
    business_name:       Optional[str] = None
    dba:                 Optional[str] = None
    ein:                 Optional[str] = None
    website:             Optional[str] = None
    entity_type:         Optional[EntityType] = None
    business_start_date: Optional[date] = None
    years_in_business:   Optional[int] = None

    # Classification
    naics_code:             Optional[str] = None
    naics_description:      Optional[str] = None
    sic_code:               Optional[str] = None
    # NCCI class code — required for accurate Workers Comp rating.
    # NAICS/SIC are business-activity codes; NCCI is the WC-specific
    # rate-book code. Research 2026-04-24 (INSURANCE_SCHEMAS.md) flagged
    # this gap: without ncci_code, WC LOB cannot be properly rated.
    # Format: 4-digit string (e.g. "8810" = Clerical Office, "5606" = Executive Supervisor).
    ncci_code:              Optional[str] = None
    ncci_description:       Optional[str] = None
    operations_description: Optional[str] = None
    nature_of_business:     Optional[str] = None

    # Addresses
    business_address: Optional[Address] = None
    mailing_address:  Optional[Address] = None

    # Contacts
    email:    Optional[str] = None
    phone:    Optional[str] = None
    contacts: List[Contact] = Field(default_factory=list)

    # Workforce + financials
    full_time_employees: Optional[int]     = None
    part_time_employees: Optional[int]     = None
    annual_revenue:      Optional[Decimal] = None
    annual_payroll:      Optional[Decimal] = None
    subcontractor_cost:  Optional[Decimal] = None

    # Policy framing
    policy_number: Optional[str]          = None
    policy_status: Optional[PolicyStatus] = None
    policy_dates:  Optional[PolicyDates]  = None
    billing_plan:  Optional[BillingPlan]  = None
    payment_plan:  Optional[PaymentPlan]  = None

    # Producer (agency filling the submission)
    producer: Optional[Producer] = None

    # Supporting lists
    additional_interests: List[AdditionalInterest] = Field(default_factory=list)
    loss_history:         List[LossHistory]        = Field(default_factory=list)
    locations:            List[Location]           = Field(default_factory=list)

    # LOB-specific
    lob_details: Optional[LobDetails] = None

    # Enrichment conflicts — written by inline enrichers, never by the LLM.
    # Excluded from the extraction schema (see extractor.py) to prevent the
    # LLM from hallucinating conflict entries.
    conflicts: List[FieldConflict] = Field(default_factory=list)
