"""Explainer — CustomerSubmission → human-readable multi-section text.

Pure function. No I/O, no LLM, deterministic. Used by:
  - Responder prompt composition (may replace model_dump_json)
  - API endpoints that show a session summary
  - Debug / ops views
  - Tests that assert on current state readably

Output is plain text — ready for UI display or LLM prompt embedding.
Empty/missing fields are OMITTED (not rendered as "(none)") so partial
submissions look cleaner than a schema dump.
"""
from __future__ import annotations

from typing import List, Optional

from accord_ai.schema import (
    AdditionalInterest,
    Address,
    Classification,
    CommercialAutoCoverage,
    CommercialAutoDetails,
    CustomerSubmission,
    Driver,
    GeneralLiabilityCoverage,
    GeneralLiabilityDetails,
    LossHistory,
    PayrollByClass,
    PolicyDates,
    Vehicle,
    WorkersCompCoverage,
    WorkersCompDetails,
)


def explain(submission: CustomerSubmission) -> str:
    """Render a CustomerSubmission as human-readable multi-section text."""
    sections: List[str] = []

    block = _explain_business(submission)
    if block:
        sections.append(block)
    block = _explain_policy(submission.policy_dates)
    if block:
        sections.append(block)
    block = _explain_lob(submission.lob_details)
    if block:
        sections.append(block)
    block = _explain_additional_interests(list(submission.additional_interests))
    if block:
        sections.append(block)
    block = _explain_loss_history(list(submission.loss_history))
    if block:
        sections.append(block)

    if not sections:
        return "(empty submission)"
    return "\n\n".join(sections)


# --- Business info ---

def _explain_business(s: CustomerSubmission) -> str:
    lines: List[str] = []
    if s.business_name and s.dba:
        lines.append(f"Business: {s.business_name} (DBA: {s.dba})")
    elif s.business_name:
        lines.append(f"Business: {s.business_name}")
    elif s.dba:
        lines.append(f"DBA: {s.dba}")

    if s.business_address:
        addr = _format_address(s.business_address)
        if addr:
            lines.append(f"Address: {addr}")
    if s.mailing_address:
        addr = _format_address(s.mailing_address)
        if addr:
            lines.append(f"Mailing: {addr}")

    if s.ein:
        lines.append(f"EIN: {s.ein}")
    if s.email:
        lines.append(f"Email: {s.email}")
    if s.phone:
        lines.append(f"Phone: {s.phone}")
    return "\n".join(lines)


def _format_address(a: Address) -> str:
    parts: List[str] = []
    if a.line_one:
        parts.append(a.line_one)
    if a.line_two:
        parts.append(a.line_two)
    city_state_zip: List[str] = []
    if a.city:
        city_state_zip.append(a.city)
    state_zip = " ".join(x for x in (a.state, a.zip_code) if x)
    if state_zip:
        city_state_zip.append(state_zip)
    if city_state_zip:
        parts.append(", ".join(city_state_zip))
    return ", ".join(parts)


# --- Policy dates ---

def _explain_policy(p: Optional[PolicyDates]) -> str:
    if p is None or (p.effective_date is None and p.expiration_date is None):
        return ""
    eff = str(p.effective_date) if p.effective_date else "(not set)"
    exp = str(p.expiration_date) if p.expiration_date else "(not set)"
    return f"Policy: {eff} to {exp}"


# --- LOB dispatch ---

def _explain_lob(lob) -> str:
    if lob is None:
        return ""
    if isinstance(lob, CommercialAutoDetails):
        return _explain_commercial_auto(lob)
    if isinstance(lob, GeneralLiabilityDetails):
        return _explain_general_liability(lob)
    if isinstance(lob, WorkersCompDetails):
        return _explain_workers_comp(lob)
    return f"Line of Business: {getattr(lob, 'lob', 'unknown')}"


# --- Commercial Auto ---

def _explain_commercial_auto(d: CommercialAutoDetails) -> str:
    lines: List[str] = ["Line of Business: Commercial Auto"]
    if d.radius_of_operations:
        lines.append(f"Radius of operations: {d.radius_of_operations}")
    if d.hazmat is not None:
        lines.append(f"Hazmat: {'yes' if d.hazmat else 'no'}")

    if d.drivers:
        count = d.driver_count if d.driver_count is not None else len(d.drivers)
        lines.append("")
        lines.append(f"Drivers ({count}):")
        for drv in d.drivers:
            lines.append(f"  - {_format_driver(drv)}")
    elif d.driver_count is not None:
        lines.append(f"Drivers declared: {d.driver_count}")

    if d.vehicles:
        count = d.vehicle_count if d.vehicle_count is not None else len(d.vehicles)
        lines.append("")
        lines.append(f"Vehicles ({count}):")
        for v in d.vehicles:
            lines.append(f"  - {_format_vehicle(v)}")
    elif d.vehicle_count is not None:
        lines.append(f"Vehicles declared: {d.vehicle_count}")

    if d.coverage:
        cov = _format_ca_coverage(d.coverage)
        if cov:
            lines.append("")
            lines.append("Coverage:")
            for c in cov:
                lines.append(f"  - {c}")
    return "\n".join(lines)


def _format_driver(drv: Driver) -> str:
    name_parts = [p for p in (drv.first_name, drv.middle_initial, drv.last_name) if p]
    name = " ".join(name_parts) or "(unnamed)"
    extras: List[str] = []
    if drv.date_of_birth:
        extras.append(f"DOB {drv.date_of_birth}")
    if drv.license_number:
        prefix = f"{drv.license_state}-" if drv.license_state else ""
        extras.append(f"License {prefix}{drv.license_number}")
    if drv.years_experience is not None:
        extras.append(f"{drv.years_experience}y exp")
    return name + (f" ({', '.join(extras)})" if extras else "")


def _format_vehicle(v: Vehicle) -> str:
    ymm = " ".join(str(x) for x in (v.year, v.make, v.model) if x)
    extras: List[str] = []
    if v.vin:
        extras.append(f"VIN {v.vin}")
    if v.body_type:
        extras.append(v.body_type)
    if v.garage_address:
        ga = _format_address(v.garage_address)
        if ga:
            extras.append(f"garaged at {ga}")
    desc = ymm or "(unspecified)"
    return desc + (f" ({', '.join(extras)})" if extras else "")


def _format_ca_coverage(c: CommercialAutoCoverage) -> List[str]:
    out: List[str] = []
    if c.liability_limit_csl is not None:
        out.append(f"Liability CSL: ${c.liability_limit_csl:,}")
    if c.bi_per_person is not None:
        out.append(f"BI per person: ${c.bi_per_person:,}")
    if c.bi_per_accident is not None:
        out.append(f"BI per accident: ${c.bi_per_accident:,}")
    if c.pd_per_accident is not None:
        out.append(f"PD per accident: ${c.pd_per_accident:,}")
    if c.uim_limit is not None:
        out.append(f"UIM: ${c.uim_limit:,}")
    if c.medpay_limit is not None:
        out.append(f"MedPay: ${c.medpay_limit:,}")
    if c.comp_deductible is not None:
        out.append(f"Comp deductible: ${c.comp_deductible:,}")
    if c.coll_deductible is not None:
        out.append(f"Coll deductible: ${c.coll_deductible:,}")
    if c.hired_auto is not None:
        out.append(f"Hired auto: {'yes' if c.hired_auto else 'no'}")
    if c.non_owned_auto is not None:
        out.append(f"Non-owned auto: {'yes' if c.non_owned_auto else 'no'}")
    return out


# --- General Liability ---

def _explain_general_liability(d: GeneralLiabilityDetails) -> str:
    lines: List[str] = ["Line of Business: General Liability"]
    if d.employee_count is not None:
        lines.append(f"Employees: {d.employee_count}")
    if d.classifications:
        lines.append("")
        lines.append(f"Classifications ({len(d.classifications)}):")
        for c in d.classifications:
            lines.append(f"  - {_format_classification(c)}")
    if d.coverage:
        cov = _format_gl_coverage(d.coverage)
        if cov:
            lines.append("")
            lines.append("Coverage:")
            for x in cov:
                lines.append(f"  - {x}")
    return "\n".join(lines)


def _format_classification(c: Classification) -> str:
    parts: List[str] = []
    if c.class_code:
        parts.append(f"class {c.class_code}")
    if c.naics_code:
        parts.append(f"NAICS {c.naics_code}")
    if c.description:
        parts.append(c.description)
    extras: List[str] = []
    if c.annual_gross_receipts is not None:
        extras.append(f"receipts ${c.annual_gross_receipts:,}")
    if c.annual_payroll is not None:
        extras.append(f"payroll ${c.annual_payroll:,}")
    core = " ".join(parts) if parts else "(unspecified)"
    return core + (f" ({', '.join(extras)})" if extras else "")


def _format_gl_coverage(c: GeneralLiabilityCoverage) -> List[str]:
    out: List[str] = []
    if c.each_occurrence_limit is not None:
        out.append(f"Each occurrence: ${c.each_occurrence_limit:,}")
    if c.general_aggregate_limit is not None:
        out.append(f"General aggregate: ${c.general_aggregate_limit:,}")
    if c.products_ops_aggregate_limit is not None:
        out.append(f"Products/ops aggregate: ${c.products_ops_aggregate_limit:,}")
    if c.personal_advertising_injury_limit is not None:
        out.append(f"Personal/advertising injury: ${c.personal_advertising_injury_limit:,}")
    if c.damage_to_rented_premises_limit is not None:
        out.append(f"Damage to rented premises: ${c.damage_to_rented_premises_limit:,}")
    if c.medical_expense_limit is not None:
        out.append(f"Medical expense: ${c.medical_expense_limit:,}")
    if c.deductible is not None:
        out.append(f"Deductible: ${c.deductible:,}")
    if c.claims_made_basis is not None:
        out.append(f"Basis: {'claims-made' if c.claims_made_basis else 'occurrence'}")
    return out


# --- Workers Comp ---

def _explain_workers_comp(d: WorkersCompDetails) -> str:
    lines: List[str] = ["Line of Business: Workers Compensation"]
    if d.experience_mod is not None:
        lines.append(f"Experience mod: {d.experience_mod}")
    if d.owner_exclusion is not None:
        lines.append(f"Owner exclusion: {'yes' if d.owner_exclusion else 'no'}")
    if d.waiver_of_subrogation is not None:
        lines.append(f"Waiver of subrogation: {'yes' if d.waiver_of_subrogation else 'no'}")
    if d.payroll_by_class:
        lines.append("")
        lines.append(f"Payroll by class ({len(d.payroll_by_class)}):")
        for p in d.payroll_by_class:
            lines.append(f"  - {_format_payroll(p)}")
    if d.coverage:
        cov = _format_wc_coverage(d.coverage)
        if cov:
            lines.append("")
            lines.append("Employer's Liability:")
            for x in cov:
                lines.append(f"  - {x}")
    return "\n".join(lines)


def _format_payroll(p: PayrollByClass) -> str:
    header_parts: List[str] = []
    if p.class_code:
        header_parts.append(f"class {p.class_code}")
    if p.description:
        header_parts.append(p.description)
    if p.state:
        header_parts.append(f"({p.state})")
    extras: List[str] = []
    if p.payroll is not None:
        extras.append(f"${p.payroll:,}")
    if p.employee_count is not None:
        extras.append(f"{p.employee_count} empl")
    core = " ".join(header_parts) if header_parts else "(unspecified)"
    return core + (f" — {', '.join(extras)}" if extras else "")


def _format_wc_coverage(c: WorkersCompCoverage) -> List[str]:
    out: List[str] = []
    if c.employers_liability_per_accident is not None:
        out.append(f"Per accident: ${c.employers_liability_per_accident:,}")
    if c.employers_liability_per_employee is not None:
        out.append(f"Per employee: ${c.employers_liability_per_employee:,}")
    if c.employers_liability_per_policy is not None:
        out.append(f"Per policy: ${c.employers_liability_per_policy:,}")
    return out


# --- Additional interests + loss history ---

def _explain_additional_interests(interests: List[AdditionalInterest]) -> str:
    if not interests:
        return ""
    lines = [f"Additional Interests ({len(interests)}):"]
    for ai in interests:
        lines.append(f"  - {_format_additional_interest(ai)}")
    return "\n".join(lines)


def _format_additional_interest(ai: AdditionalInterest) -> str:
    name = ai.name or "(unnamed)"
    role = f" ({ai.role})" if ai.role else ""
    addr_txt = ""
    if ai.address:
        a = _format_address(ai.address)
        if a:
            addr_txt = f" at {a}"
    return f"{name}{role}{addr_txt}"


def _explain_loss_history(losses: List[LossHistory]) -> str:
    if not losses:
        return ""
    lines = [f"Loss History ({len(losses)}):"]
    for lh in losses:
        lines.append(f"  - {_format_loss(lh)}")
    return "\n".join(lines)


def _format_loss(lh: LossHistory) -> str:
    date = str(lh.date_of_loss) if lh.date_of_loss else "(date unknown)"
    kind = lh.type_of_loss or "unknown type"
    amount = f", paid ${lh.amount_paid:,}" if lh.amount_paid is not None else ""
    status = f", {lh.claim_status}" if lh.claim_status else ""
    desc = f" — {lh.description}" if lh.description else ""
    return f"{date}: {kind}{amount}{status}{desc}"
