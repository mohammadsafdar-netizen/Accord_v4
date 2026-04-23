"""Frontend human-label → PDF widget-name translator.

The BroCopilot frontend sends /complete payloads keyed by the friendly labels
from the 1,130-field baseline spec (e.g. "Driver First Name A", "Vehicle VIN
B", "Named Insured"), not by PDF widget names. Widget names use a different
PascalCase_with_underscores convention (e.g. "Driver_GivenName_A",
"Vehicle_VINIdentifier_B", "NamedInsured_FullName_A").

This module translates between the two. It also filters the sentinel value
"NullObject" (the FE's marker for "user did not fill this field") so we don't
literally render the string "NullObject" in the output PDF.

The label map is built programmatically from patterns observed in real FE
traffic plus the authoritative baseline spec. When a key is neither a known
label nor a widget name, we pass it through untouched (widget fill will
silently drop it; the api observability log will flag it).
"""

from __future__ import annotations

import string
from typing import Any


# Values the FE sends for "empty" / "null". These must NOT be written to
# widgets — they'd literally appear as the text "NullObject" on the PDF.
NULL_SENTINELS = frozenset({
    "NullObject", "nullobject", "null", "None", "none", "undefined", "N/A", "n/a",
})


def _is_null(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return True
        if s in NULL_SENTINELS:
            return True
    return False


# ---------------------------------------------------------------------------
# Label map construction — one dict per form.
#
# Keys are the EXACT strings the FE sends (including odd whitespace/punct).
# Values are PDF widget names. Slot letters (A-M, A-E, A-D, 1-9) are
# expanded programmatically to keep this readable.
# ---------------------------------------------------------------------------


def _expand(prefix_label: str, widget_template: str, slots: str) -> dict[str, str]:
    """Build {f"{prefix_label} {slot}": widget_template.format(slot=slot)}."""
    return {f"{prefix_label} {s}": widget_template.format(slot=s) for s in slots}


# Common header fields shared across ACORD forms (no slot suffix).
_COMMON: dict[str, str] = {
    "Named Insured": "NamedInsured_FullName_A",
    "Named Insured DBA": "NamedInsured_FullName_B",
    "Policy Effective Date": "Policy_EffectiveDate_A",
    "Policy Expiration Date": "Policy_ExpirationDate_A",
    "Policy Number": "Policy_PolicyNumberIdentifier_A",
    "Producer Name": "Producer_FullName_A",
    "Producer Contact Name": "Producer_ContactPerson_FullName_A",
    "Producer Phone": "Producer_ContactPerson_PhoneNumber_A",
    "Producer Email": "Producer_ContactPerson_EmailAddress_A",
    "Producer Fax": "Producer_FaxNumber_A",
    "Producer Customer ID": "Producer_CustomerIdentifier_A",
    "Producer License": "Producer_StateLicenseIdentifier_A",
    "Producer Address": "Producer_MailingAddress_LineOne_A",
    "Producer Address Line 2": "Producer_MailingAddress_LineTwo_A",
    "Producer City": "Producer_MailingAddress_CityName_A",
    "Producer State": "Producer_MailingAddress_StateOrProvinceCode_A",
    "Producer ZIP": "Producer_MailingAddress_PostalCode_A",
    "Mailing Address": "NamedInsured_MailingAddress_LineOne_A",
    "Mailing Address Line 2": "NamedInsured_MailingAddress_LineTwo_A",
    "City": "NamedInsured_MailingAddress_CityName_A",
    "State": "NamedInsured_MailingAddress_StateOrProvinceCode_A",
    "ZIP Code": "NamedInsured_MailingAddress_PostalCode_A",
    "FEIN": "NamedInsured_TaxIdentifier_A",
    "SIC Code": "NamedInsured_SICCode_A",
    "NAICS Code": "NamedInsured_NAICSCode_A",
    "Business Start Date": "NamedInsured_BusinessStartDate_A",
    "Contact Name": "NamedInsured_Contact_FullName_A",
    "Phone Number": "NamedInsured_Contact_PrimaryPhoneNumber_A",
    "Email Address": "NamedInsured_Contact_PrimaryEmailAddress_A",
    "Contact Description": "NamedInsured_Contact_ContactDescription_A",
    "Website": "NamedInsured_Primary_WebsiteAddress_A",
    "Full Time Employees": "BusinessInformation_FullTimeEmployeeCount_A",
    "Part Time Employees": "BusinessInformation_PartTimeEmployeeCount_A",
}

# Legal-entity indicators (125).
_LEGAL_ENTITY: dict[str, str] = {
    "Corporation Indicator": "NamedInsured_LegalEntity_CorporationIndicator_A",
    "Partnership Indicator": "NamedInsured_LegalEntity_PartnershipIndicator_A",
    "LLC Indicator": "NamedInsured_LegalEntity_LimitedLiabilityCorporationIndicator_A",
    "Individual Indicator": "NamedInsured_LegalEntity_IndividualIndicator_A",
    "S-Corp Indicator": "NamedInsured_LegalEntity_SubchapterSCorporationIndicator_A",
    "Joint Venture Indicator": "NamedInsured_LegalEntity_JointVentureIndicator_A",
    "Not For Profit Indicator": "NamedInsured_LegalEntity_NotForProfitIndicator_A",
    "Trust Indicator": "NamedInsured_LegalEntity_TrustIndicator_A",
}

# Policy / payment / LOB indicators (125).
_POLICY_125: dict[str, str] = {
    "Quote Indicator": "Policy_Status_QuoteIndicator_A",
    "Renew Indicator": "Policy_Status_RenewIndicator_A",
    "Bound Indicator": "Policy_Status_BoundIndicator_A",
    "Direct Bill Indicator": "Policy_Payment_DirectBillIndicator_A",
    "Producer Bill Indicator": "Policy_Payment_ProducerBillIndicator_A",
    "Payment Schedule": "Policy_Payment_PaymentScheduleCode_A",
    "Deposit Amount": "Policy_Payment_DepositAmount_A",
    "Estimated Total Amount": "Policy_Payment_EstimatedTotalAmount_A",
    "Minimum Premium Amount": "Policy_Payment_MinimumPremiumAmount_A",
    "Business Auto Indicator": "Policy_LineOfBusiness_BusinessAutoIndicator_A",
    "Commercial Vehicle Premium": "CommercialVehicleLineOfBusiness_PremiumAmount_A",
    "Operations Description": "CommercialPolicy_OperationsDescription_A",
    "Flammable/Explosives Exposure": (
        "CommercialPolicy_AnyExposureToFlammableExplosivesChemicalsExplanation_A"
    ),
    "AAH Code": "CommercialPolicy_Question_AAHCode_A",
    "Remarks": "CommercialPolicy_RemarkText_A",
    "Safety Manual": (
        "CommercialPolicy_FormalSafetyProgram_SafetyManualIndicator_A"
    ),
    "Safety Position": (
        "CommercialPolicy_FormalSafetyProgram_SafetyPositionIndicator_B"
    ),
    "Monthly Safety Meetings": (
        "CommercialPolicy_FormalSafetyProgram_MonthlyMeetingsIndicator_B"
    ),
    "OSHA Compliance": "CommercialPolicy_FormalSafetyProgram_OSHAIndicator_B",
    "Other Safety Program": "CommercialPolicy_FormalSafetyProgram_OtherIndicator_B",
    "Other Safety Description": (
        "CommercialPolicy_FormalSafetyProgram_OtherDescription_B"
    ),
    "No Prior Losses": "LossHistory_NoPriorLossesIndicator_A",
    "Loss History Years": "LossHistory_InformationYearCount_A",
    "Loss History - Total Amount": "LossHistory_TotalAmount_A",
}

# Business type indicators (125).
_BUSINESS_TYPE: dict[str, str] = {
    "Business Type - Manufacturing": "BusinessInformation_BusinessType_ManufacturingIndicator_A",
    "Business Type - Office": "BusinessInformation_BusinessType_OfficeIndicator_A",
    "Business Type - Retail": "BusinessInformation_BusinessType_RetailIndicator_A",
    "Business Type - Restaurant": "BusinessInformation_BusinessType_RestaurantIndicator_A",
    "Business Type - Wholesale": "BusinessInformation_BusinessType_WholesaleIndicator_A",
    "Business Type - Service": "BusinessInformation_BusinessType_ServiceIndicator_A",
    "Business Type - Contractor": "BusinessInformation_BusinessType_ContractorIndicator_A",
    "Business Type - Institutional": "BusinessInformation_BusinessType_InstitutionalIndicator_A",
    "Business Type - Apartments": "BusinessInformation_BusinessType_ApartmentsIndicator_A",
    "Business Type - Condominiums": "BusinessInformation_BusinessType_CondominiumsIndicator_A",
    "Business Type - Other": "BusinessInformation_BusinessType_OtherIndicator_A",
    "Business Type - Other Description": "BusinessInformation_BusinessType_OtherDescription_A",
}

# Loss history slots A/B/C (125).
_LOSS_SLOTS = "ABC"
_LOSS: dict[str, str] = {}
for s in _LOSS_SLOTS:
    _LOSS[f"Loss History - Occurrence Date {s}"] = f"LossHistory_OccurrenceDate_{s}"
    _LOSS[f"Loss History - Claim Date {s}"] = f"LossHistory_ClaimDate_{s}"
    _LOSS[f"Loss History - LOB {s}"] = f"LossHistory_LineOfBusiness_{s}"
    _LOSS[f"Loss History - Description {s}"] = f"LossHistory_OccurrenceDescription_{s}"
    _LOSS[f"Loss History - Paid Amount {s}"] = f"LossHistory_PaidAmount_{s}"
    _LOSS[f"Loss History - Reserved Amount {s}"] = f"LossHistory_ReservedAmount_{s}"
    _LOSS[f"Loss History - Open Status {s}"] = f"LossHistory_ClaimStatus_OpenCode_{s}"
    _LOSS[f"Loss History - Subrogation {s}"] = f"LossHistory_ClaimStatus_SubrogationCode_{s}"

# Prior coverage (auto) slots A/B/C (125).
_PRIOR: dict[str, str] = {}
for s in _LOSS_SLOTS:
    _PRIOR[f"Prior Auto Insurer {s}"] = f"PriorCoverage_Automobile_InsurerFullName_{s}"
    _PRIOR[f"Prior Auto Policy Number {s}"] = f"PriorCoverage_Automobile_PolicyNumberIdentifier_{s}"
    _PRIOR[f"Prior Auto Effective Date {s}"] = f"PriorCoverage_Automobile_EffectiveDate_{s}"
    _PRIOR[f"Prior Auto Expiration Date {s}"] = f"PriorCoverage_Automobile_ExpirationDate_{s}"
    _PRIOR[f"Prior Auto Premium {s}"] = f"PriorCoverage_Automobile_TotalPremiumAmount_{s}"
    _PRIOR[f"Prior Policy Year {s}"] = f"PriorCoverage_PolicyYear_{s}"


# Vehicle slot labels — A-E covers both 127 (A-D) and 129 (A-E). Extras map
# to non-existent widgets and get dropped harmlessly by the fill layer.
_VEHICLE_SLOTS = "ABCDE"


def _vehicle_labels() -> dict[str, str]:
    m: dict[str, str] = {}
    for s in _VEHICLE_SLOTS:
        m[f"Vehicle VIN {s}"] = f"Vehicle_VINIdentifier_{s}"
        m[f"Vehicle Year {s}"] = f"Vehicle_ModelYear_{s}"
        m[f"Vehicle Make {s}"] = f"Vehicle_ManufacturersName_{s}"
        m[f"Vehicle Model {s}"] = f"Vehicle_ModelName_{s}"
        m[f"Vehicle Body Code {s}"] = f"Vehicle_BodyCode_{s}"
        m[f"Vehicle GVW {s}"] = f"Vehicle_GrossVehicleWeight_{s}"
        m[f"Vehicle Cost New {s}"] = f"Vehicle_CostNewAmount_{s}"
        m[f"Vehicle Type Private Passenger {s}"] = f"Vehicle_VehicleType_PrivatePassengerIndicator_{s}"
        m[f"Vehicle Type Commercial {s}"] = f"Vehicle_VehicleType_CommercialIndicator_{s}"
        m[f"Vehicle Type Special {s}"] = f"Vehicle_VehicleType_SpecialIndicator_{s}"
        m[f"Vehicle Seating Capacity {s}"] = f"Vehicle_SeatingCapacityCount_{s}"
        m[f"Vehicle Use Commercial {s}"] = f"Vehicle_Use_CommercialIndicator_{s}"
        m[f"Vehicle Use Service {s}"] = f"Vehicle_Use_ServiceIndicator_{s}"
        m[f"Vehicle Use Retail {s}"] = f"Vehicle_Use_RetailIndicator_{s}"
        m[f"Vehicle Use Pleasure {s}"] = f"Vehicle_Use_PleasureIndicator_{s}"
        m[f"Vehicle Use Farm {s}"] = f"Vehicle_Use_FarmIndicator_{s}"
        m[f"Vehicle Use For Hire {s}"] = f"Vehicle_Use_ForHireIndicator_{s}"
        m[f"Vehicle Use Other {s}"] = f"Vehicle_Use_OtherIndicator_{s}"
        m[f"Vehicle Use Other Desc {s}"] = f"Vehicle_Use_OtherDescription_{s}"
        m[f"Vehicle Under 15 Miles {s}"] = f"Vehicle_Use_UnderFifteenMilesIndicator_{s}"
        m[f"Vehicle 15+ Miles {s}"] = f"Vehicle_Use_FifteenMilesOrOverIndicator_{s}"
        m[f"Vehicle Industry Class {s}"] = f"Vehicle_SpecialIndustryClassCode_{s}"
        m[f"Vehicle Radius {s}"] = f"Vehicle_RadiusOfUse_{s}"
        m[f"Vehicle Farthest Zone {s}"] = f"Vehicle_FarthestZoneCode_{s}"
        m[f"Vehicle Territory {s}"] = f"Vehicle_RatingTerritoryCode_{s}"
        m[f"Vehicle Reg State {s}"] = f"Vehicle_Registration_StateOrProvinceCode_{s}"
        m[f"Vehicle Garage Address {s}"] = f"Vehicle_PhysicalAddress_LineOne_{s}"
        m[f"Vehicle Garage City {s}"] = f"Vehicle_PhysicalAddress_CityName_{s}"
        m[f"Vehicle Garage County {s}"] = f"Vehicle_PhysicalAddress_CountyName_{s}"
        m[f"Vehicle Garage State {s}"] = f"Vehicle_PhysicalAddress_StateOrProvinceCode_{s}"
        m[f"Vehicle Garage ZIP {s}"] = f"Vehicle_PhysicalAddress_PostalCode_{s}"
        m[f"Vehicle Liability {s}"] = f"Vehicle_Coverage_LiabilityIndicator_{s}"
        m[f"Vehicle Comprehensive {s}"] = f"Vehicle_Coverage_ComprehensiveIndicator_{s}"
        m[f"Vehicle Comp Deductible Ind {s}"] = f"Vehicle_Coverage_ComprehensiveDeductibleIndicator_{s}"
        m[f"Vehicle Comp/SCL Deductible {s}"] = (
            f"Vehicle_Coverage_ComprehensiveOrSpecifiedCauseOfLossDeductibleAmount_{s}"
        )
        m[f"Vehicle Collision {s}"] = f"Vehicle_Coverage_CollisionIndicator_{s}"
        m[f"Vehicle Collision Deductible {s}"] = f"Vehicle_Collision_DeductibleAmount_{s}"
        m[f"Vehicle Med Pay {s}"] = f"Vehicle_Coverage_MedicalPaymentsIndicator_{s}"
        m[f"Vehicle UM {s}"] = f"Vehicle_Coverage_UninsuredMotoristsIndicator_{s}"
        m[f"Vehicle UIM {s}"] = f"Vehicle_Coverage_UnderinsuredMotoristsIndicator_{s}"
        m[f"Vehicle Towing {s}"] = f"Vehicle_Coverage_TowingAndLabourIndicator_{s}"
        m[f"Vehicle Rental {s}"] = f"Vehicle_Coverage_RentalReimbursementIndicator_{s}"
        m[f"Vehicle SCL {s}"] = f"Vehicle_Coverage_SpecifiedCauseOfLossIndicator_{s}"
        m[f"Vehicle SCL Deductible Ind {s}"] = f"Vehicle_Coverage_SpecifiedCauseOfLossDeductibleIndicator_{s}"
        m[f"Vehicle No-Fault {s}"] = f"Vehicle_Coverage_NoFaultIndicator_{s}"
        m[f"Vehicle Add'l No-Fault {s}"] = f"Vehicle_Coverage_AdditionalNoFaultIndicator_{s}"
        m[f"Vehicle Fire {s}"] = f"Vehicle_Coverage_FireIndicator_{s}"
        m[f"Vehicle Fire & Theft {s}"] = f"Vehicle_Coverage_FireTheftIndicator_{s}"
        m[f"Vehicle Fire Theft Wind {s}"] = f"Vehicle_Coverage_FireTheftWindstormIndicator_{s}"
        m[f"Vehicle Full Glass {s}"] = f"Vehicle_Coverage_FullGlassIndicator_{s}"
        m[f"Vehicle Ltd Perils {s}"] = f"Vehicle_Coverage_LimitedSpecifiedPerilsIndicator_{s}"
        m[f"Vehicle Agreed/Stated Amt {s}"] = f"Vehicle_Coverage_AgreedOrStatedAmount_{s}"
        m[f"Vehicle ACV Valuation {s}"] = f"Vehicle_Coverage_ValuationActualCashValueIndicator_{s}"
        m[f"Vehicle Agreed Valuation {s}"] = f"Vehicle_Coverage_ValuationAgreedAmountIndicator_{s}"
        m[f"Vehicle Stated Valuation {s}"] = f"Vehicle_Coverage_ValuationStatedAmountIndicator_{s}"
        m[f"Vehicle Symbol {s}"] = f"Vehicle_SymbolCode_{s}"
        m[f"Vehicle Collision Symbol {s}"] = f"Vehicle_CollisionSymbolCode_{s}"
        m[f"Vehicle Comp Symbol {s}"] = f"Vehicle_ComprehensiveSymbolCode_{s}"
        m[f"Vehicle Rate Class {s}"] = f"Vehicle_RateClassCode_{s}"
        m[f"Vehicle Net Rating Factor {s}"] = f"Vehicle_NetRatingFactor_{s}"
        m[f"Vehicle Liability Rating Factor {s}"] = f"Vehicle_PrimaryLiabilityRatingFactor_{s}"
        m[f"Vehicle Premium {s}"] = f"Vehicle_TotalPremiumAmount_{s}"
        m[f"Vehicle Ref {s}"] = f"Vehicle_ProducerIdentifier_{s}"
    # Modified equipment (A-B on 127)
    for s in "AB":
        m[f"Modified Equipment Desc {s}"] = f"Vehicle_Question_ModifiedEquipmentDescription_{s}"
        m[f"Modified Equipment Cost {s}"] = f"Vehicle_Question_ModifiedEquipmentCostAmount_{s}"
    return m


_VEHICLE = _vehicle_labels()


def _driver_labels() -> dict[str, str]:
    m: dict[str, str] = {}
    for s in string.ascii_uppercase[:13]:  # A-M (13 drivers on 127)
        m[f"Driver First Name {s}"] = f"Driver_GivenName_{s}"
        m[f"Driver Last Name {s}"] = f"Driver_Surname_{s}"
        m[f"Driver Middle Initial {s}"] = f"Driver_OtherGivenNameInitial_{s}"
        m[f"Driver DOB {s}"] = f"Driver_BirthDate_{s}"
        m[f"Driver Gender {s}"] = f"Driver_GenderCode_{s}"
        m[f"Driver Marital Status {s}"] = f"Driver_MaritalStatusCode_{s}"
        m[f"Driver License Number {s}"] = f"Driver_LicenseNumberIdentifier_{s}"
        m[f"Driver License State {s}"] = f"Driver_LicensedStateOrProvinceCode_{s}"
        m[f"Driver Experience Years {s}"] = f"Driver_ExperienceYearCount_{s}"
        m[f"Driver Licensed Year {s}"] = f"Driver_LicensedYear_{s}"
        m[f"Driver Hired Date {s}"] = f"Driver_HiredDate_{s}"
        m[f"Driver Tax ID {s}"] = f"Driver_TaxIdentifier_{s}"
        m[f"Driver ID {s}"] = f"Driver_ProducerIdentifier_{s}"
        m[f"Driver Vehicle ID {s}"] = f"Driver_Vehicle_ProducerIdentifier_{s}"
        m[f"Driver Vehicle Use Pct {s}"] = f"Driver_Vehicle_UsePercent_{s}"
        m[f"Driver Other Car Code {s}"] = f"Driver_Coverage_DriverOtherCarCode_{s}"
        m[f"Driver Broadened No-Fault {s}"] = f"Driver_Coverage_BroadenedNoFaultCode_{s}"
        m[f"Driver City {s}"] = f"Driver_MailingAddress_CityName_{s}"
        m[f"Driver State {s}"] = f"Driver_MailingAddress_StateOrProvinceCode_{s}"
        m[f"Driver ZIP {s}"] = f"Driver_MailingAddress_PostalCode_{s}"
    # Accident/Conviction (single A slot on 127)
    m["Accident/Conviction Driver ID"] = "AccidentConviction_DriverProducerIdentifier_A"
    m["Violation Date"] = "AccidentConviction_TrafficViolationDate_A"
    m["Violation Description"] = "AccidentConviction_TrafficViolationDescription_A"
    m["Violation Year Count"] = "AccidentConviction_ViolationYearCount_A"
    m["Incident Location"] = "AccidentConviction_PlaceOfIncident_A"
    return m


_DRIVER = _driver_labels()


# Additional Interest (127) — slots A-D for name, A-B for address detail
def _addl_interest_labels() -> dict[str, str]:
    m: dict[str, str] = {}
    for s in "ABCD":
        m[f"Additional Interest Name {s}"] = f"AdditionalInterest_FullName_{s}"
    for s in "AB":
        m[f"Additional Interest Address {s}"] = f"AdditionalInterest_MailingAddress_LineOne_{s}"
        m[f"Additional Interest Address 2 {s}"] = f"AdditionalInterest_MailingAddress_LineTwo_{s}"
        m[f"Additional Interest City {s}"] = f"AdditionalInterest_MailingAddress_CityName_{s}"
        m[f"Additional Interest State {s}"] = f"AdditionalInterest_MailingAddress_StateOrProvinceCode_{s}"
        m[f"Additional Interest ZIP {s}"] = f"AdditionalInterest_MailingAddress_PostalCode_{s}"
        m[f"Lienholder Indicator {s}"] = f"AdditionalInterest_Interest_LienholderIndicator_{s}"
        m[f"Loss Payee Indicator {s}"] = f"AdditionalInterest_Interest_LossPayeeIndicator_{s}"
        m[f"Additional Insured Indicator {s}"] = f"AdditionalInterest_Interest_AdditionalInsuredIndicator_{s}"
        m[f"Lenders Loss Payable {s}"] = f"AdditionalInterest_Interest_LendersLossPayableIndicator_{s}"
        m[f"Employee Lessor {s}"] = f"AdditionalInterest_Interest_EmployeeAsLessorIndicator_{s}"
        m[f"Owner Indicator {s}"] = f"AdditionalInterest_Interest_OwnerIndicator_{s}"
        m[f"Registrant Indicator {s}"] = f"AdditionalInterest_Interest_RegistrantIndicator_{s}"
        m[f"Account Number {s}"] = f"AdditionalInterest_AccountNumberIdentifier_{s}"
        m[f"Certificate Required {s}"] = f"AdditionalInterest_CertificateRequiredIndicator_{s}"
        m[f"Interest Vehicle Ref {s}"] = f"AdditionalInterest_Item_VehicleProducerIdentifier_{s}"
        m[f"Interest Location Ref {s}"] = f"AdditionalInterest_Item_LocationProducerIdentifier_{s}"
    return m


_ADDL_INTEREST = _addl_interest_labels()


# Form 137 — coverages, symbols, hired/non-owned, trailer.
def _form137_labels() -> dict[str, str]:
    m: dict[str, str] = {}
    for s in "ABC":
        m[f"CSL Limit {s}"] = f"Vehicle_CombinedSingleLimit_LimitIndicator_{s}"
        m[f"BI Per Person Limit {s}"] = f"Vehicle_BodilyInjury_PerPersonLimitAmount_{s}"
        m[f"BI Per Accident Limit {s}"] = f"Vehicle_BodilyInjury_PerAccidentLimitAmount_{s}"
        m[f"PD Per Accident Limit {s}"] = f"Vehicle_PropertyDamage_PerAccidentLimitAmount_{s}"
        m[f"Med Pay Limit {s}"] = f"Vehicle_MedicalPayments_PerPersonLimitAmount_{s}"
        m[f"UM BI Per Person Limit {s}"] = f"Vehicle_UninsuredMotorists_BodilyInjuryPerPersonLimitAmount_{s}"
        m[f"UM BI Per Accident Limit {s}"] = f"Vehicle_UninsuredMotorists_BodilyInjuryPerAccidentLimitAmount_{s}"
        m[f"UM PD Per Accident Limit {s}"] = f"Vehicle_UninsuredMotorists_PropertyDamagePerAccidentLimit_{s}"
        m[f"Towing Limit {s}"] = f"Vehicle_TowingAndLabour_LimitAmount_{s}"
        m[f"Collision Deductible {s}"] = f"Vehicle_Collision_DeductibleAmount_{s}"
        m[f"Collision Deductible Waiver {s}"] = f"Vehicle_Collision_DeductibleWaiverIndicator_{s}"
        m[f"Comp Deductible {s}"] = f"Vehicle_Comprehensive_DeductibleAmount_{s}"
        m[f"SCL Deductible {s}"] = f"Vehicle_SpecifiedCauseOfLoss_DeductibleAmount_{s}"
        # Non-owned / hired
        m[f"Non-Owned Yes {s}"] = f"Vehicle_NonOwned_YesIndicator_{s}"
        m[f"Non-Owned No {s}"] = f"Vehicle_NonOwned_NoIndicator_{s}"
        m[f"Non-Owned Employee Indicator {s}"] = f"Vehicle_NonOwnedGroup_EmployeeIndicator_{s}"
        m[f"Non-Owned Employee Count {s}"] = f"Vehicle_NonOwnedGroup_EmployeeCount_{s}"
        m[f"Non-Owned Partner Indicator {s}"] = f"Vehicle_NonOwnedGroup_PartnerIndicator_{s}"
        m[f"Non-Owned Partner Count {s}"] = f"Vehicle_NonOwnedGroup_PartnerCount_{s}"
        m[f"Non-Owned Volunteer Indicator {s}"] = f"Vehicle_NonOwnedGroup_VolunteerIndicator_{s}"
        m[f"Non-Owned Volunteer Count {s}"] = f"Vehicle_NonOwnedGroup_VolunteerCount_{s}"
        m[f"Hired Yes {s}"] = f"Vehicle_HiredBorrowed_YesIndicator_{s}"
        m[f"Hired No {s}"] = f"Vehicle_HiredBorrowed_NoIndicator_{s}"
        m[f"Hired Cost {s}"] = f"Vehicle_HiredBorrowed_HiredCostAmount_{s}"
        m[f"Hired If Any Basis {s}"] = f"Vehicle_HiredBorrowed_IfAnyBasisIndicator_{s}"
        m[f"Hired PD Primary {s}"] = f"Vehicle_HiredBorrowed_PrimaryIndicator_{s}"
        m[f"Hired PD Secondary {s}"] = f"Vehicle_HiredBorrowed_SecondaryIndicator_{s}"
        m[f"Hired PD Vehicle Count {s}"] = f"Vehicle_HiredBorrowed_VehicleCount_{s}"
        m[f"Hired PD Day Count {s}"] = f"Vehicle_HiredBorrowed_DayCount_{s}"
    # BI Each Person A-H
    for s in string.ascii_uppercase[:8]:
        m[f"BI Each Person Indicator {s}"] = f"Vehicle_BodilyInjury_EachPersonLimitIndicator_{s}"
    # Symbols
    m["Symbol 1"] = "Vehicle_BusinessAutoSymbol_OneIndicator_A"
    m["Symbol 9"] = "Vehicle_BusinessAutoSymbol_NineIndicator_A"
    for s in string.ascii_uppercase[:8]:
        m[f"Symbol 2 {s}"] = f"Vehicle_BusinessAutoSymbol_TwoIndicator_{s}"
        m[f"Symbol 3 {s}"] = f"Vehicle_BusinessAutoSymbol_ThreeIndicator_{s}"
        m[f"Symbol 4 {s}"] = f"Vehicle_BusinessAutoSymbol_FourIndicator_{s}"
        m[f"Symbol 7 {s}"] = f"Vehicle_BusinessAutoSymbol_SevenIndicator_{s}"
        m[f"Symbol 8 {s}"] = f"Vehicle_BusinessAutoSymbol_EightIndicator_{s}"
        m[f"Other Symbol Indicator {s}"] = f"Vehicle_BusinessAutoSymbol_OtherSymbolIndicator_{s}"
        m[f"Other Symbol Code {s}"] = f"Vehicle_BusinessAutoSymbol_OtherSymbolCode_{s}"
    # Trailers A-F
    for s in "ABCDEF":
        m[f"Trailer Count {s}"] = f"Vehicle_TrailerCount_{s}"
        m[f"Trailer Radius {s}"] = f"Vehicle_TrailerRadius_{s}"
        m[f"Trailer Day Count {s}"] = f"Vehicle_TrailerDayCount_{s}"
        m[f"Trailer State {s}"] = f"Vehicle_TrailerStateOrProvinceCode_{s}"
    for s in "AB":
        m[f"Trailer Value {s}"] = f"Vehicle_TrailerValueAmount_{s}"
    for s in "BC":
        m[f"Trailer Collision Deductible {s}"] = f"Vehicle_TrailerCollisionDeductibleAmount_{s}"
    return m


_FORM_137 = _form137_labels()


# Form 163 — single-driver header labels (the FE only sends row-1 labels;
# rows 2+ come through as raw Text##[0] keys per form_163_layout).
_FORM_163: dict[str, str] = {
    "Driver Number": "Text15[0]",
    "Driver First Name": "Text16[0]",
    "Driver Middle Initial": "Text17[0]",
    "Driver Last Name": "Text18[0]",
    "Driver Address": "Text19[0]",
    "Driver City": "Text20[0]",
    "Driver State": "Text21[0]",
    "Driver ZIP": "Text22[0]",
    "Driver Sex": "Text23[0]",
    "Driver DOB": "Text24[0]",
    "Driver Years Licensed": "Text25[0]",
    "Driver Year Licensed": "Text26[0]",
    "Driver License Number": "Text27[0]",
    "Driver SSN": "Text28[0]",
    "Driver License State": "Text29[0]",
    "Driver Date Hired": "Text30[0]",
    "Driver Vehicle Number": "Text33[0]",
    "Driver Pct Use": "Text34[0]",
    "Driver Marital Status": "marital[0]",
}


# Final merged map — later dicts win on conflict, but all entries here are
# disjoint by design (common labels appear only in _COMMON, vehicle-slot labels
# only in _VEHICLE, etc.).
LABEL_TO_WIDGET: dict[str, str] = {
    **_COMMON,
    **_LEGAL_ENTITY,
    **_POLICY_125,
    **_BUSINESS_TYPE,
    **_LOSS,
    **_PRIOR,
    **_VEHICLE,
    **_DRIVER,
    **_ADDL_INTEREST,
    **_FORM_137,
    **_FORM_163,
}


# ---------------------------------------------------------------------------
# Public translation API
# ---------------------------------------------------------------------------


def translate_payload(values: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
    """Translate a FE-shaped dict into a widget-name keyed dict.

    Returns (translated, unknown_keys) where unknown_keys lists keys that
    neither matched a known label nor looked like a widget name. NullObject
    and other null sentinels are filtered out entirely.

    Widget-name keys pass through unchanged so FE can mix both conventions.
    """
    translated: dict[str, str] = {}
    unknown: list[str] = []
    for k, v in values.items():
        if _is_null(v):
            continue
        # Raw widget-name passthrough (Text##[0] or PascalCase_*).
        if "[0]" in k or ("_" in k and " " not in k and not k.startswith("_")):
            translated[k] = str(v)
            continue
        # Label lookup.
        widget = LABEL_TO_WIDGET.get(k)
        if widget:
            translated[widget] = str(v)
        else:
            unknown.append(k)
    return translated, unknown
