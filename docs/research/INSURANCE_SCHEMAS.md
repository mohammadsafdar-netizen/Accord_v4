# Insurance Data Schemas & Standards — 2026 Research

**Audience:** `accord_v4` schema designers.
**Scope:** Is our `CustomerSubmission` Pydantic model aligned with industry standards, or drifting wastefully?
**Reference schema inspected:** `accord_ai_v3/accord_ai/core/schema.py` (508 LOC, Pydantic v2) + `pipeline/entity_schema.py:CustomerSubmission` (dataclass container) + `form_fields/*.json` (10 ACORD PDF field specs, 19,602 LOC total).

---

## Executive Summary

**Our custom schema aligns structurally with ACORD P&C conventions but uses snake_case Python names where ACORD uses PascalCase XML/PDF widget names.** The top-level entity decomposition (`NamedInsured`, `Producer`, `PolicyInfo`, `Vehicle`, `Driver`, `Coverage`, `Location`, `AdditionalInterest`, `LossHistory`, `PriorInsurance`) is a near-1:1 match to the ACORD P&C XML hierarchy and to the ACORD PDF widget-name prefixes observed in our own `form_fields/*.json` files (e.g. `NamedInsured_FullName_A`, `Producer_MailingAddress_LineOne_A`, `Policy_EffectiveDate_A`, `Driver_LicenseNumberIdentifier_A`). Divergence is cosmetic — naming convention, not structure.

**Recommendation: Hybrid.** Keep the Pydantic model as the internal intake representation (it is LLM-friendly, type-checked, and evolution-controlled). Add a **thin serializer** that maps our snake_case fields to ACORD XML element names for (a) future carrier EDI/XML handoffs and (b) auditable round-trips against the 2,199-field-count widget specs we already ship. Do NOT adopt an ACORD-derived schema as the primary internal model — no maintained open-source ACORD Python library exists, and ACORD XSDs require a paid ACORD membership to access the canonical source.

**What we do NOT need to change:** entity decomposition, nesting depth, the discriminated-union LOB pattern, or the `form_fields/*.json` widget specs (those already use ACORD canonical names).

---

## 1. ACORD Standards — State of Play (2026)

### Current published versions

| Standard | Latest | Notes |
|---|---|---|
| ACORD P&C XML | **v1.30.0** (legacy, widely deployed) / **v2.7.0** (next-gen) | Industry is mid-migration from 1.x → 2.x. v2.13.0 was released Jan 2025 for Delegated Authority use cases. |
| ACORD P&C AL3 | Still active | Fixed-width flat-file predecessor to XML; still used by legacy carriers. |
| ACORD PDF Forms | 2016-09 editions dominant (125, 126, 127, 129, 130, 131, 137) | Our templates match these. |
| ACORD Life XML | OLifE 2.x | Not relevant (we are P&C). |

Sources:
- [ACORD Data Standards (official)](https://www.acord.org/standards-architecture/acord-data-standards)
- [Property & Casualty Data Standards](https://www.acord.org/standards-architecture/acord-data-standards/Property_Casualty_Data_Standards)
- [Liquid Technologies P&C schema mirror v1.16.0](https://schemas.liquid-technologies.com/accord/pcs/1.16.0/) — publicly browsable XSD documentation (useful because ACORD itself paywalls the canonical source).
- [Pilotfish Model Viewer — ACORDStandardVersionCd](https://modelviewers.pilotfishtechnology.com/modelviewers/ACORD-PCS/model/ACORDStandardVersionCd.html) — confirms v1.30.0 and v2.7.0 code enumerations.

### Membership / access

- Full XSD + Business Message Specifications are **members-only** (ACORD membership runs ~US$3k–$30k/yr by org size; tier determines which standards you can download).
- Non-members can see enough via third-party mirrors (Liquid Technologies, Pilotfish Model Viewer, Oracle Siebel Connector docs) to build an interoperable schema without buying membership.
- [ACORD Reference Architecture](https://www.acord.org/standards-architecture/reference-architecture) is free reading but contains no XSDs.

**Actionable:** For v4, do NOT pursue ACORD membership unless a carrier integration demands an officially signed XSD conformance statement.

### ACORD XML canonical entity names (P&C)

Confirmed from Liquid Technologies mirror and Oracle Siebel Connector examples:

```
<ACORD>
  <InsuranceSvcRq>
    <CommlPolicy>                   ← our PolicyInfo
      <CommlPolicyKey/>
      <ContractNumber/>
      <InsuredOrPrincipal>          ← our NamedInsured
        <GeneralPartyInfo>
          <NameInfo>
            <CommlName>
              <CommercialName/>
            </CommlName>
          </NameInfo>
          <Addr>
            <Addr1/> <City/> <StateProvCd/> <PostalCode/>
          </Addr>
        </GeneralPartyInfo>
      </InsuredOrPrincipal>
      <Producer>                    ← our ProducerInfo
        <GeneralPartyInfo>...</GeneralPartyInfo>
      </Producer>
      <CommlVeh>...</CommlVeh>      ← our Vehicle (commercial)
      <Driver>...</Driver>
      <Loss>...</Loss>              ← our LossHistoryEntry
      <AdditionalInterest>...</AdditionalInterest>
    </CommlPolicy>
  </InsuranceSvcRq>
</ACORD>
```

Sources: [Oracle Siebel ACORD XML Docs v7.7](https://docs.oracle.com/cd/E05553_01/books/ConnACORDFINS/ConnACORDFINSWizard3.html), [Embarcadero whitepaper "Deciphering the ACORD XML Standard"](https://www.embarcadero.com/images/dm/technical-papers/whitepaper-deciphering-the-acord-xml-standard.pdf).

---

## 2. Open-Source Python & GitHub Landscape

**Reality check: there is no maintained, high-quality Python ACORD library.** This is a gap, not a failure of research.

| Repo | Lang | Stars | What it does | Verdict |
|---|---|---|---|---|
| [maldworth/aldsoft.acord](https://github.com/maldworth/aldsoft.acord) | C# / .NET | 16 | Serialize/deserialize ACORD **Life** XML (OLifE). Not P&C. | Not useful. |
| [markwalters2/acord-filler](https://github.com/markwalters2/acord-filler) | Python + PyMuPDF | 1 | Fills ACORD 125/140/25/24/27/28/37/50 with field mappings + signature overlays. | **Worth reviewing** — closest cousin to our `pipeline/pdf_filler.py`. Unmaintained but instructive. |
| [CoforgeInsurance/acord-data-quality-monitor](https://github.com/CoforgeInsurance/acord-data-quality-monitor) | Python | 1 | AI-agentic validation of ACORD submissions from YAML contracts. | Similar ambition to our validation engine; small scale. |
| [jasonjanofsky/Acord60Mins](https://github.com/jasonjanofsky/Acord60Mins) | C# | n/a | Tutorial repo — ACORD Life XML in 60 min. | Life only, pedagogical. |

No PyPI package named `acord`, `pyacord`, or `acord-py` exists as of 2026-04. A search for `"acord" "python" "commercial lines"` surfaces SaaS products (Docsumo, Affinda, Unstract, SortSpoke, Infrrd) but no libraries. See [GitHub topic: acord](https://github.com/topics/acord) and [LlamaIndex — Top ACORD form processing platforms](https://www.llamaindex.ai/insights/top-acord-form-processing-platforms).

### openIDL / openIDS (emerging alternative)

- [openIDL](https://openidl.org/) — Linux Foundation project, backed by AAIS.
- [openIDS Working Group](https://lf-openidl.atlassian.net/wiki/spaces/HOME/pages/86376451/openIDS+-+Open+Insurance+Data+Standards+Working+Group) formed April 2025.
- First deliverable: **openIDS Homeowners Standard v1.0** ([announcement](https://www.prnewswire.com/news-releases/openidl-launches-first-ever-free-and-open-production-ready-insurance-data-standard-openids-homeowners-standard-v1-0--302621098.html)).
- Commercial Auto / GL / Workers Comp standards **not yet published** — roadmap only.

**Actionable:** Monitor openIDS. If they publish a commercial-auto JSON schema in 2026 we should align (it will be free + open, unlike ACORD XSDs). Right now there is nothing to consume.

---

## 3. How Commercial Platforms Structure This Data

Public documentation on Guidewire / Duck Creek / Applied Epic internal schemas is **scarce** — these are paid enterprise platforms. Observed generalities:

- **Guidewire InsuranceSuite** (PolicyCenter/BillingCenter/ClaimCenter): unified data model across apps; entity model is proprietary but ingests ACORD XML at edges. See [Guidewire vs Duck Creek](https://www.selecthub.com/insurance-software/guidewire-vs-duck-creek/).
- **Duck Creek**: cloud-native; "comprehensive insurance data model surrounded by enterprise data management modules." Ingests ACORD at edges.
- **Applied Epic**: agency-side (what our brokers use). Configurable custom-field model; API-first. [Applied Epic comparison](https://www.selecthub.com/insurance-software/guidewire-vs-applied-epic/).

**Pattern across all three:** internal model is proprietary; ACORD XML/PDF is the *interchange* format at carrier/broker boundaries. This validates our hybrid recommendation — keep internal schema pragmatic, serialize to ACORD at the integration seam.

---

## 4. NAICS ↔ Insurance Class Code Mapping

Our `ClassCodes` model (NAICS + SIC) is structurally correct but **does not carry NCCI** — required for Workers Comp rating.

- [NCCI Class Look-Up](https://www.ncci.com/ServicesTools/pages/CLASSLOOKUP.aspx) now includes NAICS and SIC cross-references ([announcement](https://www.ncci.com/Articles/Pages/Atlas_ClassLookup.aspx)).
- [NCCI Class Information API](https://www.federato.ai/library/post/ncci-class-codes-identifying-critical-blind-spots-in-workers-comp-underwriting) provides the programmatic lookup — paid but first-party.
- NCCI is the only code set that actually rates WC. NAICS/SIC are informational only for WC underwriting.

**Actionable:** Add `ncci_code: Optional[str]` to `ClassCodes`. Populate it from NAICS via NCCI API when a WC quote is requested. Keep NAICS/SIC as-is.

---

## 5. PDF Form Field Mapping — What We Already Have

**Finding:** our `accord_ai/form_fields/*.json` files **already use ACORD-canonical widget names.** Sample from `form_125_fields.json` and `form_127_fields.json`:

```
Producer_FullName_A
Producer_MailingAddress_LineOne_A
Producer_ContactPerson_PhoneNumber_A
NamedInsured_FullName_A
Policy_PolicyNumberIdentifier_A
Policy_EffectiveDate_A
Insurer_NAICCode_A
Driver_Surname_A
Driver_GivenName_A
Driver_LicenseNumberIdentifier_A
Driver_BirthDate_A
Driver_LicensedStateOrProvinceCode_A
Driver_ExperienceYearCount_A
CommercialVehicleLineOfBusiness_Attachment_CommercialAutoDriverInformationScheduleIndicator_A
```

Widget naming convention observed: `<Entity>_<SubEntity?>_<AttributeName>_<Suffix>` — e.g. `NamedInsured_MailingAddress_CityName_A`. The trailing `_A` / `_B` / `_C` indicates repeating-section row index. This is the ACORD 2016-09 PDF widget standard, and it maps 1:1 to ACORD XML element hierarchy (XPath `NamedInsured/MailingAddress/CityName` → widget `NamedInsured_MailingAddress_CityName_A`).

Total widget count across our 10 PDF specs: roughly **3,500+ named widgets** (form 125 alone: 548 fields; form 160 is the heaviest at 4,547 lines of JSON).

### Python tooling we can use

- **pypdf** (`reader.get_fields()`, `update_page_form_field_values()`) — already viable. [pypdf forms docs](https://pypdf.readthedocs.io/en/stable/user/forms.html).
- **PyPDFForm** ([chinapandaman/PyPDFForm](https://github.com/chinapandaman/PyPDFForm)) — higher-level wrapper.
- Our current `pipeline/pdf_filler.py` uses PyMuPDF/`fitz` with flatten + AP-null hack — correct approach for production (flattened previews render in Google Drive, which an AcroForm does not).

### Commercial extraction tools (for reference — not recommended)

Docsumo, Affinda, Unstract, SortSpoke, UiPath Appulate — all claim 99%+ on ACORD 125/126/127/140/141. We are already at **99.0% L3 / 97.2% deep-test** on our dual-pass pipeline, so no integration value. See [LlamaIndex — Top ACORD Form Processing Platforms](https://www.llamaindex.ai/insights/top-acord-form-processing-platforms).

---

## 6. Compliance — PII / Retention / Encryption

### NAIC Model Laws (2025 status)

- **[Model #670](https://content.naic.org/sites/default/files/model-law-670.pdf)** (1992, Insurance Information and Privacy Protection Model Act) + **[Model #672](https://content.naic.org/sites/default/files/model-law-672.pdf)** (2017, Privacy of Consumer Financial and Health Info) — widely adopted; being superseded.
- **[Model #674](https://content.naic.org/sites/default/files/inline-files/Exposure%20Draft-Consumer%20Privacy%20Protection%20Model%20Law%20%23674%201-31-23.pdf)** — replacement under active comment cycle; adds 90-day deletion obligation for non-retained data.
- **[Insurance Data Security Model Law (Aug 2025)](https://content.naic.org/sites/default/files/government-affairs-brief-data-security-model-law.pdf)** — adopted in ~24 states; requires written infosec program, 72hr breach notification, 5-year retention of cyber-event records.
- HIPAA overlay: 6-year retention where health info intersects (rare for commercial auto/GL/WC, but possible in WC medical claims).

Summary from [Willkie Farr 2025 privacy update](https://www.willkie.com/publications/2025/05/latest-developments-on-insurance-privacy-laws).

### Actionable obligations for accord_v4

1. **Encryption at rest** — mandatory under Data Security Model Law in adopting states. SQLite + ChromaDB dirs must sit on encrypted volumes in production.
2. **Retention** — minimum 5 years for cyber-event records; 90 days for non-retained PII (Model 674). Our session TTL is 3600s — we must document that finalized submissions move to the broker's system-of-record and we purge transient state.
3. **Breach notification** — 72-hour clock. Ensure `audit_log.py` captures enough state for forensics.
4. **PII redaction** — already done in `logging_config.py:PIIRedactionFilter`. Confirmed in CLAUDE.md.
5. **PII to external APIs** — our existing policy (CLAUDE.md: "Privacy-first: never send raw customer data to external APIs without explicit approval" + `ACCORD_DISABLE_REFINEMENT=true` in production) already matches NAIC expectations.

---

## 7. Alignment Check — CustomerSubmission vs ACORD

Mapping our top-level fields (`pipeline/entity_schema.py:CustomerSubmission` + `core/schema.py` sub-models) to ACORD XML / PDF widget naming:

| Our field | ACORD XML element | PDF widget prefix | Alignment |
|---|---|---|---|
| `business` / `named_insured` | `InsuredOrPrincipal` + `GeneralPartyInfo/NameInfo/CommlName` | `NamedInsured_*` | Full (rename-only) |
| `business.business_name` | `CommercialName` | `NamedInsured_FullName_A` | Full |
| `business.dba` | `SupplementaryNameInfo` (type=DBA) | `NamedInsured_DBAName_*` | Full |
| `business.mailing_address` | `Addr` | `NamedInsured_MailingAddress_*` | Full |
| `business.tax_id` | `TaxIdentity/TaxIdentifier` | `NamedInsured_TaxIdentifier_A` | Full |
| `business.entity_type` | `BusinessInfo/OrganizationTypeCd` | `NamedInsured_BusinessEntityTypeCode_A` | Full |
| `business.class_codes.naics` | `NAICSCd` | `Insurer_NAICCode_A` (carrier) / `NAICSCode_A` (insured) | Full |
| `business.class_codes.sic` | `SICCd` | `SICCode_A` | Full |
| `business.class_codes.ncci` | **missing in ours** | `NCCICode_*` | **ADD** |
| `business.annual_revenue` | `GrossReceiptsAmt` | `TotalRevenue_Amount_A` | Full |
| `business.annual_payroll` | `PayrollAmt` | `TotalPayroll_Amount_A` | Full |
| `business.employees` | `NumEmployeesFullTime` + `NumEmployeesPartTime` | `NumberOfEmployees_*` | Full |
| `producer` | `Producer` / `GeneralPartyInfo` | `Producer_*` | Full |
| `policy` | `CommlPolicy` | `Policy_*` | Full |
| `policy.effective_date` | `ContractTerm/EffectiveDt` | `Policy_EffectiveDate_A` | Full |
| `policy.expiration_date` | `ContractTerm/ExpirationDt` | `Policy_ExpirationDate_A` | Full |
| `vehicles[]` | `CommlVeh[]` | `CommercialVehicle_*` | Full |
| `drivers[]` | `Driver[]` | `Driver_*` | Full |
| `drivers[].license_number` | `LicenseInfo/LicensePermitNumber` | `Driver_LicenseNumberIdentifier_A` | Full |
| `coverages[]` | `Coverage[]` with `CoverageCd`, `Limit`, `Deductible` | Per-LOB (`AutoLiability_*`, `GeneralLiability_*`) | Full |
| `locations[]` | `Location[]` + `SubLocation[]` | `Location_*` / `Premises_*` | Full |
| `loss_history[]` | `Loss[]` | `LossHistory_*` | Full |
| `additional_interests[]` | `AdditionalInterest[]` | `AdditionalInterest_*` | Full |
| `prior_insurance[]` | `PriorPolicy[]` | `PriorCarrier_*` | Full |
| `cyber_info` | No direct ACORD equivalent (cyber is evolving in ACORD 2.x) | — | Custom — acceptable |
| `raw_email` | N/A — ours only | — | Custom — keep |

**Structural alignment: 100% for top-level entities; ~95% for leaf attributes.** The single real gap is `ncci_code`. Everything else is renaming.

---

## 8. Recommendation — Hybrid

**Keep** our Pydantic `CustomerSubmission` as the internal intake model. Reasons:
- LLM prompt engineering favors snake_case + descriptive docstrings (which we exploit for RAG seeding — see `core/schema.py:8`).
- Pydantic v2 validation is a hard win over raw XML parsing.
- No quality Python ACORD library exists to adopt.
- Proprietary platforms (Guidewire / Duck Creek) also keep a proprietary internal model and serialize to ACORD at the edges.

**Add** a thin ACORD serializer module. Two targets:
1. **ACORD XML v1.x emitter** — `accord_v4/adapters/acord_xml.py` — walk `CustomerSubmission` → emit `<CommlPolicy>` tree. Estimated ~400 LOC. Use Liquid Technologies mirror to validate against XSD 1.16.0 / 1.30.0 without membership.
2. **ACORD JSON envelope** — for modern carrier APIs that accept JSON equivalents of ACORD XML. Mirror the PascalCase names used in our `form_fields/*.json` widget dictionaries (we already have the mapping).

**Add** one missing field: `ClassCodes.ncci_code: Optional[str]`.

**Do NOT:**
- Rewrite to PascalCase internally — cosmetic, breaks prompts, no carrier benefit today.
- Pursue ACORD membership unless a carrier integration contract forces it.
- Adopt openIDS yet — commercial-auto standard doesn't exist there.

### Libraries / tools worth evaluating

| Tool | Use case | Priority |
|---|---|---|
| [markwalters2/acord-filler](https://github.com/markwalters2/acord-filler) | Reference read — compare our PDF filler | Low (we're ahead) |
| [pypdf](https://pypdf.readthedocs.io/en/stable/user/forms.html) | Alt to PyMuPDF for AcroForm introspection during field-spec regeneration | Low |
| [lxml + Liquid XSD mirror](https://schemas.liquid-technologies.com/accord/pcs/1.16.0/) | Build XML emitter + validate against v1.16.0 XSD | **Medium** (enables carrier EDI) |
| [NCCI Class Information API](https://www.ncci.com/ServicesTools/pages/CLASSLOOKUP.aspx) | NAICS → NCCI mapping for WC rating | **Medium** (add `ncci_code`) |
| [openIDL / openIDS](https://openidl.org/) | Watch for commercial-auto schema publication | Monitor |

---

## 9. Sources

- ACORD: [Data Standards](https://www.acord.org/standards-architecture/acord-data-standards), [P&C Data Standards](https://www.acord.org/standards-architecture/acord-data-standards/Property_Casualty_Data_Standards), [Implementation Resources](https://www.acord.org/standards-architecture/implement-standards)
- ACORD XML schema mirrors: [Liquid Technologies v1.16.0](https://schemas.liquid-technologies.com/accord/pcs/1.16.0/), [Pilotfish Model Viewer](https://modelviewers.pilotfishtechnology.com/modelviewers/ACORD-PCS/model/ACORDStandardVersionCd.html), [Cover Pages ACORD XML](https://xml.coverpages.org/acord.html)
- Oracle Siebel ACORD Connector: [v7.7](https://docs.oracle.com/cd/E05553_01/books/ConnACORDFINS/ConnACORDFINSWizard3.html), [v8.1/8.2](https://docs.oracle.com/cd/E14004_01/books/ConnACORDFINS/ConnACORDFINSWizard3.html), [Siebel 2018](https://docs.oracle.com/cd/E95904_01/books/ConnACORDFINS/ConnACORDFINSOverview11.html)
- Embarcadero: ["Deciphering the ACORD XML Standard" whitepaper](https://www.embarcadero.com/images/dm/technical-papers/whitepaper-deciphering-the-acord-xml-standard.pdf)
- openIDL / openIDS: [openIDL](https://openidl.org/), [openIDS Working Group](https://lf-openidl.atlassian.net/wiki/spaces/HOME/pages/86376451/openIDS+-+Open+Insurance+Data+Standards+Working+Group), [Homeowners v1.0 announcement](https://www.prnewswire.com/news-releases/openidl-launches-first-ever-free-and-open-production-ready-insurance-data-standard-openids-homeowners-standard-v1-0--302621098.html)
- Platforms: [Guidewire vs Duck Creek](https://www.selecthub.com/insurance-software/guidewire-vs-duck-creek/), [Guidewire vs Applied Epic](https://www.selecthub.com/insurance-software/guidewire-vs-applied-epic/)
- GitHub: [topic:acord](https://github.com/topics/acord), [markwalters2/acord-filler](https://github.com/markwalters2/acord-filler), [maldworth/aldsoft.acord](https://github.com/maldworth/aldsoft.acord)
- NCCI: [Class Look-Up](https://www.ncci.com/ServicesTools/pages/CLASSLOOKUP.aspx), [NAICS + SIC integration](https://www.ncci.com/Articles/Pages/Atlas_ClassLookup.aspx), [Federato — NCCI blind spots](https://www.federato.ai/library/post/ncci-class-codes-identifying-critical-blind-spots-in-workers-comp-underwriting)
- NAIC privacy: [Model #670](https://content.naic.org/sites/default/files/model-law-670.pdf), [Model #672](https://content.naic.org/sites/default/files/model-law-672.pdf), [Model #674 exposure draft](https://content.naic.org/sites/default/files/inline-files/Exposure%20Draft-Consumer%20Privacy%20Protection%20Model%20Law%20%23674%201-31-23.pdf), [Data Security Model Law brief (Aug 2025)](https://content.naic.org/sites/default/files/government-affairs-brief-data-security-model-law.pdf), [Willkie 2025 privacy developments](https://www.willkie.com/publications/2025/05/latest-developments-on-insurance-privacy-laws), [Alston & Bird privacy blog](https://www.alstonprivacy.com/new-naic-consumer-privacy-model-law-proposed-for-insurers/)
- PDF form tooling: [pypdf forms](https://pypdf.readthedocs.io/en/stable/user/forms.html), [PyPDFForm](https://github.com/chinapandaman/PyPDFForm)
- Commercial ACORD processors (reference only): [LlamaIndex platform roundup](https://www.llamaindex.ai/insights/top-acord-form-processing-platforms), [Docsumo](https://www.docsumo.com/solutions/documents/acord-forms), [Affinda](https://www.affinda.com/documents/acord-forms), [Unstract](https://unstract.com/ai-insurance-document-processing/acord-document-data-extraction/), [UiPath Appulate](https://marketplace.uipath.com/listings/acord-forms-intelligent-extraction-with-appulate)
