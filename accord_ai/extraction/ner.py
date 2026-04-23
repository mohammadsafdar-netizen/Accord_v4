"""spaCy NER pre-extraction and post-extraction layer (Phase A steps 5-6).

Ported from accord_ai_v3/extraction/ner.py with v4 schema-path adaptation.

Two roles:

PRE-EXTRACTION (``tag_entities`` + ``format_ner_hints``): Tags entities
in the user message before the LLM sees them. Classifies PERSON vs ORG
names; detects FEIN, VIN, phone, email, ZIP, website patterns; feeds
hints into the extraction prompt so the LLM starts aligned.

POST-EXTRACTION (``validate_extraction_with_ner``): Cross-checks the LLM's
extraction against NER classifications. Four fixes (v3 rules, v4 paths):

  1. ``contacts[0].full_name`` was assigned an ORG name → remove it.
  2. Contact name was NOT extracted but NER found a PERSON → inject it.
  3. ``business_name`` was NOT extracted (in delta AND current state) but
     NER found an ORG → inject it, with aggressive junk-filter.
  4. ``website`` not extracted but URL regex found one → inject it.

Optional spaCy dep: if spacy/en_core_web_sm aren't available, tag_entities
returns regex-only results (no PERSON/ORG/DATE). Downstream validator
calls degrade gracefully — no fixes applied, but no crash.

v3 schema paths translated:
  named_insured.business_name        → business_name
  named_insured.contact.full_name    → contacts[0].full_name
  named_insured.website              → website
"""
from __future__ import annotations

import re
import threading
from typing import Any, Dict, List, Optional

from accord_ai.logging_config import get_logger

_logger = get_logger("extraction.ner")


# ---------------------------------------------------------------------------
# Lazy singleton spaCy model
# ---------------------------------------------------------------------------

_nlp: Any = None
_nlp_lock = threading.Lock()


def _get_nlp() -> Optional[Any]:
    """Load spaCy model lazily. Returns None if spaCy isn't installed."""
    global _nlp
    if _nlp is None:
        with _nlp_lock:
            if _nlp is None:
                try:
                    import spacy   # type: ignore[import-not-found]
                    _nlp = spacy.load(
                        "en_core_web_sm",
                        disable=["parser", "lemmatizer"],
                    )
                    _logger.info("NER: loaded en_core_web_sm")
                except Exception as exc:
                    _logger.warning("NER: spaCy unavailable — %s", exc)
                    _nlp = False    # sentinel — don't retry
    return _nlp if _nlp is not False else None


# ---------------------------------------------------------------------------
# Pattern detectors (work without spaCy)
# ---------------------------------------------------------------------------

_PHONE_RE = re.compile(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_EIN_RE   = re.compile(r"\b(\d{2})[-–](\d{7})\b")
_VIN_RE   = re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b")
_ZIP_RE   = re.compile(r"\b\d{5}(?:-\d{4})?\b")

_TLD = (
    r"(?:com|org|net|io|co|ai|us|biz|info|gov|edu|mil|"
    r"uk|ca|de|fr|au|jp|me|tv|tech|solutions|services|agency|"
    r"app|dev|shop|store|group|pro|company|inc|llc|insurance|auto)"
)
_URL_RE = re.compile(
    r"(?<![\w@])"
    r"(?:https?://)?"
    r"(?:www\.)?"
    r"(?:[a-zA-Z0-9][a-zA-Z0-9-]{0,62}\.)+"
    + _TLD
    + r"(?:/[\w\-./?%&=]*)?"
    r"(?![\w@])"
)

_ORG_SUFFIXES = re.compile(
    r"\b(?:LLC|L\.L\.C|Inc\.?|Corp\.?|Corporation|Ltd\.?|Company|Co\.?|"
    r"LP|L\.P\.?|LLP|Group|Partners|Enterprise|Services|Solutions|"
    r"Logistics|Transport|Trucking|Hauling|Freight|Builders|"
    r"Construction|Plumbing|Electric|Landscaping|Distribution)\b",
    re.IGNORECASE,
)

_NAME_REJECT_WORDS = {
    "the", "a", "an", "my", "our", "your", "this", "that", "it",
    "insurance", "coverage", "policy", "premium", "auto", "commercial",
    "llc", "inc", "corp", "ltd", "co", "company", "enterprise", "group",
}


# ---------------------------------------------------------------------------
# tag_entities — pre-extraction hints
# ---------------------------------------------------------------------------

def tag_entities(text: str) -> Dict[str, List[str]]:
    """Run NER + regex on user message. Return detected entities.

    Keys: persons, orgs, phones, emails, eins, vins, zips, websites, dates.
    """
    result: Dict[str, List[str]] = {
        "persons":  [],
        "orgs":     [],
        "phones":   [],
        "emails":   [],
        "eins":     [],
        "vins":     [],
        "zips":     [],
        "websites": [],
        "dates":    [],
    }
    if not text:
        return result

    # Regex detectors (always run)
    result["phones"] = [m.group() for m in _PHONE_RE.finditer(text)]
    result["emails"] = [m.group() for m in _EMAIL_RE.finditer(text)]
    result["eins"]   = [f"{m.group(1)}-{m.group(2)}" for m in _EIN_RE.finditer(text)]
    result["vins"]   = [m.group() for m in _VIN_RE.finditer(text)]
    result["zips"]   = [m.group() for m in _ZIP_RE.finditer(text)]

    # Website — filter overlaps with email spans.
    email_spans = {(m.start(), m.end()) for m in _EMAIL_RE.finditer(text)}
    websites: List[str] = []
    for m in _URL_RE.finditer(text):
        if any(es[0] <= m.start() < es[1] for es in email_spans):
            continue
        url = m.group().lower().rstrip("/.,;")
        for prefix in ("https://", "http://"):
            if url.startswith(prefix):
                url = url[len(prefix):]
                break
        if url and url not in websites:
            websites.append(url)
    result["websites"] = websites

    # spaCy NER — optional, graceful degrade.
    nlp = _get_nlp()
    if nlp:
        try:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    name = ent.text.strip()
                    if _is_valid_person_name(name):
                        result["persons"].append(name)
                elif ent.label_ == "ORG":
                    result["orgs"].append(ent.text.strip())
                elif ent.label_ == "DATE":
                    result["dates"].append(ent.text.strip())
        except Exception as exc:
            _logger.warning("NER: spaCy doc processing failed — %s", exc)

    # Reclassify ORG-suffixed PERSONs as ORG ("Acme Trucking LLC" → ORG).
    reclassified: List[str] = []
    for person in list(result["persons"]):
        if _ORG_SUFFIXES.search(person):
            result["orgs"].append(person)
            reclassified.append(person)
    for name in reclassified:
        result["persons"].remove(name)

    return result


def _is_valid_person_name(text: str) -> bool:
    text = text.strip()
    if len(text) < 3 or len(text) > 50:
        return False
    parts = text.split()
    if len(parts) < 2:
        return False
    for part in parts:
        if part.lower().rstrip(".,") in _NAME_REJECT_WORDS:
            return False
    if re.search(r"\d", text):
        return False
    if _ORG_SUFFIXES.search(text):
        return False
    return True


# ---------------------------------------------------------------------------
# format_ner_hints — inject into extraction prompt
# ---------------------------------------------------------------------------

def format_ner_hints(tags: Dict[str, List[str]]) -> str:
    """Return NER hints formatted for the extraction prompt, or empty string.

    Defensive emission:

    * PERSON hints are suppressed on fleet-size inputs (>1 detected PERSON)
      to avoid nudging the LLM to write a driver's name into contacts[0].

    * ORG hints are passed through ``_looks_like_business`` — spaCy's
      default model (en_core_web_sm) false-positives heavily on entity-
      rich text, tagging vehicle makes, state-code lists, ID patterns,
      and addresses as ORG entities. Handing those to the LLM as
      "Detected ORGANIZATION names" actively hurts extraction quality
      (observed: multi-vehicle scenarios regressed when unfiltered ORG
      hints were emitted). We reuse the same filter the post-validator
      uses so both paths have consistent "what's a real business" logic.
    """
    hints: List[str] = []
    persons = tags.get("persons", [])
    if len(persons) == 1:
        hints.append(f"Detected PERSON names: {', '.join(persons)}")
    raw_orgs = tags.get("orgs", [])
    clean_orgs = [o for o in raw_orgs if _looks_like_business(o)]
    if clean_orgs:
        hints.append(f"Detected ORGANIZATION names: {', '.join(clean_orgs)}")
    if tags.get("eins"):
        hints.append(f"Detected FEIN/EIN: {', '.join(tags['eins'])}")
    if tags.get("phones"):
        hints.append(f"Detected phone numbers: {', '.join(tags['phones'])}")
    if tags.get("emails"):
        hints.append(f"Detected emails: {', '.join(tags['emails'])}")
    if tags.get("vins"):
        hints.append(f"Detected VINs: {', '.join(tags['vins'])}")
    if tags.get("websites"):
        hints.append(
            f"Detected website(s) → route to `website`: "
            f"{', '.join(tags['websites'])}"
        )
    if not hints:
        return ""
    return "═══ NER ENTITY HINTS ═══\n" + "\n".join(hints)


# ---------------------------------------------------------------------------
# Post-extraction validation — four v3 fixes adapted to v4's flat schema
# ---------------------------------------------------------------------------

# Junk-ORG filters — tokens/patterns that are never real business names.
_BARE_REJECT = {
    "ein", "fein", "dba", "ssn", "vin", "dot", "mc", "cdl", "dl",
    "naics", "sic", "bop", "gl", "cgl", "wc", "ca", "cp", "do",
    "llc", "inc", "corp", "llp", "lp", "dot#", "mc#",
    "acord", "cert", "coi", "eff", "exp",
    "pdf", "csv", "xlsx", "doc", "docx",
    "ny", "tx", "ca", "fl", "il", "oh", "pa", "ga",
}
_VEHICLE_MAKES = {
    "ford", "chev", "chevy", "chevrolet", "gmc", "dodge", "ram", "jeep",
    "toyota", "honda", "nissan", "mazda", "subaru", "hyundai", "kia",
    "bmw", "mercedes", "mercedes-benz", "audi", "volkswagen", "vw",
    "kenworth", "peterbilt", "volvo", "mack", "freightliner",
    "international", "navistar", "isuzu", "hino", "western", "sterling",
    "autocar",
}
_JARGON_TOKENS = {
    "csl", "cov", "coverage", "bi", "pd", "um", "uim", "acv", "rcv",
    "limit", "premium", "deductible", "aggregate", "occurrence",
    "umbrella", "excess", "medpay", "liability", "comp",
    "collision", "comprehensive", "towing", "rental",
    "license", "licensed", "plate", "registration", "registered",
    "cdl", "dl", "dob", "ssn", "vin", "ein", "fein", "mc", "dot",
    "naics", "sic", "nacis",
    "policy", "claim", "loss", "insured", "applicant", "named",
    "endorsement", "schedule", "form", "acord", "cert", "coi",
    "driver", "drivers", "vehicle", "vehicles", "trailer", "trailers",
    "fleet", "operator", "owner",
}
_STATE_SET = {
    "al","ak","az","ar","ca","co","ct","de","fl","ga","hi","id",
    "il","in","ia","ks","ky","la","me","md","ma","mi","mn","ms",
    "mo","mt","ne","nv","nh","nj","nm","ny","nc","nd","oh","ok",
    "or","pa","ri","sc","sd","tn","tx","ut","vt","va","wa","wv",
    "wi","wy","dc",
}
_ID_PATTERN = re.compile(
    r"\b(?:cdl|dl|vin|mc|dot|ein|naics|sic|ssn|fein)(?:[#\-:]|[\s]*\d)",
    re.IGNORECASE,
)
_ADDR_PATTERN = re.compile(
    r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b|\b\d{5}(?:-\d{4})?\b",
)


def _looks_like_business(name: str) -> bool:
    """Filter candidate ORG names down to plausible businesses.

    Rejects: bare acronyms, vehicle makes, insurance jargon, address-
    shaped strings, state-code lists, ID-number patterns, pure-code-
    looking strings.
    """
    n = name.strip()
    if len(n) < 3 or len(n) > 80:
        return False
    n_clean = n.lower().strip(".,#:- ")
    if n_clean in _BARE_REJECT:
        return False
    if n_clean in _VEHICLE_MAKES:
        return False
    first_token = n_clean.split()[0] if n_clean else ""
    if first_token in _VEHICLE_MAKES:
        return False
    tokens = {t.strip(".,:#-") for t in n_clean.split()}
    if tokens & _JARGON_TOKENS:
        return False
    digit_ratio = sum(1 for c in n if c.isdigit()) / max(len(n), 1)
    if digit_ratio > 0.3:
        return False
    words = n.split()
    has_legal_suffix = bool(_ORG_SUFFIXES.search(n))
    if len(words) < 2 and not has_legal_suffix:
        return False
    if _ID_PATTERN.search(n):
        return False
    if _ADDR_PATTERN.search(n):
        return False
    if tokens and all(t in _STATE_SET for t in tokens):
        return False
    if re.fullmatch(r"[A-Z0-9\-\s]{3,}", n) and not _ORG_SUFFIXES.search(n):
        if sum(1 for c in n if c.isalpha() and c.isupper()) > len(n) * 0.4:
            return False
    return True


def validate_extraction_with_ner(
    delta: Dict[str, Any],
    ner_tags: Dict[str, List[str]],
    current_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Apply four v3 NER fixes to an LLM extraction delta. Mutates + returns.

    All four fixes use v4's flat schema paths (business_name at root,
    contacts[0].*, website at root). On non-dict input, returns unchanged.
    """
    if not isinstance(delta, dict):
        return delta

    delta_biz = delta.get("business_name", "") or ""
    session_biz = ""
    if isinstance(current_state, dict):
        session_biz = current_state.get("business_name", "") or ""

    detected_persons = set(p.lower() for p in ner_tags.get("persons", []))
    detected_orgs_lower = set(o.lower() for o in ner_tags.get("orgs", []))

    # Work on contacts[0] if it exists (v4's contacts list).
    contacts = delta.get("contacts")
    contact = contacts[0] if isinstance(contacts, list) and contacts else None
    if not isinstance(contact, dict):
        contact = None

    # --- Fix 1: contact name classified as ORG → remove ---
    if contact is not None:
        contact_name = contact.get("full_name", "") or ""
        if contact_name:
            contact_lower = contact_name.strip().lower()
            is_org = (
                contact_lower in detected_orgs_lower
                and contact_lower not in detected_persons
            )
            has_org_suffix = bool(_ORG_SUFFIXES.search(contact_name))
            if is_org or has_org_suffix:
                _logger.info(
                    "NER: removing contacts[0].full_name=%r "
                    "(ORG-like; is_org=%s suffix=%s)",
                    contact_name, is_org, has_org_suffix,
                )
                contact.pop("full_name", None)

    # --- Fix 2: no contact name but NER found a PERSON → inject ---
    # Skipped on multi-person inputs (>1 detected PERSON): on fleet
    # intakes a turn like "drivers are Alice Jones, Bob Smith, Carol
    # Davis" surfaces 3+ PERSON entities, and injecting one as the
    # contact overwrites the business contact with a driver's name.
    # Live-measured regression: multi-five-vehicle-fleet dropped from
    # F1=0.324 to F1=0.121 when fix 2 fired on multi-driver messages.
    # The harness rules tell the LLM to route drivers to
    # lob_details.drivers; don't fight that with a contact-level inject.
    if len(ner_tags.get("persons", [])) == 1:
        best_person = ner_tags["persons"][0]
        if best_person and best_person.lower() != delta_biz.strip().lower():
            if contact is None:
                contact = {}
                contacts = contacts or []
                contacts.insert(0, contact)
                delta["contacts"] = contacts
            if not contact.get("full_name"):
                contact["full_name"] = best_person
                _logger.info(
                    "NER: injected contacts[0].full_name=%r from PERSON entity",
                    best_person,
                )

    # --- Fix 3: no business_name anywhere but NER found an ORG → inject ---
    if not delta_biz and not session_biz and ner_tags.get("orgs"):
        candidates = [
            o for o in ner_tags["orgs"] if _looks_like_business(o)
        ]
        best_org = max(candidates, key=len, default=None) if candidates else None
        if best_org:
            delta["business_name"] = best_org
            _logger.info(
                "NER: injected business_name=%r from ORG entity", best_org,
            )

    # --- Fix 4: no website but URL regex found one → inject ---
    detected_websites = ner_tags.get("websites", [])
    if detected_websites and not delta.get("website"):
        biz_name = delta.get("business_name", "") or session_biz
        best = detected_websites[0]
        biz_key = re.sub(r"[^a-z0-9]", "", biz_name.lower()) if biz_name else ""
        if biz_key and len(biz_key) >= 6:
            for w in detected_websites:
                wk = re.sub(r"[^a-z0-9]", "", w.lower())
                if biz_key[:6] in wk:
                    best = w
                    break
        delta["website"] = best
        _logger.info("NER: injected website=%r from URL regex", best)

    return delta
