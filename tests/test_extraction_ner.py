"""Phase A steps 5+6 — NER pre-tag + post-validator (unit tests)."""
from __future__ import annotations

from accord_ai.extraction.ner import (
    _looks_like_business,
    format_ner_hints,
    tag_entities,
    validate_extraction_with_ner,
)


# ---------------------------------------------------------------------------
# tag_entities — regex-only paths always work; spaCy paths degrade gracefully
# ---------------------------------------------------------------------------

def test_tag_entities_empty_returns_empty_result():
    r = tag_entities("")
    assert r["persons"] == [] and r["orgs"] == []
    assert r["eins"] == [] and r["vins"] == []


def test_tag_entities_detects_phone():
    r = tag_entities("call me at 512-555-0100")
    assert "512-555-0100" in r["phones"]


def test_tag_entities_detects_phone_parens_format():
    r = tag_entities("phone: (512) 555-0100")
    assert any("512" in p for p in r["phones"])


def test_tag_entities_detects_email():
    r = tag_entities("write to ops@acme.test")
    assert "ops@acme.test" in r["emails"]


def test_tag_entities_detects_ein():
    r = tag_entities("FEIN is 12-3456789")
    assert "12-3456789" in r["eins"]


def test_tag_entities_detects_vin():
    vin = "1FT7W2BT4NED38571"
    r = tag_entities(f"VIN {vin}")
    assert vin in r["vins"]


def test_tag_entities_detects_zip():
    r = tag_entities("Austin TX 78701")
    assert "78701" in r["zips"]


def test_tag_entities_detects_website():
    r = tag_entities("our site is acme.com")
    assert "acme.com" in r["websites"]


def test_tag_entities_website_excludes_email_domain():
    """'joe@acme.com' — 'acme.com' is the email domain, not a separate site."""
    r = tag_entities("email joe@acme.com")
    assert "acme.com" not in r["websites"]


def test_tag_entities_reclassifies_person_with_org_suffix():
    """'Acme Trucking LLC' — spaCy may tag as PERSON; we reclassify to ORG."""
    r = tag_entities("We are Acme Trucking LLC")
    # Regardless of spaCy's tagging, an ORG-suffixed name ends up in orgs
    # and not persons after reclassification.
    assert all("llc" not in p.lower() for p in r["persons"])


# ---------------------------------------------------------------------------
# format_ner_hints
# ---------------------------------------------------------------------------

def test_format_ner_hints_empty_when_no_tags():
    assert format_ner_hints({
        "persons": [], "orgs": [], "phones": [], "emails": [],
        "eins": [], "vins": [], "zips": [], "websites": [], "dates": [],
    }) == ""


def test_format_ner_hints_includes_each_populated_category():
    tags = {
        "persons": ["Jane Doe"],
        "orgs": ["Acme LLC"],
        "phones": ["512-555-0100"],
        "emails": ["jane@acme.test"],
        "eins": ["12-3456789"],
        "vins": ["1FT7W2BT4NED38571"],
        "zips": [],
        "websites": ["acme.com"],
        "dates": [],
    }
    out = format_ner_hints(tags)
    assert "Jane Doe" in out
    assert "Acme LLC" in out
    assert "12-3456789" in out
    assert "1FT7W2BT4NED38571" in out
    assert "acme.com" in out
    assert "═══ NER ENTITY HINTS ═══" in out


def test_format_ner_hints_suppresses_multiple_persons():
    """3 detected persons = fleet driver list, not a business contact —
    suppress the PERSON hint to avoid nudging the LLM to misroute one."""
    tags = {
        "persons": ["Alice Jones", "Bob Smith", "Carol Davis"],
        "orgs": ["Acme Trucking LLC"],
        "phones": [], "emails": [], "eins": [], "vins": [],
        "zips": [], "websites": [], "dates": [],
    }
    out = format_ner_hints(tags)
    # ORGs still appear; PERSON hint omitted.
    assert "Acme Trucking LLC" in out
    assert "Alice Jones" not in out
    assert "PERSON" not in out


def test_format_ner_hints_filters_junk_orgs():
    """NER (en_core_web_sm) false-tags vehicle makes, state-code lists,
    ID patterns, and addresses as ORG entities. Those must NOT reach
    the prompt as 'Detected ORGANIZATION names'."""
    tags = {
        "persons": [], "phones": [], "emails": [], "eins": [], "vins": [],
        "zips": [], "websites": [], "dates": [],
        "orgs": [
            "Acme Trucking LLC",       # real
            "Ford F-250",               # vehicle make — junk
            "TX OK NM",                 # state list — junk
            "VIN# 1HGBH41",            # ID pattern — junk
            "Austin TX 78701",          # address — junk
        ],
    }
    out = format_ner_hints(tags)
    assert "Acme Trucking LLC" in out
    assert "Ford F-250" not in out
    assert "TX OK NM" not in out
    assert "VIN#" not in out
    assert "Austin TX 78701" not in out


def test_format_ner_hints_suppresses_org_section_when_all_filtered():
    """If every detected ORG is junk, the ORGANIZATION hint line is omitted."""
    tags = {
        "persons": [], "phones": [], "emails": [], "eins": [], "vins": [],
        "zips": [], "websites": [], "dates": [],
        "orgs": ["Ford", "TX OK", "VIN# 12345"],
    }
    out = format_ner_hints(tags)
    assert "ORGANIZATION" not in out


# ---------------------------------------------------------------------------
# _looks_like_business — junk-ORG filter
# ---------------------------------------------------------------------------

def test_looks_like_business_accepts_legal_suffix():
    assert _looks_like_business("Acme Trucking LLC") is True
    assert _looks_like_business("GlobeX Inc") is True
    assert _looks_like_business("Northern Star Logistics Corp") is True


def test_looks_like_business_rejects_vehicle_makes():
    assert _looks_like_business("Ford") is False
    assert _looks_like_business("Chev Silverado") is False
    assert _looks_like_business("Freightliner Cascadia") is False


def test_looks_like_business_rejects_acronyms():
    assert _looks_like_business("EIN") is False
    assert _looks_like_business("NAICS") is False
    assert _looks_like_business("DOT") is False


def test_looks_like_business_rejects_state_code_lists():
    assert _looks_like_business("TX OK NM") is False
    assert _looks_like_business("NY CA") is False


def test_looks_like_business_rejects_addresses():
    assert _looks_like_business("Austin TX 78701") is False
    assert _looks_like_business("Suite 200 Chicago IL 60601") is False
    assert _looks_like_business("12345") is False


def test_looks_like_business_rejects_id_patterns():
    assert _looks_like_business("CDL-A W123456") is False
    assert _looks_like_business("VIN# 1HGBH41") is False
    assert _looks_like_business("MC 98765") is False


def test_looks_like_business_rejects_jargon():
    assert _looks_like_business("Cov 1M CSL") is False
    assert _looks_like_business("License California") is False


def test_looks_like_business_requires_suffix_or_multiple_words():
    assert _looks_like_business("Acme") is False          # single word, no suffix
    assert _looks_like_business("Acme Trucking") is True   # two words


# ---------------------------------------------------------------------------
# validate_extraction_with_ner — four fixes (v4 paths)
# ---------------------------------------------------------------------------

def test_validator_fix1_removes_org_from_contact_name():
    """Contact named 'Acme Trucking LLC' is an ORG, not a PERSON → remove."""
    delta = {"contacts": [{"full_name": "Acme Trucking LLC"}]}
    tags = {"persons": [], "orgs": ["acme trucking llc"]}
    out = validate_extraction_with_ner(delta, tags)
    # full_name removed; contact dict may be emptied.
    assert not out["contacts"][0].get("full_name")


def test_validator_fix1_removes_contact_with_org_suffix():
    """Even without spaCy ORG tag, an ORG-suffixed name in the contact
    field gets removed by the regex suffix check."""
    delta = {"contacts": [{"full_name": "Widget Corp"}]}
    tags = {"persons": [], "orgs": []}
    out = validate_extraction_with_ner(delta, tags)
    assert not out["contacts"][0].get("full_name")


def test_validator_fix1_keeps_real_person_contact():
    delta = {"contacts": [{"full_name": "Jane Doe"}]}
    tags = {"persons": ["jane doe"], "orgs": []}
    out = validate_extraction_with_ner(delta, tags)
    assert out["contacts"][0]["full_name"] == "Jane Doe"


def test_validator_fix2_injects_person_into_missing_contact():
    """No contact extracted, NER found a PERSON → inject."""
    delta = {}
    tags = {"persons": ["Jane Doe"], "orgs": []}
    out = validate_extraction_with_ner(delta, tags)
    assert out["contacts"][0]["full_name"] == "Jane Doe"


def test_validator_fix2_skipped_on_multi_person():
    """Fleet inputs surface many drivers as PERSON entities. Fix 2 must
    NOT inject one of them as the contact — the business contact comes
    from a specific 'Contact is X' mention the LLM handles directly."""
    delta = {}
    tags = {"persons": ["Alice Jones", "Bob Smith", "Carol Davis"], "orgs": []}
    out = validate_extraction_with_ner(delta, tags)
    assert "contacts" not in out or not out.get("contacts")


def test_validator_fix2_skips_injection_if_person_is_business_name():
    """If NER's PERSON happens to be the business_name, don't inject
    as contact (would create a false identity)."""
    delta = {"business_name": "Acme Smith Construction"}
    tags = {"persons": ["Acme Smith Construction"], "orgs": []}
    out = validate_extraction_with_ner(delta, tags)
    # Contact not injected because PERSON equals business_name.
    assert "contacts" not in out or not out["contacts"] or not out["contacts"][0].get("full_name")


def test_validator_fix3_injects_org_into_missing_business_name():
    delta = {}
    tags = {"persons": [], "orgs": ["Acme Trucking LLC"]}
    current_state = {"business_name": ""}
    out = validate_extraction_with_ner(delta, tags, current_state)
    assert out["business_name"] == "Acme Trucking LLC"


def test_validator_fix3_skips_when_delta_has_business_name():
    delta = {"business_name": "Real Name Corp"}
    tags = {"persons": [], "orgs": ["Something Else LLC"]}
    out = validate_extraction_with_ner(delta, tags)
    assert out["business_name"] == "Real Name Corp"


def test_validator_fix3_skips_when_session_has_business_name():
    """Already-populated session — NER must NOT overwrite."""
    delta = {}
    tags = {"persons": [], "orgs": ["Different Company LLC"]}
    current = {"business_name": "Existing Biz Inc"}
    out = validate_extraction_with_ner(delta, tags, current)
    # delta unchanged (validator didn't inject).
    assert "business_name" not in out


def test_validator_fix3_filters_junk_orgs():
    """Vehicle make strings get filtered and are NOT injected as business."""
    delta = {}
    tags = {"persons": [], "orgs": ["Ford F250", "EIN"]}
    out = validate_extraction_with_ner(delta, tags, {})
    assert "business_name" not in out


def test_validator_fix4_injects_website():
    delta = {"business_name": "Acme Trucking"}
    tags = {"persons": [], "orgs": [], "websites": ["acmetrucking.com"]}
    out = validate_extraction_with_ner(delta, tags)
    assert out["website"] == "acmetrucking.com"


def test_validator_fix4_prefers_business_matching_website():
    delta = {"business_name": "Reliable Plumbing"}
    tags = {"persons": [], "orgs": [],
            "websites": ["unrelated.com", "reliableplumbing.com"]}
    out = validate_extraction_with_ner(delta, tags)
    assert out["website"] == "reliableplumbing.com"


def test_validator_fix4_skips_when_website_already_set():
    delta = {"website": "existing.com"}
    tags = {"persons": [], "orgs": [], "websites": ["other.com"]}
    out = validate_extraction_with_ner(delta, tags)
    assert out["website"] == "existing.com"


def test_validator_non_dict_input_passthrough():
    out = validate_extraction_with_ner(None, {})
    assert out is None
    out2 = validate_extraction_with_ner("string", {})
    assert out2 == "string"
