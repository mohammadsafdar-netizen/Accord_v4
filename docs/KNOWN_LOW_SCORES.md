# Known Low Scores — Forward-Looking Candidates

Scenarios below 0.5 F1 as of 2026-04-20 55-eval (F1=72.1% aggregate). Documented
for Phase 3 (controller) and Phase 4 (training). No action in Phase 1.

---

## standard-hvac-contractor — F1=0.333

**Eval stats:** matched=6/17, translated=19, still_failing=4/4 turns

**Key observation:** `still_failing=4` — ALL turns fail the harness verdict, not just the
last one. This is a cascade: the judge never marks the session "in progress" because
required early-turn fields are missing. Likely stops the controller from asking follow-up
questions in the right order.

**Suspected root causes (to verify with per-field comparison when LLM is live):**

1. **Abbreviation normalization — "Chevy" → "Chevrolet"**
   Turn 3 says "Chevy Express 2500". Expected `vehicles[0].make: "Chevrolet"`. The LLM
   likely emits "Chevy" rather than expanding the abbreviation. Similar pattern seen in
   other scenarios with make abbreviations. Fix target: postprocess `coerce_list_fields`
   or a dedicated vehicle-make normalizer.

2. **Entity-type normalization — "S-corp" → "subchapter_s"**
   Turn 1 says "We're an S-corp." Expected `entity_type: "subchapter_s"`. The LLM likely
   emits "s_corp" or "s-corp" (hyphen variant). The schema enum is `subchapter_s`. Fix
   target: postprocess step or schema-level alias mapping.

3. **Multi-driver ordering across turns**
   Turn 4 introduces both drivers in one message. Driver ordering (index 0/1) may be
   inconsistent if the LLM doesn't preserve insertion order or if postprocess re-sorts.

4. **Scenario authoring note:** 17 expected fields, 4 turns. This is a high field density
   per turn (avg ~4.25 fields/turn). The judge rejecting all 4 turns suggests the verdict
   required fields don't align with what this LOB actually needs. Worth reviewing the
   harness required-field list for commercial_auto to see if it's over-specified.

**Phase target:** Phase 3 (controller) for verdict alignment; Phase 4 (training) for
abbreviation normalization.

---

## upload-merge-with-session — F1=0.227
## upload-partial-prefill — F1=0.361

**Eval stats:** both in the upload-* family, which tests document ingestion pre-filling
the submission before the conversation starts.

**Suspected root cause:** The upload-to-session prefill pipeline may not be fully wired
in v4 (no equivalent of v3's document parsing). These scenarios likely fail because the
pre-filled fields are never loaded into the initial submission state.

**Phase target:** Phase 2 (document upload) or Phase 5 (advanced pipeline).

---

## negation-combined — F1=0.500
## negation-question-format — F1=0.500

**Eval stats:** both in the negation family, which otherwise performs well (4/9 at ≥0.80).

**Suspected root cause for negation-question-format:** User asks "Do you cover hired auto?"
rather than stating "no hired auto." The negation rule fires on "no hired auto" keywords;
question-format triggers don't match. Fix target: add question-form pattern to the
negation rule.

**Suspected root cause for negation-combined:** Multiple negations in one turn. The rule
may apply only the first match. Fix target: iterate all negation matches.

**Phase target:** Phase 1.3 or Phase 3 (controller).
