# Phase A Postmortem — Correction-family regression root causes

**Date:** 2026-04-20
**Eval sample:** all 6 correction-* scenarios, Phase A Steps 1-4 active
**Result:** aggregate F1 0.7089 vs prior-baseline ~1.0 on the subset where baseline measured

## Summary

Instrumented `Extractor.extract()` with `extraction_route` + `extraction_output` log lines (per-turn prompt choice, field_hint, extracted top-level keys, correlated by MD5(turn_text)[:8]). Cross-referenced against the 6 correction scenarios' turn shapes and expected outputs.

**The correction branch itself is not the regression source.** When `is_correction()` fires and the field hint is correct, the focused prompt consistently emits exactly the targeted field (`ein`, `entity_type`, `business_name`, `policy_dates`). The regression concentrates in TWO distinct failure modes, neither of which is failure mode (c) "bad SYSTEM_CORRECTION_V1 prompt text" or (d) "schema narrowing" (v4 never narrows — `schema_paths_kept=full` on every trace).

## Failure mode (a) — DEFAULT extraction fails on vehicle/driver middle turns

**Frequency:** 3 of 4 regressed scenarios. Dominant driver of the F1 drop.

**Scenarios affected:**
- `correction-ein` (F1=0.500): turn 1 "2024 Ford F-350 VIN 1FT8W3DT0RED45678 for service, garaged Traverse City MI 49684" — **extraction_output missing, `schema validation failed`** in log
- `correction-driver-dob` (F1=0.308): turn 1 "Driver Neil Andersen DOB 03/21/1984 WA license..." — extraction_output missing
- `correction-entity-type` (F1=0.625): turn 1 "2023 Ford F-450 VIN... for hauling" — extraction_output missing

These turns were routed through the DEFAULT `SYSTEM_V3` prompt (which is `SYSTEM_V2` + `HARNESS_RULES`, ~2000 tokens). The LLM's output failed pydantic validation → `ExtractionOutputError` → graceful-degrade → session state unchanged → later scoring misses the vehicle/driver paths.

**Root cause:** SYSTEM_V3's expanded harness content (field routing, negation rules, correction recognition, entity-type enum, address parsing, numeric disambig, temporal dates, prior insurance, loss history, cross-field contamination) competes for attention against the `lob_details.vehicles[N]` / `lob_details.drivers[N]` nested-object emission the LLM needs to produce on these turns. Base Qwen3.5-9B with a schema-constrained (guided_json) decoder is already at its structural-emission budget; the extra system-prompt content pushes the output off the valid-JSON manifold on complex-object turns.

**Same-model baseline (SYSTEM_V2, no harness):** these vehicle/driver turns extracted cleanly in the 68.4% F1 baseline run.

## Failure mode (b) — correction field-hint gaps

**Frequency:** 1 of 4 regressed scenarios (contributing factor).

**Scenario:** `correction-driver-dob` (F1=0.308): turn 2 "Wait, my birthday is March 12 not March 21. It's 03/12/1984."

**Trace:**
```
🔀 CORRECTION  prompt=SYSTEM_CORRECTION_V1  field_hint=None
extracted: []
```

The user said "birthday" — not in `_CORRECTION_FIELD_HINTS`. My map has "dob" and "date of birth" but not "birthday". Without a hint, the correction prompt's "output {} if you cannot determine what is being corrected" rule activated → empty extraction. Driver DOB never got corrected.

**Root cause:** incomplete keyword → v4-path map. Several natural-language synonyms missing. Easy fix: add aliases.

**Related regex gap — `correction-effective-date`:** turn 1 "The effective date needs to change to June 1 2026." Did NOT match `is_correction()` regex because `change\s+it\s+to` requires the word "it" ("change it to"), and the user said "change to" without "it". The turn was routed through DEFAULT and happened to land correctly — but if the extractor had been harder on the turn, this scenario would also have regressed via the same path as failure mode (a). The DEFAULT path's success here is incidental, not robust.

## Non-failure modes confirmed out of scope

- **(c) bad SYSTEM_CORRECTION_V1 text:** on the 4 correction turns where field_hint was detected correctly (`ein`, `entity_type`, `business_name`, etc.), the prompt produced the expected targeted extraction. The prompt shape works.
- **(d) schema narrowing:** every trace emits `schema_paths_kept=full`. v4 never narrows the guided_json schema. Not a failure mode in this implementation.

## Surgical remediation plan

**Primary — revert extractor DEFAULT path from SYSTEM_V3 to SYSTEM_V2.** The harness content belongs in the refiner (where it has smaller output and is consumed by the LLM already committed to a fix), not the extractor where it competes with schema-constrained complex-object emission. Expected lift: the vehicle/driver turns that are failing recover to the 68.4% baseline shape. Keep SYSTEM_V3 as a shelved constant with the revert-reason documented.

**Secondary — expand `_CORRECTION_FIELD_HINTS`** with natural-language aliases: `birthday`, `birth date`, `birthdate`, `starting date`, `start date`, `fleet`. Also loosen `_CORRECTION_RE` to match `change\s+(?:it\s+)?to` so "needs to change to" routes through the correction branch. These add hit-rate on correction detection without changing the focused prompt's proven behavior.

**Tertiary — keep Step 1 (adaptive max_tokens) and Step 2 (postprocess pipeline) as-is.** Neither contributed to the regression. Both are defenses that trigger on edge cases the correction family didn't exercise, so they're neutral here and (per prior bulk-scenario diagnostics) net-positive on others.

## Expected after remediation

- correction-business-name: 1.0 (unchanged — DEFAULT was never the issue on this scenario)
- correction-vehicle-year: 1.0 (DEFAULT extraction succeeded, correction turn had no output to start from so the state from turn 0 was sufficient)
- correction-ein: 0.500 → ~0.833 (turn 1 vehicle extraction recovers)
- correction-driver-dob: 0.308 → ~0.667 (turn 1 driver recovers; +alias fix recovers dob correction)
- correction-effective-date: 0.750 → ~1.0 (turn 1 contact_name in turn 0 recovers)
- correction-entity-type: 0.625 → ~0.875 (turn 1 vehicle recovers)

Mean correction-family F1: 0.709 → ~0.87 (+16 points on the subfamily). Aggregate F1 lift depends on similar vehicle/driver middle-turn recoveries in standard/bulk/multi families — likely larger since those families have more such turns.

---

**tl;dr for status reply:** SYSTEM_V3's expanded harness is killing vehicle/driver extraction on middle turns of multi-turn scenarios (failure mode a, 3 of 4 regressed). Correction branch itself works correctly when triggered. Secondary: hint map missing "birthday" / regex missing "change to" (b). Schema-narrowing (d) doesn't apply — v4 never narrows. Remedy: revert extractor to SYSTEM_V2 (keep harness off the extractor; refiner path can keep it); expand hint map + regex. Keep Steps 1+2 as-is.
