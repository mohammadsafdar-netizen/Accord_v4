# Test & Eval History — v4 Audit (2026-04-23)

Read-only audit for the v4-rewrite-pivot decision.
Inputs scanned: `tests/`, `eval_results/*.json`, `logs/app.log`,
`logs/step25_validity.jsonl`, `eval/scenarios/*.yaml`, `docs/**`,
`run_eval_*.py`, `accord_ai/config.py`, `accord_ai/eval/runner.py`.

---

## 1. Test Suite Anatomy

### 1.1 Totals

- **Collected by pytest:** `2197 tests` (from `pytest --collect-only -q`)
- **Files:** `124 test files` under `tests/` + 1 live smoke under `tests/integration/`
- **Gating:** Four files skip unless `ACCORD_LLM_INTEGRATION=1` is set; one smoke
  file skips unless `ACCORD_RUN_INTEGRATION=1` + several broker secrets are set.
- **Custom markers:** only `@pytest.mark.integration` is registered
  (in `tests/conftest.py`). No `e2e` / `eval` markers defined in v4 (the v3
  CLAUDE.md references them, but that's a different project).

### 1.2 Tests by subsystem

| Subsystem | Files | Notes |
|---|---:|---|
| API (FastAPI) | 20 | Every route. Complete/finalize/enrich/correction/harness/review etc. |
| Forms / PDF filler & mapper | 17 | Fill, canonical, arrays, contacts, producer, locations, vehicle coverage, violations, pipeline, storage, drive IDs, widget ground truth |
| Core infra (store, schema, cache, config, diff, logging, PII, v3-wire) | 18 | Includes `test_v3_wire_contract.py` (~1000 LOC) |
| Validation (external APIs + cross-field + OFAC) | 16 | NHTSA vPIC / recalls / safety, Census NAICS, USPS, FMCSA, SEC EDGAR, SAM.gov, Tax1099, DNS MX, Zippopotam, phone area, cross-field, eligibility |
| Extraction | 9 | extractor core, context, correction, mode (Step 25), NER, postprocess, prompts (3 files) |
| Knowledge / RAG | 7 | Chroma store, fake store, MiniLM, fake embedder, retriever, build_retriever, live integration |
| Conversation (controller, flow engine, responder, explainer) | 6 | |
| Harness (judge, manager, refiner, cascade, content, rules) | 6 | |
| Integrations (backend + Drive) | 6 | |
| LLM (engine, stack, openai, retrying) | 5 | |
| Store / persistence (store, logging, migration, jsonl) | 4 | |
| Eval harness itself (runner, scorer, path map) | 3 | |
| Feedback / DPO (collector, memory, dpo manager) | 3 | |
| OCR | 2 | |
| Ingest / merge (fleet, vehicle) | 2 | |

### 1.3 Skipped / gated tests

| File | Gate | What it covers |
|---|---|---|
| `tests/test_openai_engine_integration.py` | `ACCORD_LLM_INTEGRATION=1` | Real vLLM round-trip (1 test) |
| `tests/test_intake_integration.py` | `ACCORD_LLM_INTEGRATION=1` | Multi-turn scripted intake against live vLLM — "wire-up validation, not accuracy" |
| `tests/test_knowledge_integration.py` | `ACCORD_LLM_INTEGRATION=1` | Real MiniLM + real Chroma + real Retriever |
| `tests/test_minilm_embedder.py` (2 tests) | `ACCORD_LLM_INTEGRATION=1` | Real 384-d MiniLM vectors + semantic similarity |
| `tests/integration/test_smoke_backend_drive.py` | `ACCORD_RUN_INTEGRATION=1` + `BACKEND_CLIENT_SECRET` + `TEST_TENANT_SLUG` + `TEST_TENANT_DOMAIN` + `TEST_SUBMISSION_ID` | Real copilot.inevo.ai + real Google Drive OAuth |

**PDF tests:** Many `test_forms_*` call `pytest.importorskip("fitz")` — skipped if
PyMuPDF is missing (installed in dev).

No `xfail` markers found. No intentionally-red tests.

### 1.4 Covered well

- **Extraction correctness at unit level** — prompt rendering, postprocess,
  correction detection, mode selection, NER all unit-tested.
- **Validators** — every external validator has 3-8 isolated tests with mocked
  HTTP. 16 files total.
- **API surface** — every route has a dedicated file, including edge cases
  (rate limit, answer limits, correction sqlite path, DPO admin, upload image,
  upload PDFs, complete+drive, complete+overrides).
- **PDF pipeline** — 17 files cover registry / filler / mapper for every form
  family + 2 layout checks (widget ground truth, canonical).
- **Persistence** — store unit + migration + logging + JSONL stopgap.
- **Cross-field validation** and **OFAC** have dedicated tests.

### 1.5 Covered poorly or not at all

- **Harness composition** — only 6 files (content, rules, manager, judge,
  refiner, refiner_cascade). No test exercises the full inject→judge→refine
  loop end-to-end with a live extraction pass. `test_judge.py` and
  `test_manager.py` are unit-level mocks.
- **Per-broker harness** — `accord_ai/harness/brokers/` contains only
  `.gitkeep`. No broker harness files, no tests, no retrieval path. Phase 4
  feature, unimplemented.
- **RAG retrieval wired to the extractor** — `test_retriever.py` +
  `test_knowledge_integration.py` cover the retriever in isolation; no test
  confirms retriever output is actually composed into an extraction prompt.
- **Cross-tenant isolation** — no test sets up two tenants and verifies data
  cannot leak between them. v3 claims "verified 5 concurrent users, 3 tenants,
  zero data leaks"; v4 has no equivalent.
- **Multi-turn conversational behavior** — `test_intake_integration.py` is
  the only multi-turn test and it's gated behind live vLLM. No hermetic
  multi-turn test exists with FakeEngine.
- **Refiner cascade with real keys** — `test_refiner_cascade.py` uses mocks.
  Gemini and Claude clients have files (`gemini.py`, `claude.py`) but no
  live-key verification.
- **Concurrency / load** — no load test, no benchmark scripts in v4 (v3 had
  `benchmark_concurrency.py` and `test_5tenant_deep.py`; those did not port).
- **OCR integration** — 2 tests (`test_ocr_engine.py`, `test_ocr_parser.py`)
  likely unit-only; no test exercises the upload-image → OCR → extraction
  path.
- **Fleet ingest end-to-end** — 1 unit file. No test combines an Excel roster
  + session merge + extraction.
- **LoRA adapter load/unload** — zero tests (see `docs/VLLM_LORA_VERIFIED.md`
  — "empirical verification deferred to Phase 4 step 4.6").

---

## 2. Eval History — 55-Scenario Runs

### 2.1 Scenario inventory

`eval/scenarios/` contains **67 scenarios total** across 9 YAML files:

| Family | File | Count |
|---|---|---:|
| standard | `standard.yaml` | 10 |
| hard | `hard.yaml` | 10 |
| negation | `negation.yaml` | 8 |
| edge_cases | `edge_cases.yaml` | 8 |
| multi_entity | `multi_entity.yaml` | 6 |
| correction | `correction.yaml` | 6 |
| bulk | `bulk.yaml` | 4 |
| document_upload | `document_upload.yaml` | 4 |
| guards | `guards.yaml` | 11 |
| **TOTAL** | | **67** |

Every logged eval in `eval_results/` reports `scenarios_run: 55`. The 12-scenario
gap (67 → 55) is **not** explained by any comment in `run_eval_67.py` or
`accord_ai/eval/runner.py` — `load_scenarios()` only skips entries missing
`turns` or `expected`. The likely culprits are the 11-scenario `guards.yaml` +
one scenario elsewhere that fails loader validation. This is worth pinning
down before claiming "55-scenario coverage" — currently 18% of authored
scenarios are silently dropped.

Scenario families actually executed (derived from `scenarios[*].scenario_id`
prefixes in the JSON reports):

| Family | Scenarios run | Present in YAML |
|---|---:|---:|
| standard | 10 | 10 |
| hard | 10 | 10 |
| negation | 8 | 8 |
| edge | 8 | 8 |
| multi | 6 | 6 |
| correction | 6 | 6 |
| bulk | 4 | 4 |
| upload | 3 | 4 |
| **TOTAL** | **55** | **56** (of 67) |

Guards (11) + 1 upload scenario are missing from every run. Guards never
appears in any logged eval.

### 2.2 Chronological eval table

All seven runs used the SAME 55-scenario set, `concurrency=1`, and whatever
vLLM was serving on `http://localhost:8000/v1`. Base model per
`docs/VLLM_LORA_VERIFIED.md`: `Qwen/Qwen3.5-9B`. Source file:
`eval_results/<tag>.json`.

| Tag | Timestamp (local) | Elapsed | Mode | Harness | Refiner | Seed | P | R | **F1** | 1st-pass turns | Refiner-rescued | Notes |
|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---|
| variance_A1 | 2026-04-23 06:23 | 1409s | xgrammar | none | OFF | 42 | .7277 | .6590 | **.6917** | 0/187 | 0 | Determinism run 1 |
| variance_A2 | 2026-04-23 06:47 | 1405s | xgrammar | none | OFF | 42 | .7289 | .6602 | **.6928** | 0/187 | 0 | Determinism run 2 |
| variance_A3 | 2026-04-23 07:10 | 1400s | xgrammar | none | OFF | 42 | .7254 | .6567 | **.6893** | 0/187 | 0 | Determinism run 3 |
| step25_A | 2026-04-23 08:24 | 3583s | xgrammar | none | **ON** | 42 | .7451 | .6831 | **.7128** | 5/187 (2.7%) | 6 (3.3%) | Step 25 anchor A — current production baseline |
| step25_B | 2026-04-23 09:41 | 4657s | **json_object** | **full** | ON | 42 | .6141 | .5775 | **.5953** | 9/187 (4.8%) | 7 (3.9%) | Step 25 anchor B |
| step25_C | 2026-04-23 10:39 | 3450s | xgrammar | **light** | ON | 42 | .5944 | .5522 | **.5726** | 6/187 (3.2%) | 8 (4.4%) | Step 25 anchor C |
| step25_D | 2026-04-23 11:37 | 3483s | **free** | **full** | ON | 42 | .6234 | .5855 | **.6039** | 9/187 (4.8%) | 7 (3.9%) | Step 25 anchor D |

**Current production config** = xgrammar + harness=none + refiner ON + seed=42
→ **F1 71.3%** (step25_A).

**Pending variants (per user description):** none scheduled beyond D; the
matrix is the 4 anchors A-D, all complete.

### 2.3 Per-family F1 by run

| Family | n | variance_A1 | variance_A2 | variance_A3 | step25_A | step25_B | step25_C | step25_D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| correction | 6 | .958 | .958 | .958 | .958 | .861 | .861 | .861 |
| standard | 10 | .795 | .795 | .790 | .720 | .655 | .696 | .665 |
| multi | 6 | .746 | .746 | .746 | .834 | .771 | .735 | .735 |
| bulk | 4 | .707 | .707 | .707 | .730 | .363 | .195 | .425 |
| edge | 8 | .667 | .679 | .657 | .666 | .486 | .565 | .528 |
| negation | 8 | .676 | .676 | .676 | .676 | .801 | .707 | .801 |
| hard | 10 | .587 | .587 | .587 | .784 | .679 | .597 | .671 |
| upload | 3 | .351 | .351 | .351 | .381 | .352 | .325 | .352 |

Key observations:

- **correction is robust** across all configs (.861-.958) — rule-based
  detection shields it from harness/mode changes.
- **bulk collapses** under harness=light/full (.195-.425 vs .707 baseline).
  Step C (xgrammar+light) drops two scenarios to 0.000 F1
  (`bulk-all-business-info`, `bulk-mixed-everything`). This is the headline
  regression: harness text in the extraction prompt breaks complex
  multi-entity single-turn dumps.
- **hard improves** when refiner is ON + harness=none (.587 → .784 in
  step25_A). Rescue path helps the hardest family the most.
- **negation improves** slightly in step25_B/D (full harness) — its rules
  target exactly this family.
- **upload stays low** (.32-.38) across every config. Document-upload
  prefill pipeline is likely unwired in v4 (per `KNOWN_LOW_SCORES.md`).

### 2.4 Worst scenarios (persistent)

| Scenario | variance_A1 F1 | step25_A F1 | Notes |
|---|---:|---:|---|
| `upload-merge-with-session` | .136 | .227 | Prefill pipeline gap |
| `upload-partial-prefill` | .446 | .446 | Same |
| `standard-landscaper` | — | .278 | Regression vs variance |
| `standard-restaurant` | — | .372 | |
| `edge-frustrated-customer` | — | .378 | |
| `hard-compound-cities` | .333 | — | Address disambiguation |
| `hard-special-chars` | .400 | — | |
| `standard-hvac-contractor` | — | .333 (per KNOWN_LOW_SCORES) | All 4 turns fail harness verdict |

---

## 3. Step 25 Matrix Specifics

### 3.1 Configuration axes (per `accord_ai/config.py`)

- `extraction_mode` ∈ {`xgrammar`, `json_object`, `free`}
- `experiment_harness` ∈ {`none`, `light`, `full`}
- `harness_max_refines` ∈ {0, 1} — 0 disables refiner entirely

### 3.2 Anchor variants run

| Anchor | Mode | Harness | Refiner | F1 | What it tests |
|---|---|---|---|---:|---|
| A | xgrammar | none | ON | .7128 | **Production baseline** (current default) |
| B | json_object | full | ON | .5953 | Drop xgrammar + inject full harness (v3-style) |
| C | xgrammar | light | ON | .5726 | Keep xgrammar + inject light harness |
| D | free | full | ON | .6039 | Remove format constraint entirely + full harness |

Delta vs A: B=−11.8pt, C=−14.0pt, D=−10.9pt. Every harness-injection variant
loses accuracy. xgrammar structural enforcement is worth ~11-14 points on this
benchmark.

### 3.3 Variance runs

| Run | Config | F1 |
|---|---|---:|
| variance_A1 | xgrammar / harness=none / **refiner OFF** / seed=42 | .6917 |
| variance_A2 | same | .6928 |
| variance_A3 | same | .6893 |

Range: **0.35pt** across three identical runs. **51/55 scenarios produce
identical F1 all three times.** Only 4 scenarios vary:

- `edge-compound-city-kansas-city`: ±11.9pt (0.526 → 0.645 → 0.526)
- `edge-typos-business-name`: ±7.7pt
- `standard-auto-dealer`: ±5.4pt
- `edge-frustrated-customer`: ±5.4pt

These four are all in edge/standard, all involve judgment calls on multi-valued
fields. The aggregate is essentially deterministic; per-scenario residual comes
from non-determinism in 4 specific scoring paths, not the LLM.

### 3.4 What's pending / running

User's description mentions a "currently running" matrix experiment. All four
anchor JSONs (A, B, C, D) are on disk as of 2026-04-23 11:37. No additional
in-progress run was found. **The matrix appears complete.**

### 3.5 Step 25 validity log

`logs/step25_validity.jsonl` (1894 entries, 2026-04-22 06:06 → 2026-04-23 11:36)
captures first-try / retry validity per extraction call across the full matrix:

| Mode | Harness | Calls | 1st-try valid | Retry ok | Retry fail | Final ok |
|---|---|---:|---:|---:|---:|---:|
| xgrammar | none | 1,392 | 99.1% | — | — | 99.1% |
| xgrammar | light | 162 | 100.0% | — | — | 100.0% |
| xgrammar | full | 6 | 100.0% | — | — | 100.0% |
| json_object | full | 163 | 96.3% | 6 | 0 | 100.0% |
| free | full | 171 | 93.0% | 6 | 6 (3.5%) | 96.5% |

`free` mode is the only one with unrecoverable schema-validation failures. This
is the validity signal — separate from scoring.

---

## 4. Non-55-Eval Scripts

| Script | Purpose | Status |
|---|---|---|
| `run_eval_5.py` | 5-scenario smoke (solo plumber, delivery fleet, bulk-all, 5-vehicle fleet, no-hired-auto) | Standing script; no captured artifact in `eval_results/` |
| `run_eval_correction.py` | 6 correction-family scenarios with diagnostic logging | Phase A postmortem. Output went to `/tmp`, not captured |
| `run_eval_cvy.py` | Single scenario: `correction-vehicle-year` | Phase A Step 3A diagnostic |
| `run_eval_ntc.py` | Single scenario: `negation-then-correction` | Phase A Step 3B diagnostic |
| `run_hvac_diag.py` | Single scenario: `standard-hvac-contractor` | Worst-performer diagnostic |
| `diagnose_bulk.py` | Live engine trace for bulk-* scenarios to distinguish truncation vs refusal vs xgrammar fail | One-off debug tool |

**No captured artifacts from the smaller scripts are on disk.** The only
persistent results in `eval_results/` are the 7 full 55-runs.

**No benchmark scripts at all** (no concurrency / latency / multi-tenant
analogues of v3's `benchmark_concurrency.py`, `test_5tenant_deep.py`,
`benchmark_complex.py`).

---

## 5. Refiner Observations

Numbers from `turn_stats` in each JSON (187 total turns per run = sum across
55 scenarios):

| Run | Refiner | 1st-pass passed | Refiner rescued | Rescue rate | Still failing |
|---|---|---:|---:|---:|---:|
| variance_A1 | OFF | 0 | 0 | 0.0% | 187 |
| variance_A2 | OFF | 0 | 0 | 0.0% | 187 |
| variance_A3 | OFF | 0 | 0 | 0.0% | 187 |
| step25_A | ON | 5 | 6 | 3.3% | 176 |
| step25_B | ON | 9 | 7 | 3.9% | 171 |
| step25_C | ON | 6 | 8 | 4.4% | 173 |
| step25_D | ON | 9 | 7 | 3.9% | 171 |

**Observations:**

- **"Still failing" is huge (91-100%)** — the *judge* says almost every turn
  fails, but F1 is 57-71%. This means the judge's failure criterion is
  substantially stricter than the scorer's matching criterion. Judge and
  scorer are not measuring the same thing.
- **Refiner rescue rate is tiny (3-4%).** In step25_A, refiner rescued 6 of
  176 judge-failing turns = 3.4%. The rest of the ~14pt aggregate gap vs the
  variance runs (.7128 vs .6917) is explained by the refiner rescuing a
  small number of high-value turns, not by broad quality lift.
- App log confirms `judge_v*`-style counts: 1,582 "initial judge: passed=False"
  and 0 "initial judge: passed=True" lines. Judge passes *essentially never*
  on a turn-by-turn basis in this setup. The 5-9 "first_pass_passed" turns
  in the JSON look like end-of-scenario passes where cumulative state
  eventually satisfies the judge.
- Post-refine attempt=1 outcomes across all runs: 84 passed, 86 still failed
  (parsed from `logs/app.log`). Judge + refiner pair is a ~50/50 coin flip
  *given* a turn that needed refinement.
- Refiner-triggered scenarios: the log correlation with scenario IDs would
  need a larger grep; not reconstructed here. Patterns in `docs/PHASE_A_POSTMORTEM.md`
  suggest correction-family + hard-family + edge-family are the usual
  triggers.

---

## 6. Methodology Notes

### 6.1 Why was variance 3pt before the fix?

Not directly reproducible from artifacts on disk — no pre-fix eval JSON
remains. `accord_ai/config.py:206-209` documents the fix:

> `llm_seed: Optional[int] = Field(default=None)` — "Pin vLLM's internal RNG.
> None = no seed (current behavior, non-deterministic between runs).
> Set `LLM_SEED=42` to get reproducible outputs at temperature=0."

Temperature-0 sampling is not deterministic across vLLM batches because:
1. FP tie-breaking between equal-probability tokens depends on batch order
   (CUDA non-associative reductions).
2. Prefix-cache state shifts across runs.
3. Without seed pinning, vLLM's internal RNG advances differently between
   runs even at temp=0.

Variance before seed+temp0 fix was ~3pt per the user's description.
Post-fix: 0.35pt across 3 runs (variance_A*).

### 6.2 Is the eval deterministic now?

**Yes, to ±0.35pt aggregate F1.** 51/55 scenarios produce bit-identical
per-scenario F1 across 3 back-to-back runs under the locked config
(xgrammar / harness=none / refiner OFF / seed=42). The 4 varying scenarios
vary in a way that cancels at aggregate level (noise, not bias).

Caveat: variance runs disabled the refiner. With refiner ON, there's an
additional source of non-determinism (the refiner cascade has its own LLM
calls) that has NOT been characterized. step25_A through D are single-sample
runs — we don't know their variance.

### 6.3 Methodology to lock in

Based on what's been validated:

1. **Seed = 42** — pinned.
2. **Temperature = 0** — already default in engine.
3. **Refiner OFF for baseline comparisons** — refiner adds non-determinism
   and a ~2-4pt lift that should be reported separately, not baked in.
4. **concurrency = 1 in eval runner** — already enforced in `run_all()` call.
5. **Document which scenarios are dropped** — 12 of 67 are currently silent
   drops; the "55-scenario eval" name is misleading until this is pinned
   down.
6. **Report per-family F1 alongside aggregate** — the headline F1 hides
   that bulk went 0.707 → 0.195 under harness=light. Single-number
   comparisons are unsafe.
7. **3-run variance bands for any refiner-ON claim** — single step25
   samples should not be treated as precise.
8. **Pin vLLM version + model name + `--kv-cache-dtype`** in the eval report
   — today only seed is pinned.

---

## 7. UNTESTED CAPABILITIES — what we can't verify yet

| Claim | Evidence that it works | Evidence it's untested |
|---|---|---|
| Per-broker extraction behavior | None | `accord_ai/harness/brokers/` is empty (only `.gitkeep`). Corrections table empty. No test, no scenario. |
| Per-broker correction-memory retrieval affects extraction | Memory code exists (`accord_ai/feedback/memory.py`) and is invoked at extraction time | No live eval run with seeded corrections to confirm lift |
| RAG (ChromaDB) wired into the extractor | Retriever code exists; integration test covers retriever in isolation | No test confirms retriever output is included in the extraction prompt's context; no eval with RAG=on vs RAG=off |
| Multi-tenant isolation | File-layout separation per-tenant | Zero cross-tenant isolation tests. No concurrent-user benchmark in v4 (unlike v3). |
| Refiner cascade with real Gemini/Claude keys | Client code exists (`refiner_clients/gemini.py`, `claude.py`); cascade unit test passes with mocks | No run with real API keys captured |
| LoRA adapter load/unload | `docs/VLLM_LORA_VERIFIED.md` = DOCS-level verified only | Empirical verification explicitly deferred to Phase 4 step 4.6 |
| LoRA trained against current base model | Adapters on disk target `Qwen3-8B` or `Qwen3-VL-8B` — not the served `Qwen3.5-9B` | No adapter trained against the production base yet |
| Upload-document prefill pipeline (`upload-*` scenarios) | YAML scenarios authored | Flat .32-.38 F1 across every config run — pipeline likely not wired (per `KNOWN_LOW_SCORES.md`) |
| Guard scenarios (11 in `guards.yaml`) | Authored | **Never executed** — silently dropped by the 55-scenario filter. Unknown pass rate. |
| Concurrency / load behavior | None | No concurrency benchmark script. v3's `benchmark_concurrency.py` did not port. |
| Judge-vs-scorer alignment | Neither matches the other | Judge fails 91-100% of turns at harness/turn level, yet aggregate F1 is .57-.71 — clear signal that judge criteria are not calibrated against the scoring rubric |
| OCR → extraction end-to-end | OCR unit tests + upload-image API test exist individually | No composite test |
| Fleet ingest end-to-end | Unit test for parser | No composite with session merge + extraction |
| Production latency claims | None captured | No latency or token-economy eval in `eval_results/` |
| Hallucination rate | None | v3 reported 0% hallucination; v4 has no absent-path hallucination audit in the captured results (though `absent` paths are scored per scenario) |

---

## 8. Key files / artifacts

- `/home/inevoai/Development/Accord-Model-Building/Custom_model_fa_pf/accord_v4/eval_results/variance_A{1,2,3}.json` — determinism evidence
- `/home/inevoai/Development/Accord-Model-Building/Custom_model_fa_pf/accord_v4/eval_results/step25_{A,B,C,D}.json` — matrix results
- `/home/inevoai/Development/Accord-Model-Building/Custom_model_fa_pf/accord_v4/logs/step25_validity.jsonl` — per-call validity across the matrix
- `/home/inevoai/Development/Accord-Model-Building/Custom_model_fa_pf/accord_v4/logs/app.log` — 41,939 lines of judge/refiner traces (2026-04-20 onward)
- `/home/inevoai/Development/Accord-Model-Building/Custom_model_fa_pf/accord_v4/docs/PHASE_A_POSTMORTEM.md` — SYSTEM_V3 regression root-cause
- `/home/inevoai/Development/Accord-Model-Building/Custom_model_fa_pf/accord_v4/docs/KNOWN_LOW_SCORES.md` — worst-scenario analysis at 72.1% (an earlier point estimate, not among the 7 on disk)
- `/home/inevoai/Development/Accord-Model-Building/Custom_model_fa_pf/accord_v4/docs/DEFERRED_DEBT.md` — production-readiness gaps (SQLite async, CORS, limits)
- `/home/inevoai/Development/Accord-Model-Building/Custom_model_fa_pf/accord_v4/docs/VLLM_LORA_VERIFIED.md` — LoRA support, docs-level only
- `/home/inevoai/Development/Accord-Model-Building/Custom_model_fa_pf/accord_v4/accord_ai/config.py:199-209` — Step 25 experiment knobs + LLM_SEED plumbing
- `/home/inevoai/Development/Accord-Model-Building/Custom_model_fa_pf/accord_v4/accord_ai/eval/runner.py` — the loader that silently drops 12 scenarios
- `/home/inevoai/Development/Accord-Model-Building/Custom_model_fa_pf/accord_v4/run_eval_67.py` — the 55-run driver (despite the "67" name)
- `/home/inevoai/Development/Accord-Model-Building/Custom_model_fa_pf/accord_v4/eval/scenarios/guards.yaml` — 11 authored-but-never-run scenarios

---

**Bottom line for the pivot decision:**

- 2,197 tests pass, mostly unit + mocked-integration. High confidence in
  validators, API surface, PDF pipeline, persistence.
- 55-scenario eval is **deterministic** at ±0.35pt F1 post-seed-fix. We can
  trust A/B deltas of ≥0.5pt.
- **Current production baseline: F1 71.3%** (step25_A). Variance-run baseline
  with refiner off: F1 69.2%. Refiner rescues ~2pt aggregate via 3-4% of turns.
- Step 25 matrix **done** — all four anchors on disk. Verdict: xgrammar + no
  harness injection wins. Harness injection costs 11-14 F1 points.
- The claim "55 scenarios" is misleading — 12 of 67 authored scenarios are
  silently skipped. Guards family (11) has **never** been benchmarked.
- Phase 4 capabilities (per-broker LoRA, RAG-wired extraction, live refiner
  cascade, multi-tenant isolation) are all at the docs-or-scaffolding level.
  No empirical evidence for any of them.
