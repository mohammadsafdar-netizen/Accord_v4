# V4 Current State — Honest Audit (2026-04-23)

Audit for v4 rewrite-pivot decision. Read-only snapshot of shipped code, degraded
pieces, reverts, flags, debt, and test coverage. File paths are absolute.

Prior audits examined: `docs/DEFERRED_DEBT.md`, `docs/PHASE_A_POSTMORTEM.md`,
`docs/KNOWN_LOW_SCORES.md`. This document does not repeat those — it
cross-references them where relevant.

---

## 1. Shipped and working well

### 1a. Multi-tenant infrastructure
- SQLite schema threads `tenant` column through every row
  (`accord_ai/core/store.py:30,46-51` — migrations 1-5 in one `_MIGRATIONS`
  tuple; 5 tables: sessions, messages, audit_events, corrections/feedback/
  training_pairs, flow_state_json column).
- WAL mode, per-thread connections via `threading.local()`
  (`accord_ai/core/store.py:217-239`).
- Optimistic concurrency via `expected_updated_at` +
  `ConcurrencyError` (`accord_ai/core/store.py:137-143`).
- ChromaDB PersistentClient, per-tenant collection naming
  (`accord_ai/knowledge/chroma_vector_store.py:4,26`).
- Request-scoped `ContextVars` for tenant/session/request_id
  (`accord_ai/request_context.py`, wired in
  `accord_ai/api.py:1166,1177`).

### 1b. Wire-contract endpoints (25 routes)

| Method | Path | Impl | Notes |
|---|---|---|---|
| GET | /health | Real | `api.py:1263` — checks vLLM + knowledge |
| POST | /start-session | Real | `api.py:1272` — creates session + first flow-engine question |
| POST | /answer | Real | `api.py:1305` — full `process_turn` |
| POST | /finalize | Real | `api.py:1320` |
| POST | /upload-document | Real | `api.py:1347` — fleet Excel/CSV |
| POST | /complete | Real | `api.py:1435` — PDFs + Drive + validation |
| GET | /pdf/{session_id}/{form_number} | Real | `api.py:1712` |
| GET | /session/{session_id} | Real | `api.py:1751` |
| GET | /sessions | Real | `api.py:1801` (limit=0/-1 bug per D4) |
| GET | /explain/{field_path} | Real | `api.py:1827` — RAG-backed |
| POST | /enrich | Real | `api.py:1869` |
| GET | /review/{session_id} | Real | `api.py:1915` |
| POST | /review/{session_id}/resolve | **Stub** | `api.py:1967` — 2 lines, no body |
| POST | /correction | Real | `api.py:1973` |
| POST | /feedback | Real | `api.py:1994` |
| POST | /admin/dpo/export/{tenant} | Real | `api.py:2015` — DPOManager |
| GET | /admin/dpo/status/{tenant} | Real | `api.py:2054` (touches `_threshold` private attr — D7) |
| POST | /upload-image | Real | `api.py:2075` — OCR |
| POST | /upload-filled-pdfs | Real | `api.py:2171` |
| POST | /upload-blank-pdfs | Real | `api.py:2217` |
| GET | /debug/session/{session_id} | Real | `api.py:2246` |
| GET | /fields/{session_id} | Real | `api.py:2297` |
| GET | /harness | Real | `api.py:2316` |
| GET | /harness/audit | Real | `api.py:2333` |
| GET | /harness/history | Real | `api.py:2353` |
| POST | /harness/rollback | Real | `api.py:2368` |
| GET | /harness/provenance | Real | `api.py:2376` |
| GET | /harness/review-queue | Real | `api.py:2384` |

Stubs: `/review/{session_id}/resolve` is shipped as an empty route.

### 1c. Validators

12 validators live under `accord_ai/validation/`. Control flag
`ENABLE_EXTERNAL_VALIDATION=false` **by default** (`validation/engine.py:72`).
When enabled, the following behavior:

| Validator | Inline-eligible | Keyed? | Always active |
|---|---|---|---|
| OFACValidator | no | local SDN | yes |
| NhtsaVpicValidator | **yes** | no | yes |
| ZippopotamValidator | **yes** | no | yes |
| NhtsaRecallsValidator | no | no | yes |
| NhtsaSafetyValidator | no | no | yes |
| NaicsValidator (census) | **yes** | no | yes |
| PhoneAreaValidator | no | CSV | yes |
| DnsMxValidator | no | no | yes |
| SecEdgarValidator | no | UA only | yes |
| CrossFieldValidator | no | no | yes |
| UspsValidator | no | key+secret | only if keys set |
| Tax1099Validator | no | key | only if key set |
| FmcsaValidator | no | key | only if key set |
| SamGovValidator | no | key | only if key set |

Inline runner (`accord_ai/validation/inline.py`) fires 3 inline validators
per turn when `ACCORD_INLINE_ENRICHMENT=true` (default on). Runner is wired
in `app.py:94-97` with a hardcoded 3-validator list — the key-gated
validators are **not** fed through the inline runner.

### 1d. Persistence
- All 6 tables migrated (store.py:28-121).
- DPO flow: collector → eligibility → dpo_prompt → DPOManager
  (`accord_ai/feedback/` — 7 files). Tables ship empty in `accord_ai.db`
  (correction memory DB is functional but unused in any live session).

### 1e. Flow engine (Phase 3.2 shipped)
- `accord_ai/conversation/flow_engine.py` (206 lines) — pure state machine
  over `flows.yaml` (271 lines, 11 flows).
- `FlowState` persisted via migration 5.
- Deterministic `next_action()` returning `ask` or `finalize`.
- Conditions: `field_set`, `field_equals`, `any_of`, `all_of`.

### 1f. OCR pipeline
- `accord_ai/extraction/ocr/` — engine.py + parser.py + errors.py.
- Wired into `/upload-image`; handles DL, insurance card, registration.

### 1g. Refiner cascade (built, disabled by default)
- `harness/refiner_cascade.py` + `refiner_clients/{gemini,claude,local}.py`.
- Gated: `ACCORD_DISABLE_REFINEMENT=true` default → returns `None`
  (`harness/refiner.py:137`). Privacy-safe.
- `HarnessManager` at `app.py:92` falls back to `Refiner(refiner_llm)` when
  cascade is None.

### 1h. Correction memory
- `accord_ai/feedback/memory.py:CorrectionMemory` — SQL-only, tenant-scoped,
  age-capped (`ACCORD_CORRECTION_MEMORY_MAX_AGE_DAYS=30`, limit=5).
- Injected into prompt via `extraction/extractor.py:113-135`
  (`_build_corrections_block`).
- Mechanism verified by unit tests (`tests/test_correction_memory.py`,
  `tests/test_extraction_correction.py`); DB is empty in practice.

---

## 2. Shipped but degraded / under-tested

### 2a. SYSTEM_V2 extraction prompt — what it says
`accord_ai/llm/prompts/extraction.py:49-82` (34 lines). Content:
- LOB routing rules (commercial_auto/general_liability/workers_comp).
- Six CRITICAL ROUTING RULES: vehicles under `lob_details`, not
  `additional_interests` (lienholders only); drivers not under
  `contacts`; garaging address on the vehicle, not top-level
  `locations`; `contacts[0].phone/email` takes precedence over root.
- Closes with "omit fields you do not know — do not use null, do not
  invent."

**Differs from v3's equivalent** (v3's harness `~2000 tokens`) — v4's
SYSTEM_V2 is a ~400-token LOB-routing skeleton, **stripped** of:
- Negation rules (now partially re-implemented as a post-extraction rule
  in `harness/rules/negation.py`)
- Correction recognition (now a separate regex branch +
  SYSTEM_CORRECTION_V1 prompt)
- Entity-type enum discipline
- Address parsing (Suite → line_two, PO Box rules)
- Numeric disambiguation
- Temporal / relative-date omission
- Loss-history discipline
- Cross-field contamination prevention

`SYSTEM_V3 = SYSTEM_V2 + "\n\n" + HARNESS_RULES` exists
(`extraction.py:88`) but is quarantined — production uses SYSTEM_V2
(`extractor.py:276`). This is the regression documented in
`PHASE_A_POSTMORTEM.md`: SYSTEM_V3 dropped bulk F1 36pt and failed
3 of 4 correction scenarios.

### 2b. Extraction modes
`accord_ai/config.py:11-16, 203`:
- `XGRAMMAR` (default, production): vLLM `guided_json` via xgrammar
- `JSON_OBJECT`: OpenAI `response_format=json_object`, no schema constraint
- `FREE`: no format constraint, parse first JSON block

All 3 modes are wired through `RetryingEngine` / `OpenAIEngine`. Mode is
stored on the Extractor at construction time (`extractor.py:175`) and
reported in the validity tracker. Step 25 matrix (A=xgrammar+none,
B/C/D=other combos) confirms XGRAMMAR is the best:

| Variant | F1 | Scenarios |
|---|---|---|
| A (xgrammar + harness none) | 0.713 | 55 |
| B | 0.595 | 55 |
| C | 0.573 | 55 |
| D | 0.604 | 55 |

### 2c. ExtractionContext — what's actually rendered
`accord_ai/extraction/context.py:12-37` (frozen dataclass):

| Field | Populated? | Rendered? |
|---|---|---|
| `current_flow` | yes | yes ("FLOW: {id}") |
| `expected_fields` | yes | yes ("FOCUS FIELDS: ...") |
| `question_text` | yes | yes ("QUESTION ASKED: ...") |
| `rag_snippets` | **reserved** | no (Phase 3.4 TODO) |

Controller builds context from prior `FlowState`
(`conversation/controller.py:265-290`) — so turn N's extraction sees what
flow asked in turn N-1. ENABLED by default (`extraction_context=True`).

### 2d. Harness Manager / Judge
- `SchemaJudge` returns `JudgeVerdict` with `passed`/`failed_paths`/
  `reasons`. Used both in the harness loop and the flow engine
  (`flow_engine.py` imports `_is_empty, _resolve` from judge).
- `HarnessManager` with `max_refines=1` (config.py:62) — runs once if
  judge fails, then returns regardless.

### 2e. Inline enrichment
- Wired with only 3 validators (`app.py:95`); inline eligibility flag
  lives on each validator but the *registry* at `app.py:95` is hardcoded —
  `ACCORD_INLINE_ENRICHMENT` toggles execution, not composition.

---

## 3. Reverted or abandoned

### 3a. Phase R (harness quarantined)
- `SYSTEM_V3 = SYSTEM_V2 + HARNESS_RULES` was production for Phase A,
  regressed, then reverted to SYSTEM_V2 in extractor
  (`extraction/extractor.py:267-276`). Harness content still lives in
  `llm/prompts/harness.py` and is **only** injected into the **refiner**
  path (`harness/refiner.py:33-48`, gated by `ACCORD_REFINER_HARNESS=1`
  default).
- A separate **frozen snapshot** of v3 harness lives in
  `extraction/harness_content/v3_harness_snapshot.py` — used only by the
  Step 25 `experiment_harness=full|light` A/B switch (default `none`).
  Step 25 B/C/D results confirm adding harness drops F1 further.

### 3b. Phase A (NER + FT/PT + correction tweaks)
- NER disabled in production extract path
  (`extraction/extractor.py:224-231`). Code still in `extraction/ner.py`
  with unit tests. Regressed multi-five-vehicle-fleet from F1 0.324 to
  0.121.
- `validate_extraction_with_ner` (Phase A step 6) never wired
  (`extractor.py:321-328`).
- FT/PT employee-split revert: not directly found in code; likely
  means `full_time_employees` + `part_time_employees` are treated
  independently by the schema; no split aggregator.
- Correction branch narrowing: when `is_corr=True` but
  `target=None`, fall through to SYSTEM_V2 (`extractor.py:246-256`) —
  fix applied after Step 3A post-mortem identified that the narrow
  prompt with full guided_json schema produced flat output that failed
  validation.

### 3c. Step 24 FOCUS FIELDS
- FOCUS FIELDS lines are **still active** in
  `extractor.py:145-146`. Step 24 may have backed out the "only extract
  focus fields" directive in favor of the current "prioritize focus
  fields; also extract any other fields" wording (`extractor.py:216-218`),
  but the focus-fields render itself was kept. No explicit postmortem
  doc.

### 3d. Other reverts / abandoned stubs
- `/review/{session_id}/resolve` stub (no body).
- `harness/brokers/` directory exists but is **empty** — intended for
  per-broker overlays, not started.

---

## 4. Feature flags inventory

| Env var | Source | Default | Recommended (best behavior) |
|---|---|---|---|
| `ACCORD_AUTH_DISABLED` | config.py:122 | False | False (disable only in dev) |
| `ACCORD_CHAT_OPEN` | config.py:123 | False | False (only for tunnel demos) |
| `ACCORD_ALLOW_EXTERNAL_LLM` | llm/__init__.py:68 | unset | unset (refiner localhost-only) |
| `ACCORD_DISABLE_REFINEMENT` | harness/refiner.py:137 | **true** | true (production; refiner off) |
| `ACCORD_REFINER_HARNESS` | harness/refiner.py:46 | 1 | 1 (harness in refiner is fine) |
| `ACCORD_INLINE_ENRICHMENT` | validation/inline.py:116 | **true** | true |
| `ENABLE_EXTERNAL_VALIDATION` | validation/engine.py:72 | **false** | true in production |
| `ENABLE_CORRECTION_MEMORY` | config.py:183 | True | True |
| `ENABLE_TRANSCRIPT_CAPTURE` | config.py:189 | True | True |
| `USE_FLOW_ENGINE` | config.py:193 | True | True |
| `EXTRACTION_CONTEXT` | config.py:197 | True | True |
| `EXTRACTION_MODE` | config.py:203 | xgrammar | xgrammar (Step 25 confirms) |
| `EXPERIMENT_HARNESS` | config.py:204 | none | **none** (Step 25 confirms) |
| `LLM_SEED` | config.py:209 | None | 42 for eval, None in prod |
| `WARMUP_ON_BOOT` | config.py:72 | False | True in production |
| `RATE_LIMIT_ENABLED` | config.py:89 | False | True for public deployments |
| `BACKEND_ENABLED` | config.py:142 | False | True in production |
| `DRIVE_ENABLED` | config.py:156 | False | True in production |
| `PII_REDACTION` | config.py:42 | True | True |

**Default-off flags that should be ON in production:**
`ENABLE_EXTERNAL_VALIDATION`, `WARMUP_ON_BOOT`, `RATE_LIMIT_ENABLED`,
`BACKEND_ENABLED`, `DRIVE_ENABLED`.

---

## 5. Known technical debt / open issues

Summary of `docs/DEFERRED_DEBT.md`:
- **D1** SQLite blocks asyncio event loop (`core/store.py`, ~30 call
  sites untreated).
- **D2** WAL not documented on `_conn()` — doc-only fix.
- **D3** CORS default `"*"` silently disables credentials
  (`config.py:124`, `api.py:2399-2404`).
- **D4** `/sessions?limit=0` or `-1` bypasses pagination (partial fix).
- **D5** TRACKER singleton non-atomic counters under concurrency
  (`llm/json_validity_tracker.py`).
- **D6** `/answer` may be missing auth dependency — unverified.
- **D7** `DPOManager._threshold` private-attr leak in API response
  (`api.py:2042`).

Additional from code scan:
- `knowledge/retriever.py:9` TODO — Phase 8.d retrieve-shape not final.
- `harness/brokers/` empty — intended but not started.
- `/review/{session_id}/resolve` stub.
- `rag_snippets` field in ExtractionContext reserved, never populated.
- RefiningCascade only enabled when `ACCORD_DISABLE_REFINEMENT=false` —
  DB on eval never gets refinement practice data; judge-rescue rate at
  3.3% in Step 25.A.

Performance signal from Step 25.A: `first_pass_rate=2.7%`,
`rescue_rate=3.3%`, `still_failing=94%`. I.e., the judge passes almost
nothing, so the "refiner rescue" loop is both rarely exercised and rarely
successful when it does fire.

---

## 6. Architecture decisions that may be wrong

### 6a. Validator inline-list is hardcoded
`app.py:94-97` hardcodes 3 validators in the inline runner. Key-gated
validators (USPS, Tax1099, FMCSA, SAM.gov) never appear inline because
they're constructed conditionally in `validation/engine.py:107-137` —
a **separate** code path. A single registry with `inline_eligible` as
the discriminator would match v3's pattern.

### 6b. Harness "principles in prompt" was designed out, then
tried again
v3's whole point was the harness constitution in the extractor prompt.
v4 started SYSTEM_V2 minimal, added SYSTEM_V3 (harness back), regressed,
reverted. The current design treats the harness as refiner-only — but
the eval shows the refiner is rarely invoked (3.3% rescue rate). So the
"harness principles" are effectively dead code on the hot path.

### 6c. Hardcoded inline validator timeout (2.0s)
`app.py:97` — not configurable without editing code.

### 6d. Empty `CorrectionMemory` + hardcoded limit
`app.py:100` builds `CorrectionMemory(db_path=settings.db_path)` without
passing the age/limit settings from `settings.correction_memory_limit`
or `settings.correction_memory_max_age_days`. Memory uses defaults
from its constructor — the config values are dead.

### 6e. SessionStore.submission deserializes full JSON per call
Every `get_session()` parses the full CustomerSubmission JSON. Under
load this is non-trivial CPU; v3 uses a lighter summary path.

### 6f. FlowEngine imports `_is_empty, _resolve` from
`harness.judge` (private helpers)
`conversation/flow_engine.py:31` — judge's private helpers are the
source of truth for "is this field set" logic. Any refactor of judge
will silently break flow-engine condition evaluation. A shared util
module would be cleaner.

---

## 7. Test coverage gaps

**Totals:** 111 test files, **1,671** test functions. Categories:

| Category | Files | Approx tests |
|---|---|---|
| API endpoints | 22 | ~260 |
| Validation (per source) | 13 | ~140 |
| Forms (mapper + filler + pipeline) | 13 | ~250 |
| Extraction (extractor, postprocess, NER, correction, context) | 6 | ~80 |
| Store / SQL | 4 | ~60 |
| Harness (judge, manager, refiner, rules) | 5 | ~75 |
| Flow engine + loader + schema + controller | 4 | ~70 |
| Eval (runner + scorer + path_map) | 3 | ~45 |
| Knowledge (chroma + embedder + retriever) | 5 | ~55 |
| Integrations (backend, drive) | 5 | ~50 |
| Logging / PII / audit / request_context | 4 | ~30 |
| Responder / Explainer / CLI | 3 | ~30 |
| Feedback (collector, memory, DPO, transcript) | 4 | ~60 |
| Other (public_surface, http_client, ...) | 15 | ~200 |

**Well-tested:**
- Forms mapper (13 focused test files, one per form family).
- Validators (one test per validator + integration test).
- Extraction correction/postprocess branches.
- Store migrations + logging paths.

**Poorly / thinly tested:**
- **End-to-end eval path** — only `test_eval_runner.py` with mocked engine.
  No fixture-sealed regression test that pins aggregate F1 to prevent
  silent quality regressions like Phase A.
- **Concurrency** — no test exercises `expected_updated_at` collision
  or the SQLite-blocks-asyncio hazard (D1).
- **Integration** — only 1 file: `tests/integration/test_smoke_backend_drive.py`.
  No integration test for the full /answer → /complete cycle.
- **Refiner cascade behavior** — `test_refiner_cascade.py` exists but
  `ACCORD_DISABLE_REFINEMENT=true` by default, so production path is
  untested end-to-end.
- **FlowEngine → Extractor context flow** — context building is unit-tested,
  but no test verifies a multi-turn flow where prior
  `flow_state_json` actually shapes turn-N extraction.
- **Step 25 experiment modes** (`JSON_OBJECT`, `FREE`) have no unit
  tests of their own — only measured through eval.
- **CORS credential handling** — no tests for D3 hazard.

---

## PRIORITY FIX CANDIDATES (ROI-ranked)

1. **Re-enable `ENABLE_EXTERNAL_VALIDATION=true` by default in
   production build path, and fold inline-validator composition into
   a single registry.** (1 day effort; unblocks 12 validators already
   written + tested; removes the `app.py:95` hardcode. High impact —
   the whole validation layer is currently gated off by default.)

2. **Decide: keep the harness-in-refiner pretence or delete it.**
   Step 25.A shows refiner rescue fires 3.3% and passes rarely. Either
   raise `harness_max_refines` and measure the rescue rate, or delete
   `SYSTEM_V3` + `harness_content/v3_harness_snapshot.py` + the
   `experiment_harness` flag and simplify the extractor. (2 hours for
   delete path; high clarity gain.)

3. **Fix SQLite-blocks-asyncio (D1) before any multi-tenant
   production rollout.** 30 store call sites, wrap with
   `asyncio.to_thread()` or migrate to aiosqlite. Not urgent at
   `concurrency=1` eval, but a latency cliff is guaranteed under real
   load. (1-2 days.)

4. **Pin an aggregate-F1 regression test** on a frozen 10-scenario
   subset with a mock engine that replays recorded LLM outputs. Phase A
   regressed 36pt silently because no such guardrail exists. (4 hours
   with existing eval infrastructure.)

5. **Wire `settings.correction_memory_limit` +
   `correction_memory_max_age_days` into `CorrectionMemory` at
   `app.py:100`.** Dead config values today; one-line fix. Also fix
   the `/review/{session_id}/resolve` stub so review resolution is an
   actual feature rather than silently accepting POSTs. (1 hour.)

---

**Audit closed.** Word count ≈ 2,300. Data as of commit on `dev/vals`,
2026-04-23.
