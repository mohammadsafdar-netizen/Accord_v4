# V3 Extraction Mechanisms — Deep Research for v4 Pivot

**Source files read** (all paths absolute, READ-ONLY):
- `/home/inevoai/Development/Accord-Model-Building/Custom_model_fa_pf/accord_ai_v3/accord_ai/extraction/prompts.py`
- `/home/inevoai/Development/Accord-Model-Building/Custom_model_fa_pf/accord_ai_v3/accord_ai/extraction/runner.py`
- `/home/inevoai/Development/Accord-Model-Building/Custom_model_fa_pf/accord_ai_v3/accord_ai/extraction/engine.py`
- `/home/inevoai/Development/Accord-Model-Building/Custom_model_fa_pf/accord_ai_v3/accord_ai/extraction/ner.py`
- `/home/inevoai/Development/Accord-Model-Building/Custom_model_fa_pf/accord_ai_v3/accord_ai/harness/{manager,judge,refiner,keyword_prejudge,schema_reference}.py`
- `/home/inevoai/Development/Accord-Model-Building/Custom_model_fa_pf/accord_ai_v3/accord_ai/knowledge/retriever.py`
- `/home/inevoai/Development/Accord-Model-Building/Custom_model_fa_pf/accord_ai_v3/accord_ai/core/config.py`
- `/home/inevoai/Development/Accord-Model-Building/Custom_model_fa_pf/accord_ai_v3/harness/core.md`
- `/home/inevoai/Development/Accord-Model-Building/Custom_model_fa_pf/accord_ai_v3/harness/lobs/{commercial_auto,general_liability}.md`
- `/home/inevoai/Development/Accord-Model-Building/Custom_model_fa_pf/accord_ai_v3/harness/harness.md` (legacy — **not loaded** when `core.md` exists)

---

## 1. System Prompt Text (EXACT)

`prompts.py:24-59` — `EXTRACTION_SYSTEM_PROMPT_BASE`:

```
You are an insurance data extraction engine. Output ONLY valid JSON — no prose.

═══ TODAY'S DATE ═══
Today is {today}. Use this to resolve relative dates:
- "next month" → the month AFTER the current month of THIS YEAR
- "in two weeks" → calculate from today
- "started 8 years ago" → {current_year} minus 8 = {eight_years_ago}
- "last year" → {last_year}
- NEVER output a date in the past for `effective_date` unless user explicitly states a past date.
- If you cannot confidently compute a date from vague language, OMIT the field.

Extract ALL entity data from the LATEST user message — business info, vehicles, drivers, coverages, loss history, prior insurance, etc. Return ONLY new or changed fields. If nothing new, output {}.
Do NOT repeat data already in CURRENT STATE unless the user explicitly corrected it.
IMPORTANT: Users frequently answer out of order or provide multiple topics at once. Extract EVERYTHING the user mentions, even if it does not match the question that was just asked.

═══ OUTPUT FORMAT ═══
- Valid JSON only. No markdown, no code fences, no explanation text.
- Dates: ALWAYS MM/DD/YYYY format.
- States: 2-letter USPS codes. Currency: digits only (no $ or commas).
- Booleans: true/false JSON values (not strings).
- If uncertain or not mentioned → OMIT the field. Never guess.

═══ ENTITY SEPARATION ═══
- loss_history, prior_insurance, additional_interests are TOP-LEVEL arrays.
- NEVER nest them inside "policy" or "operations".

═══ HALLUCINATION PREVENTION ═══
- NEVER invent values. Extract what was said, omit what wasn't.
- Do NOT echo schema defaults or examples as extracted values.

═══ SCHEMA (extract into these fields) ═══
{schema}

═══ EXTRACTION REMINDER ═══
Extract ALL entity data from the latest user message into the schema above. The user may provide vehicles, drivers, business info, or any other data — even if it does not answer the question just asked. Extract everything mentioned. Scan the ENTIRE message end-to-end: dates, radius, or policy info often appear at the END after vehicle/driver details. Do not stop early.
{harness}
```

The `{schema}` placeholder holds the **FULL** concatenated schema from `_SCHEMA_SECTIONS` (built by `_build_full_schema()`), not a trimmed one. Only the back-compat `build_correction_prompt` uses a trimmed schema. The `{harness}` placeholder is prefixed with literal string `\n═══ EXTRACTION HARNESS (learned principles) ═══\n` before the harness markdown content (see `_build_system_stable`, prompts.py:358-373).

---

## 2. Prompt Composition Order (CRITICAL FINDING)

**Harness is NOT a separate system message. It is CONCATENATED into the single SYSTEM message, at the END, AFTER the schema + reminder.** v4's port putting harness as a separate pre-`SYSTEM_V2` system message is structurally different.

`prompts.py:404-461` — `build_extraction_messages()`:

```
[0] SYSTEM  = base_preamble + dates + format + schema + reminder + harness
[1..N-2]    = prior USER/ASSISTANT turns (append-only)
[N-1] USER  = "═══ ALREADY COLLECTED ═══\n<state_summary>\n"
              + "═══ FIELD GUIDANCE (from knowledge base) ═══\n<RAG>\n"
              + "═══ QUESTION JUST ASKED (context only) ═══\n<last_q>\n"
              + "═══ LATEST USER MESSAGE ═══\n<user_message>"
```

Key design intent (quoted from docstring): *"Mutable data at the END keeps everything upstream cacheable; only the final turn pays the fresh-tokens cost on every turn."* The prefix-caching story is explicit — the SYSTEM message stays byte-identical across turns of the same session.

**Harness content placement inside SYSTEM** (prompts.py:366-373):
```python
harness_section = f"\n═══ EXTRACTION HARNESS (learned principles) ═══\n{harness_content}"
return EXTRACTION_SYSTEM_PROMPT_BASE.format(harness=harness_section, schema=_build_full_schema(), **date_ctx)
```

So the order inside SYSTEM is:
1. Base preamble ("You are an insurance data extraction engine...")
2. Today's date block
3. Extract-everything rules
4. Output format
5. Entity separation
6. Hallucination prevention
7. **Schema (full)**
8. Extraction reminder ("scan entire message end-to-end")
9. **Harness (core.md + active LOB files)** — appended last

The USER turn carries dynamic context (`ALREADY COLLECTED`, `FIELD GUIDANCE`, `QUESTION JUST ASKED`, `LATEST USER MESSAGE`). Prior turns are passed *as their original role* (not collapsed) — only last USER turn is augmented. Last-6 messages window (`context_window = 6`, runner:432).

---

## 3. Chat Template & vLLM Call Shape

`engine.py:117-129` — payload:

```python
payload = {
    "model": self.model,
    "messages": messages,
    "temperature": temp,
    "max_tokens": tokens,
    "chat_template_kwargs": {"enable_thinking": enable_thinking},  # default False on hot path
    "repetition_penalty": self.repetition_penalty,                  # 1.05
}
if self.structured_json and not enable_thinking:
    payload["response_format"] = {"type": "json_object"}
```

Hot path extraction: `temperature=0.0`, `max_tokens=3072` (clamped to 512 for <30-char messages, 4096 for >1500-char), `enable_thinking=False`, `response_format={"type": "json_object"}`. No `guided_json` / no JSON schema forcing — pure `json_object` mode. Output parsed via 7-strategy fallback (`parse_json`, engine.py:141-227): direct → regex `{...}` → `json_repair` → `ast.literal_eval` → trailing-comma strip → truncation repair → `{}` fallback. Also strips `<think>…</think>` blocks.

Judge uses same engine, `temperature=0.0`, `max_tokens=512`. Refiner (local Qwen fallback) uses `temperature=0.3`, `max_tokens=2048`.

---

## 4. Harness Content (core.md + LOB files)

**IMPORTANT FINDING**: `harness.md` at root is the LEGACY file. `manager.py:134-138` picks `core.md` if present, else falls back to `harness.md`. The active file is `/home/inevoai/Development/Accord-Model-Building/Custom_model_fa_pf/accord_ai_v3/harness/core.md` (v6.1 curated). If v4 is porting `harness.md`, it is porting the **wrong** file (older v1.0-v1.1 content from Round 4 testing, whereas core.md is v6.1 curated).

### core.md (7 sections, ~2090 chars)
- **1. Source Fidelity** — "Omit, never default", verbatim list items, corrections override
- **2. Entity & Attribute Routing** — semantic isolation, immediate-context binding, component granularity, per-vehicle attribution, contact role vs job title
- **3. Lists & Collections** — identity key required (VIN OR (year+make+model) for vehicles; full_name OR license_number for drivers; carrier_name for prior_insurance); "No implicit items"; List preservation rule ("omit the key — returning `[]` would wipe prior items"); state codes arrays
- **4. Negation & Exclusion** — has a TABLE (`| User said | Emit |`), explicit scope rules, paths listing 8 common booleans, **three full few-shot JSON examples**
- **5. Types & Formats** — Two-digit year disambiguation (past=DOB, future=policy), $1.2M → 1200000, entity type requires explicit legal form
- **6. Insurance Products vs Business Nature** — requests vs descriptions
- **7. Schema Adherence** — strict paths, `carrier_name` not `carrier`, `coverages` is dict not list, no cargo coverage field

### commercial_auto.md (appended when CA active, ~4800 chars)
- Vehicle count vs employee count keyword routing
- Use-type classification with 6 enum mappings
- Hazmat/trailer interchange (must be explicit)
- Hired vs non-owned distinction
- Coverage paths (CSL vs split, phys damage sub-dict, med_pay/towing/rental are FLAT scalars)
- Territory radius enum (local≤250, intermediate 250-500, long_haul>500)
- **7 numbered "DO NOT write" rules** listing every wrong path the refiner has historically invented

### general_liability.md (appended when GL active)
- `operations_description`, `products_sold`/`products_manufactured` distinction
- `subcontractor_usage`, `work_on_premises_pct`
- Class codes distinct from NAICS
- Coverage limits structure

### 5-10 hard-to-replicate rules

1. **List preservation rule (core.md §3)**: *"If nothing new belongs on an existing list, omit the key — returning `[]` would wipe prior items."* This is counter-intuitive — the LLM must know NOT to emit an empty array.
2. **Two-digit year directional disambiguation (core.md §5)**: *"Policy dates → future ('26'→'2026'). DOBs → past ('78'→'1978')."*
3. **Immediate-context binding (core.md §2)**: *"A deductible with no coverage context attaches to the most recently mentioned coverage or is omitted."* Stateful over the message.
4. **Silence ≠ false (core.md §4)**: triple distinction (silent=omit, denied=false, affirmed=true).
5. **Negation few-shot JSON examples (core.md §4)**: 3 exact input→JSON pairs in the harness — literal few-shot inside the instruction document.
6. **Combined negations split (core.md §4)**: `"No hired or non-owned auto"` → must set TWO different booleans.
7. **Entity type only on explicit legal form (core.md §5)**: "sole owner" does NOT become `individual`.
8. **`operations.trailer_interchange` = true ONLY when user says "interchange agreement"** (commercial_auto.md); "We have trailers" ≠ interchange.
9. **Radius enum mapping when conflict (commercial_auto.md)**: "When both descriptive ('regional') and mileage (400mi) provided and mileage clearly maps to an enum value, use the enum derived from mileage."
10. **7 explicit "DO NOT" antipatterns (commercial_auto.md §Common refiner mistakes)** — `coverages.cargo.*`, `coverages[0].limit`, `prior_insurance[N].carrier`, `business.*`, `auto_info.*`, arbitrary new top-level keys, `coverages.general_liability.limit`. These are negative examples baked into the prompt.

---

## 5. RAG Retrieval

Flow (runner.py:378-401):
1. `get_rag_queries_for_unpopulated_fields(current_state, last_user_msg)` returns ≤5 queries from a fixed map (`named_insured`→"business entity type tax id FEIN", `vehicles`→"vehicle VIN year make model garaging", etc). Only queries for sections that are BOTH unpopulated AND keyword-matched in the user message.
2. For each query: `retriever.retrieve_for_field(query, context=last_user_msg, k=3)` — calls ChromaDB via BAAI/bge-small-en-v1.5 embedder, `min_score=0.35`, merges tenant + `_shared` collections.
3. Dedupe by first 100 chars of content. Cap total at 8 results (`unique[:8]`).
4. **NER hints prepended as fake "RAG" chunk**: `knowledge_results.insert(0, {"content": ner_hints, "source": "ner", "chunk_type": "ner_hints"})`.
5. **Correction memory prepended** if present: `knowledge_results.insert(0, {"content": correction_context, "source": "correction_memory", ...})`.
6. TTL cache: 5 min, keyed by `(query, tenant, user_message[:100])`.

Rendered RAG block (prompts.py:322-335):
```
═══ FIELD GUIDANCE (from knowledge base) ═══
- <content trimmed to 200 chars>
- <content trimmed to 200 chars>
...
```
Max 8 bullets. Placed inside LAST USER turn, between `ALREADY COLLECTED` and `QUESTION JUST ASKED`.

Tenant scoping: `retriever.with_tenant(slug)` returns a shallow copy with different `_tenant`, searches `{tenant_slug}/` + `_shared/` ChromaDB collections, tenant ranked higher.

---

## 6. Refiner Cascade

**Trigger** (`refiner.py:660-685`):
1. Skipped entirely if `ACCORD_DISABLE_REFINEMENT=1` (eval mode).
2. `keyword_prejudge.prejudge(user_message, delta)` — cheap string-match check. Extracts VINs/ZIPs/EINs/dollar/date/year/phone/email/state/business-suffix from user text; checks each appears in `json.dumps(delta)`. **Skips LLM judge if all high-signal tokens present** (~60% reduction in judge calls). Empty delta ALWAYS fires.
3. LLM judge (Qwen3.5-9B, temp 0.0, 512 tokens, JSON mode) returns `{score: 0-10, issues: [{field, problem, detail, evidence}]}`. Needs-refinement threshold: **`score < 8` OR `len(issues) > 0`**.

**Routing** (`refiner.py:374-455`): Every issue path is scanned against LOB markers. If ALL issues fall within a single LOB's path namespace AND that LOB is active, target = LOB file. Otherwise target = `core`.

**Refiner cascade** (`refiner.py:718-735`):
1. **Gemini 2.5 Flash** (if `GOOGLE_API_KEY`/`GEMINI_API_KEY` set) — preferred
2. **Claude Opus** (if `ANTHROPIC_API_KEY` set) — fallback
3. **Local Qwen3.5-9B** — last resort, temp 0.3, 2048 max tokens

**What the refiner modifies**: it **REWRITES THE HARNESS**, not the submission. Input: full current harness content + issues text + user message (first 500 chars) + delta (first 1000 chars) + full v3 schema reference. Output: complete updated harness document. Never patches deltas.

**Save guardrails** (`manager.py:165-227`): 3 hard rejections:
1. `tokens > MAX_HARNESS_TOKENS (4000)`
2. Shrinkage > 25% of prior
3. Any `## Section` heading deleted

**Auto-rollback via replay** (`refiner.py:558-624`): After save, runs the original message against the NEW harness, re-judges. If residual issues + score<8 → `rollback_to_previous`, writes review queue entry. If no residual + score≥8 → "held". Provenance always logged to `harness/provenance.jsonl`.

**Rejection memoization**: 30-min TTL cache on `(target, frozenset(issue_keys))` — skip the whole LLM cycle if equivalent refinement was recently rejected (`manager.py:28-41`).

---

## 7. Non-Obvious Tricks (likely missing in v4)

Ranked by probable impact on the F1 gap:

1. **Full schema baked into SYSTEM, ALWAYS, with session-stable byte-identity** for prefix caching. `_build_full_schema()` dumps all 10 sections, not trimmed. (prompts.py:249-253). v4 porting "harness as separate system msg" breaks the caching AND changes the position relative to schema, meaning the model's attention pattern over harness rules relative to schema paths is different.

2. **`_ALWAYS_INCLUDE_SECTIONS = {"vehicles", "drivers", "loss_history", "coverages", "hired_non_owned"}`** (prompts.py:215) — list-typed sections always anchored in schema even when populated, to prevent "later turns emitting empty list that wipes previously extracted entities". This is combined with the core.md rule "omit the key — returning `[]` would wipe prior items".

3. **Phantom-item merge** (runner.py:108-190): A list item without any identity key (VIN/make/model for vehicles, full_name/license for drivers) is DROPPED — unless the current state has exactly one matching item, in which case the correction is MERGED into that item by inheriting its identity keys. Preserves corrections like "actually year is 2022" on single-vehicle sessions.

4. **NER post-extraction validation** (ner.py:227-443): After LLM returns, spaCy NER cross-validates. If `contact.full_name` is classified as ORG, it's DELETED. If `business_name` missing in delta AND session state AND NER found an ORG (with 80+ lines of heuristics rejecting vehicle makes, state lists, address-shaped candidates, jargon tokens, ZIP-containing strings, single words without legal suffix, etc), it's INJECTED. Website regex matching injects `named_insured.website`. This is a silent extraction booster that is entirely outside the LLM.

5. **Separate "correction prompt" path** (prompts.py:513-559, runner.py:404-422): `is_correction(user_message)` uses a regex (`actually|correction|wrong|mistake|change it to|i meant|...|oops|my bad|wait|hold on`) that triggers a COMPLETELY DIFFERENT prompt. Shows `CURRENT VALUES` as nested JSON, detects target field via keyword map (`"ein"→"named_insured.tax_id"`, etc.), emits "FIELD TO CORRECT: <path>" hint. The correction prompt does NOT include the scan-everything reminder — it's focused and surgical.

6. **Deterministic count post-processing** (orchestrator.py:551-562): After merge, `operations.vehicle_count = len(vehicles)` and `operations.driver_count = len(drivers)` IF list populated AND scalar not set. The LLM doesn't have to emit counts when it emits entities.

7. **Dot-key unfolding** (runner.py:54-75): LLM output `{"named_insured.tax_id": "12-3456789"}` is auto-nested. Prevents merge failures when model emits flat paths under pressure.

8. **State-list coercion** (runner.py:304-336): `"NE IA MO KS CO"` or `"Nebraska, Iowa, Missouri"` coerced to `["NE","IA","MO","KS","CO"]` at multiple keys (`operations.territory.states_of_operation`, `.registration_states`, `.states`, `hired_non_owned.hired.states`, etc). Handles both delimited strings and mixed lists.

9. **Format-validation strip** (runner.py:572-586): `validate_fields(delta)` returns format issues; any `Severity.ERROR` strips the field by path. Bad dates, bad phones, bad state codes SILENTLY removed, not patched.

10. **Schema grounding of refiner via `V3_SCHEMA_REFERENCE`** (schema_reference.py) — 200-line canonical reference of every valid field path, with 7 explicit "DO NOT write" rules. Injected into EVERY refiner prompt. Stops the refiner from inventing `coverages.cargo`, `business.contact_name`, `auto_info.hazmat`. If v4 doesn't supply this, its refiner will drift and the ported harness will accumulate schema-wrong principles.

11. **Composable harness (core + active-LOB)**: `manager.load(active_lobs)` composes `core.md + \n---\n + lobs/<lob>.md` for each active LOB. A CA-only session never sees GL rules. v4 porting a single `harness.md` blob loses this LOB targeting and the model gets cross-LOB interference.

12. **Pre-judge before LLM judge**: ~60% of turns short-circuit before Gemini/Claude is called.

13. **7-strategy JSON parse** (engine.py:141-227) including `json_repair`, `ast.literal_eval`, trailing-comma strip, truncation-repair with `}` suffixes. Silently recovers many "bad JSON" cases.

14. **Adaptive `max_tokens`** (runner.py:369-372): <30 char → 512; >1500 char → 4096; else 3072. Auto-clamped to context window (engine.py:106-115).

15. **Schema has list-cap commentary baked-in**: `_SCHEMA_LIST_CAP` (default 3) is frozen into the schema string at import time with text "Extract up to 3 vehicles per delta. The total fleet size goes into operations.vehicle_count regardless". Combined with Judge's instruction at judge.py:59-66 telling it NOT to flag vehicles[3]/drivers[3]/loss_history[3] as missed. **For eval reproduction, v3 flips `ACCORD_LIST_CAP=99`** — critical or bulk scenarios fail.

---

## 8. Hardcoded vs Configurable

**Hardcoded (in source, require code change)**:
- `EXTRACTION_SYSTEM_PROMPT_BASE` (prompts.py:24)
- `_SCHEMA_SECTIONS` (prompts.py:72-153)
- `_SECTION_KEYWORDS` (prompts.py:155-164)
- `_ALWAYS_INCLUDE_SECTIONS` (prompts.py:215)
- `_CORRECTION_RE` regex (prompts.py:171-179)
- `_CORRECTION_FIELD_HINTS` map (prompts.py:467-497)
- `_CAPPED_LISTS = ("vehicles", "drivers", "loss_history")` (runner.py:282)
- `_IDENTITY_KEYS` map (runner.py:123-130)
- Judge threshold `score < 8` (judge.py:90)
- `CHARS_PER_TOKEN = 2.8` (engine.py:44) / `3.0` in manager
- `MAX_HARNESS_TOKENS = 4000`, `TARGET_HARNESS_TOKENS = 3000`, `MAX_SHRINKAGE_PCT = 0.25` (manager.py:55-59, 165)
- `_RAG_CACHE` TTL 300s / maxsize 512 (runner.py:32)
- `_rejected_refinements` TTL 1800s (manager.py:28)
- `context_window = 6` for messages replay (prompts.py:431)
- vLLM payload: `temperature=0.0`, `repetition_penalty=1.05`, `response_format={"type":"json_object"}`, `chat_template_kwargs={"enable_thinking": False}`

**Configurable via env**:
- `ACCORD_LIST_CAP` (default 3; set to 99 for eval — **required for reproducing 99% L3**)
- `ACCORD_DISABLE_REFINEMENT` (default false; set true for eval/production to freeze harness + skip PII-to-Gemini)
- `ACCORD_AUTH_DISABLED`, `ACCORD_CHAT_OPEN` (auth, not extraction)
- `ACCORD_OCR_ALLOW_MOCK` (OCR)
- `VLLM_BASE_URL`, `AGENT_MODEL`, `AGENT_TEMPERATURE` (default 0.3 — note: extraction call overrides to 0.0), `AGENT_MAX_TOKENS` (default 4096), `MODEL_CONTEXT_LIMIT` (default 16384)
- `GOOGLE_API_KEY`/`GEMINI_API_KEY`, `ANTHROPIC_API_KEY` (refiner cascade)
- `VLLM_API_KEY` (bearer for vLLM HTTP)
- Logging: `LOG_DIR`, `LOG_LEVEL`, `LOG_FORMAT`, `PII_REDACTION`, rotation/retention knobs

Not exposed: schema trim behavior, system prompt text, NER rejection lists, correction regex, identity-key thresholds, phantom-merge policy.

---

## SURPRISING FINDINGS (for the pivot)

1. **v3's "harness" referenced in logs/docs is `harness/core.md`, NOT `harness/harness.md`**. The root `harness.md` is a LEGACY v1.1 file that only loads when `core.md` is missing (manager.py:134-138). v4 may have been porting the stale file. The active file is a v6.1 CURATED doc with substantially different structure (table-formatted negation rules, explicit JSON few-shots, 8 common-boolean path listing).

2. **Harness is NOT a separate system message** — it is concatenated at the END of the single SYSTEM message, AFTER the full schema and AFTER the "scan entire message end-to-end" reminder. v4's structure (harness as its own system turn before SYSTEM_V2) likely changes attention dynamics and breaks prefix caching. The Step-25 observation (ported harness drops 36pt on bulk) is consistent with (a) harness appearing in a position where it competes with rather than reinforces schema, and (b) losing the "always-included list-section anchoring" that partners with the harness "don't emit [] to wipe list" rule.

3. **Roughly half of extraction accuracy is NON-HARNESS scaffolding**: NER post-extraction fix-up (business_name/contact.full_name/website injection with 80+ lines of heuristic rejection), dot-key unfolding, state-list string coercion, phantom-item merge, deterministic count derivation, format-strip, 7-strategy JSON repair, separate correction-prompt path, 5-min RAG cache feeding `FIELD GUIDANCE` with NER hints and correction memory prepended as fake chunks. None of this is "the LLM prompt." A harness-only port will not recover it.

4. **The refiner has a canonical schema-reference that prevents hallucinated paths** (`schema_reference.py` — 200 lines including 7 DO-NOT patterns). Without this, the refiner (Gemini/Claude/local) invents plausible-wrong paths like `coverages.cargo.limit` and the extraction model then follows them. If v4 runs the refiner without the schema reference, the harness will actively degrade.

5. **Composable harness by active LOB** (`manager.load(active_lobs)`): only the active LOB's rules are loaded. A CA-only session never sees the GL harness. A v4 port of a single merged harness blob passes irrelevant-LOB rules on every call — they sit in context and bias the model toward wrong LOB fields. This likely explains part of the "bulk family drops 36pt" observation: if bulk scenarios span multiple LOBs and the ported harness is the full union, cross-LOB interference is maximal.
