# V3 Extraction Pipeline — Complete Code Walkthrough

**Scope:** Full code-level documentation of v3's extraction flow from user message arrival through delta integration into session state.

**Files covering this pipeline:**
- `extraction/runner.py` — orchestrator
- `extraction/prompts.py` — system/user prompt building
- `extraction/ner.py` — entity tagging (pre & post)
- `extraction/validators.py` — format validation
- `extraction/engine.py` — vLLM API client
- `extraction/__init__.py`, `extraction/external.py` (auxiliary)

---

## STAGE 1: INPUT ARRIVAL

### Entry Point: `run_extraction()` (runner.py:339-482)

**Function signature:**
```python
def run_extraction(
    engine: VLLMToolEngine,
    current_state: dict[str, Any],
    messages: list[dict[str, str]],
    harness_content: str = "",
    retriever: Any | None = None,
    last_question: str | None = None,
    max_tokens: int = 3072,
    correction_context: str | None = None,
    harness_manager: Any | None = None,
) -> dict[str, Any]:
```

**Called by:** orchestrator's main extraction handler (typically after user submits a message).

**Input types:**
- `engine`: vLLMToolEngine instance pointing to Qwen3-32b-instruct
- `current_state`: session entities dict (accrued across turns)
- `messages`: conversation history, list of `{"role": "user"/"assistant", "content": str}`
- `harness_content`: the self-improving extraction ruleset (set on session init or fetched from harness manager)
- `retriever`: optional RAG backend for field-level knowledge injection
- `last_question`: previous turn's question context (optional)
- `max_tokens`: initial output budget (adaptive at line 369-372)
- `correction_context`: prior turn's correction memory (optional)
- `harness_manager`: reference to harness layer for async judge/refiner (optional)

**Key validation (lines 356-366):**
```python
last_user_msg = ""
for msg in reversed(messages):
    if msg.get("role") == "user":
        last_user_msg = msg.get("content", "")
        break

alphanum = sum(1 for c in last_user_msg if c.isalnum())
if alphanum < 2:
    return {}  # Skip extraction on noise (fewer than 2 alphanumeric chars)
```

---

## STAGE 2: PRE-EXTRACTION — NER, RAG, CORRECTION MEMORY

### 2.1 NER Pre-tagging (runner.py:374-376)

**Code:**
```python
ner_tags = tag_entities(last_user_msg)
ner_hints = format_ner_hints(ner_tags)
```

**`tag_entities()` (ner.py:99-167):** Runs spaCy en_core_web_sm (CPU, ~12MB) + regex patterns.

**Returns dict with keys:**
- `persons` — detected person names via spaCy PERSON entity
- `orgs` — detected organization names via spaCy ORG entity
- `phones` — regex pattern `\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}`
- `emails` — regex pattern `[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}`
- `eins` — Federal Employer Identification Numbers `\d{2}-\d{7}`
- `vins` — Vehicle Identification Numbers `[A-HJ-NPR-Z0-9]{17}`
- `zips` — US ZIP codes `\d{5}(?:-\d{4})?`
- `websites` — URLs with TLD suffixes

**NER post-processing in tag_entities():**
- Reclassifies any PERSON that has `_ORG_SUFFIXES` pattern (LLC, Inc, Corp, etc.) → moves to orgs
- Validates person names with `_is_valid_person_name()` — rejects names <3 chars, without 2+ parts, with digits, or containing `_NAME_REJECT_WORDS` (the, a, an, etc.)

**`format_ner_hints()` (ner.py:192-220):** Formats tags as prompt-injectable text:
```
═══ NER ENTITY HINTS ═══
Detected PERSON names: John Smith, Jane Doe
Detected ORGANIZATION names: Acme Logistics LLC
Detected FEIN/EIN: 12-3456789
...
```

### 2.2 Adaptive max_tokens (runner.py:368-372)

**Rule:**
```python
if len(last_user_msg) < 30:
    max_tokens = min(max_tokens, 512)  # Short ack turns need less output
elif len(last_user_msg) > 1500:
    max_tokens = min(max_tokens, 4096)  # Bulk extraction can be verbose
```

This is empirically tuned from 12+ months of v3 production. Short inputs (e.g., "yes", "actually 2024") get 512 tokens to prevent hallucination; bulk dumps (e.g., full fleet + drivers + coverages) get 4096 to avoid truncation.

### 2.3 RAG Retrieval (runner.py:378-393)

**Code:**
```python
knowledge_results: list[dict[str, Any]] = []
if retriever is not None:
    rag_queries = get_rag_queries_for_unpopulated_fields(current_state, last_user_msg)
    for query in rag_queries:
        results = _cached_retrieve(retriever, query, last_user_msg, k=3)
        knowledge_results.extend(results)
    # Deduplicate by first 100 chars of content
    seen: set[str] = set()
    unique: list[dict] = []
    for r in knowledge_results:
        c = r.get("content", "")[:100]
        if c not in seen:
            seen.add(c)
            unique.append(r)
    knowledge_results = unique[:8]  # Cap at 8 unique results
```

**RAG query generation (prompts.py:293-319):**
```python
def get_rag_queries_for_unpopulated_fields(...) -> list[str]:
```

Maps unpopulated sections + message keywords to queries:
- `named_insured` unpopulated + relevant keywords → "business entity type tax id FEIN"
- `operations` unpopulated → "commercial auto operations fleet use type hazmat"
- `vehicles` unpopulated → "vehicle VIN year make model garaging"
- `drivers` unpopulated → "driver license CDL experience MVR"
- `coverages` unpopulated → "auto liability coverage CSL deductible UM UIM"
- `loss_history` unpopulated → "claims loss history accident occurrence"
- `prior_insurance` unpopulated → "prior carrier renewal premium"
- `hired_non_owned` unpopulated → "hired auto non-owned employee vehicles"

**RAG cache (runner.py:32-47):**
```python
_RAG_CACHE: TTLCache = TTLCache(maxsize=512, ttl=300)  # 5-minute TTL
key = (query, tenant, user_message[:100])
```

Deterministic query generation + message prefix prefix means the same RAG question repeated within a session (or across sessions from the same tenant) hits the cache. 10-30ms per fresh retrieval on CPU, so 5-minute TTL balances freshness against cost.

### 2.4 Correction Memory Injection (runner.py:399-401)

**Code:**
```python
if correction_context:
    knowledge_results.insert(0, {
        "content": correction_context,
        "source": "correction_memory",
        "chunk_type": "guidance"
    })
```

Correction memory (from prior turns, stored elsewhere) is injected at the head of the knowledge_results list. Tells the LLM: "Avoid these mistakes this time."

### 2.5 NER Hints Injection (runner.py:395-397)

**Code:**
```python
if ner_hints:
    knowledge_results.insert(0, {
        "content": ner_hints,
        "source": "ner",
        "chunk_type": "ner_hints"
    })
```

NER hints are prepended before RAG results, so the LLM sees detected entities first.

---

## STAGE 3: PROMPT BUILDING

### 3.1 Correction Detection (runner.py:403-422)

**Code:**
```python
is_corr = is_correction(last_user_msg)

if is_corr and current_state:
    prev_msg = ""
    for msg in reversed(messages[:-1]):
        if msg.get("role") == "user":
            prev_msg = msg.get("content", "")
            break
    system, user = build_correction_prompt(...)
    llm_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
else:
    llm_messages = build_extraction_messages(...)
```

**Correction regex (prompts.py:171-179):**
```python
_CORRECTION_RE = re.compile(
    r"\b(actually|correction|wrong|mistake|change\s+it\s+to|i\s+meant"
    r"|that(?:'s|\s+is)\s+(?:not\s+right|incorrect|wrong)"
    r"|please\s+(?:correct|fix|change|update)"
    r"|should\s+(?:be|read|say)"
    r"|oops|my\s+bad"
    r"|^(?:oh\s+)?wait\b|^hold\s+on\b)\b",
    re.IGNORECASE,
)
```

### 3.2 Correction Prompt (prompts.py:513-559)

If correction detected, use **focused correction prompt** instead of extraction prompt.

**System message:**
```python
system = f"""You are an insurance data correction engine. Output ONLY valid JSON.

Today is {date_ctx['today']}. Use this to resolve relative dates correctly.

The user is CORRECTING previously provided information. Extract ONLY the fields being changed.
Output the corrected values in the SAME nested JSON structure as the current values.
PRESERVE all other fields — do NOT re-extract or empty fields that aren't being corrected.
Output {{}} if you cannot determine what is being corrected.
{harness_section}"""
```

**User message includes:**
1. Schema
2. Current values (full current_state as JSON dump)
3. Previous context (prior turn's user message)
4. **Correction target hint** — auto-detected from keywords (prompts.py:500-510):

```python
_CORRECTION_FIELD_HINTS: dict[str, str] = {
    "ein": "named_insured.tax_id",
    "fein": "named_insured.tax_id",
    "business name": "named_insured.business_name",
    "entity type": "named_insured.entity_type",
    "effective date": "policy.effective_date",
    "phone": "named_insured.contact.phone",
    "dob": "drivers[N].dob",
    "license": "drivers[N].license_number",
    "vin": "vehicles[N].vin",
    "year": "vehicles[N].year",
    ...
}
```

If keyword matches, the prompt includes: `FIELD TO CORRECT: named_insured.tax_id`

**NOTE:** Keywords like "llc", "corporation", "s-corp" were **disabled** (prompts.py:478-480 comment) because they false-positive when the keyword appears in a business name (e.g., "Johnson LLC"), causing the correction to target entity_type instead of business_name.

### 3.3 Standard Extraction Prompt (prompts.py:404-461)

If NOT a correction, build standard extraction prompt via **`build_extraction_messages()`**.

**Returns OpenAI-style messages[] array with 3 parts:**

#### Part A: System Message (STABLE per session)

**`_build_system_stable()`** (prompts.py:358-373):
```python
def _build_system_stable(harness_content: str = "") -> str:
    harness_section = ""
    if harness_content:
        harness_section = f"\n═══ EXTRACTION HARNESS (learned principles) ═══\n{harness_content}"
    date_ctx = _build_date_context()
    return EXTRACTION_SYSTEM_PROMPT_BASE.format(
        harness=harness_section,
        schema=_build_full_schema(),
        **date_ctx,
    )
```

**Base system prompt (prompts.py:24-59):**
```
═══ TODAY'S DATE ═══
Today is {today}. Use this to resolve relative dates:
- "next month" → the month AFTER the current month of THIS YEAR
- "in two weeks" → calculate from today
- "started 8 years ago" → {current_year} minus 8
- "last year" → {last_year}
- NEVER output a date in the past for `effective_date` unless user explicitly states a past date.
- If you cannot confidently compute a date from vague language, OMIT the field.

Extract ALL entity data from the LATEST user message — business info, vehicles, drivers, coverages, loss history, prior insurance, etc. Return ONLY new or changed fields. If nothing new, output {}.

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

═══ SCHEMA (extract into these fields) ═══
{schema}

═══ EXTRACTION REMINDER ═══
Extract ALL entity data from the latest user message into the schema above. The user may provide vehicles, drivers, business info, or any other data — even if it does not answer the question just asked. Extract everything mentioned. Scan the ENTIRE message end-to-end: dates, radius, or policy info often appear at the END after vehicle/driver details. Do not stop early.
{harness}
```

**Key sections:**
- **Date resolution:** Explicitly instructs the LLM to use today's date as reference (today, current_year, last_year, eight_years_ago context injected)
- **Entity separation:** Loss history, prior insurance, additional interests are top-level arrays (NOT nested under policy/operations)
- **Scan reminder:** "Scan end-to-end" instruction to prevent stopping early when vehicles/drivers appear first
- **Schema:** Full CustomerSubmission schema (session-stable, same across all turns)
- **Harness injection:** The self-improving ruleset (if present)

**Schema structure (prompts.py:72-153):**

Full schema includes 8 sections:
1. `named_insured` — business info, contact, entity type, tax ID, address
2. `policy` — effective date, expiration date, status, policy number, premium
3. `operations` — fleet use, territory (states), safety, hazmat, vehicle/driver counts
4. `vehicles` (array, capped at `ACCORD_LIST_CAP` which defaults to 3 in production)
5. `drivers` (array, capped at 3)
6. `coverages` — liability, med pay, towing, rental, UM/UIM, physical damage
7. `hired_non_owned` — hired auto costs/days, non-owned employee count + states
8. `loss_history` (array, capped at 3)
9. `prior_insurance` (array)
10. `additional_interests` (array)

**ALWAYS_INCLUDE_SECTIONS (prompts.py:215):**
```python
_ALWAYS_INCLUDE_SECTIONS = {"vehicles", "drivers", "loss_history", "coverages", "hired_non_owned"}
```

These sections remain in the prompt schema even after they have data. **Reason:** Without the schema anchor, later turns may emit an empty list that wipes previously extracted entities. Keeping vehicles/drivers in the schema preserves the LLM's awareness of the list structure.

#### Part B & C: Conversation History + Mutable Final Turn

**Message layout in `build_extraction_messages()` (prompts.py:404-461):**

```python
context_window = 6  # Keep only recent 6 messages for context
recent = messages[-context_window:]

out: list[dict[str, str]] = [{"role": "system", "content": system}]

for i, m in enumerate(recent):
    role = m.get("role", "user")
    content = m.get("content", "")
    if i == last_user_idx:  # Final user message — append mutable state summary
        parts: list[str] = [
            "═══ ALREADY COLLECTED ═══",
            _summarize_state(current_state),
        ]
        rag_section = _build_rag_context(knowledge_results)
        if rag_section:
            parts.append("")
            parts.append(rag_section)
        if last_question:
            parts.append("")
            parts.append(f"═══ QUESTION JUST ASKED (context only) ═══\n{last_question}")
        parts.append("")
        parts.append("═══ LATEST USER MESSAGE ═══")
        parts.append(content)
        out.append({"role": "user", "content": "\n".join(parts)})
    else:
        out.append({"role": role, "content": content})

return out
```

**Prefix caching implication:**
- System message is **byte-identical** across turns within a session (full schema + harness never changes)
- vLLM caches this on turn 1; turns 2+ reuse the KV cache, paying only for new tokens
- Mutable data (state summary, RAG, final message) rides on the **final user message** to keep everything upstream static
- _summarize_state() converts nested dict to compact text (e.g., `vehicles: [3 items], named_insured: business_name=Acme Inc, phone=(555)123-4567`)

---

## STAGE 4: LLM CALL

### 4.1 vLLM API Call (engine.py:68-139)

**Engine initialization (engine.py:46-66):**
```python
class VLLMToolEngine:
    CHARS_PER_TOKEN = 2.8  # Calibrated for Qwen3
    
    def __init__(
        self,
        model: str,
        base_url: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        context_limit: int = 16384,
        structured_json: bool = True,
        repetition_penalty: float = 1.05,
    ):
```

**Generate call (runner.py:437-449):**
```python
raw = engine.generate(
    messages=llm_messages,
    temperature=0.0,        # Deterministic extraction
    max_tokens=max_tokens,  # Adaptive budget from stage 2
)
```

**vLLM request payload (engine.py:117-129):**
```python
payload: dict = {
    "model": self.model,
    "messages": messages,
    "temperature": temp,
    "max_tokens": tokens,
    "chat_template_kwargs": {"enable_thinking": enable_thinking},
    "repetition_penalty": self.repetition_penalty,
}
if self.structured_json and not enable_thinking:
    payload["response_format"] = {"type": "json_object"}
```

**Key parameters:**
- `temperature=0.0` — deterministic extraction (no sampling)
- `max_tokens` — adaptive from input length (512/2048/4096)
- `chat_template_kwargs`: `enable_thinking=False` for hot-path extraction (speed) vs `True` for cold-path judge (reasoning)
- `response_format={"type": "json_object"}` — forces JSON output (only when not using thinking mode)
- `repetition_penalty=1.05` — weak penalty to avoid token loops

**Auto-clamping (engine.py:106-115):**
```python
total_chars = sum(len(m.get("content", "") or "") for m in messages)
est_input_tokens = int(total_chars / self.CHARS_PER_TOKEN) + 100
max_allowed = self.context_limit - est_input_tokens - 64
if tokens > max_allowed >= 256:
    logger.info("Auto-clamped max_tokens %d → %d", tokens, max_allowed, ...)
    tokens = max_allowed
```

If prompt + requested max_tokens would exceed context window (default 16384), clamp down to fit. 64-token safety margin for EOS token + overhead.

---

## STAGE 5: OUTPUT PARSING

### 5.1 JSON Parse with 7 Fallbacks (engine.py:141-227)

**`parse_json()` method:**

Attempts 7 strategies in sequence:

1. **Direct json.loads()** — Most common path on valid LLM output
2. **Regex extraction `\{[\s\S]*\}`** — If LLM wrapped JSON in prose
3. **json_repair library** (if available) — Fixes missing quotes, trailing commas
4. **ast.literal_eval()** — Python literal syntax fallback
5. **Trailing comma removal** — Strips commas before `}` or `]`
6. **Truncated JSON repair** — If output ends abruptly, attempt repair + completion
7. **Empty dict fallback** — Return `{}` if all strategies fail

**Qwen3 thinking tag stripping (engine.py:157-158):**
```python
cleaned = re.sub(r"<think>[\s\S]*?</think>\s*", "", cleaned)
```

**Markdown code fence stripping (engine.py:159-161):**
```python
cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
cleaned = re.sub(r"\n?```\s*$", "", cleaned)
```

---

## STAGE 6: POSTPROCESSING

### 6.1 Orchestrated Postprocessing (runner.py:485-498)

**`_postprocess_delta()` function:**
```python
def _postprocess_delta(
    delta: dict[str, Any],
    current_state: dict[str, Any],
    ner_tags: dict[str, Any],
) -> dict[str, Any]:
    delta = _unfold_dot_keys(delta)
    delta = _strip_empty(delta)
    _drop_phantom_list_items(delta, current_state)
    _coerce_list_fields(delta)
    _cap_list_entries(delta)
    delta = validate_extraction_with_ner(delta, ner_tags, current_state=current_state)
    delta = _validate_formats(delta)
    return delta
```

**EXACT ORDER MATTERS** — each step depends on the state from the previous step.

### 6.2 Unfold Dot Keys (runner.py:54-75)

**Problem:** LLM sometimes outputs flat keys:
```json
{"named_insured.mailing_address.city": "Dallas"}
```

**Solution:**
```python
def _unfold_dot_keys(data: dict) -> dict:
    if not any("." in k for k in data):
        return data  # Fast path: no unfolding needed
    
    result = {}
    for key, value in data.items():
        if "." in key:
            parts = key.split(".")
            current = result
            for part in parts[:-1]:
                if part not in current or not isinstance(current[part], dict):
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            result[key] = value
    return result
```

Converts `{"a.b.c": v}` → `{"a": {"b": {"c": v}}}`. This normalizes the delta before further processing.

### 6.3 Strip Empty (runner.py:78-105)

**Recursively removes:**
- None values
- Empty strings (after strip())
- Empty dicts
- Empty lists
- List items that are None or empty string

```python
def _strip_empty(data: dict) -> dict:
    if not isinstance(data, dict):
        return data
    cleaned = {}
    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, dict):
            nested = _strip_empty(value)
            if nested:
                cleaned[key] = nested
        elif isinstance(value, list):
            filtered = []
            for item in value:
                if isinstance(item, dict):
                    nested_item = _strip_empty(item)
                    if nested_item:
                        filtered.append(nested_item)
                elif item is not None and item != "":
                    filtered.append(item)
            if filtered:
                cleaned[key] = filtered
        else:
            cleaned[key] = value
    return cleaned
```

Prevents downstream code from seeing partial empty structures (e.g., `{"named_insured": {}}` gets removed entirely).

### 6.4 Drop Phantom List Items (runner.py:108-191)

**Problem:** On corrections like "actually the year is 2022", the LLM emits:
```json
{"vehicles": [{"year": "2022"}]}
```

No VIN, make, or model — a phantom entity with no identity. These break downstream identity matching.

**Solution (runner.py:122-130):**
```python
_IDENTITY_KEYS: dict[str, list[str]] = {
    "vehicles": ["vin", "make", "model"],
    "drivers": ["full_name", "license_number"],
    "loss_history": ["description", "occurrence_date"],
    "prior_insurance": ["carrier_name"],
    "lienholders": ["name"],
    "locations": ["line_one", "city"],
}
```

For each list field, an item must have **at least one** identity key. If it doesn't:

**Case 1: Session has exactly 1 existing item of this type**
→ **Merge the phantom into the existing item**, carrying forward the existing identity (runner.py:154-176):
```python
if (
    isinstance(current_items, list)
    and len(current_items) == 1
    and isinstance(current_items[0], dict)
):
    inherited = {
        k: current_items[0].get(k)
        for k in identity_keys
        if current_items[0].get(k)
    }
    if inherited:
        merged = {**inherited, **item}
        surviving.append(merged)
        logger.debug("phantom-merge: %s[0] absorbed correction %s", key, list(item.keys()))
        continue
```

Example: Session has `vehicles: [{"vin": "123...", "make": "Ford", "year": "2020"}]`. User says "actually the year is 2022". LLM returns `{"vehicles": [{"year": "2022"}]}`. The phantom is merged back into the existing vehicle, becoming `{"vehicles": [{"vin": "123...", "make": "Ford", "year": "2022"}]}`.

**Case 2: Session has 0 or >1 items**
→ **Drop the phantom entirely** (runner.py:178-182).

### 6.5 Coerce List Fields (runner.py:304-337)

**Problem:** LLM sometimes returns state lists as delimited strings:
```json
{"operations": {"states_of_operation": "NE IA MO KS CO"}}
```
or
```json
{"operations": {"states_of_operation": "Nebraska, Iowa, Missouri"}}
```

**Solution:** `_normalize_state_list()` coerces strings into arrays of 2-letter USPS codes (runner.py:213-273).

**Accepts:**
- JSON list: `["NE", "IA"]` or `["Nebraska", "Iowa"]`
- Delimited string: `"NE IA MO"` or `"NE, IA, MO"` or `"Nebraska, Iowa"`
- Mixed: `["NE", "Iowa", "MO"]`

**Returns:** Deduplicated list of uppercase 2-letter codes, or None.

**Fallback logic:**
1. Try full state name match (`"Nebraska"` → `"NE"`)
2. Try 2-letter code uppercase fallback (`"ne"` → `"NE"`)
3. Drop unknown tokens

**State map (runner.py:193-208):**
```python
_STATE_NAME_TO_CODE: dict[str, str] = {
    "alabama": "AL", "alaska": "AK", ..., "wyoming": "WY",
    "district of columbia": "DC", "dc": "DC", "washington dc": "DC",
}
```

**Applied to:**
- `operations.states_of_operation`
- `operations.registration_states`
- `operations.territory.states_of_operation`
- `operations.territory.registration_states`
- `operations.territory.states`
- `operations.hired_non_owned.hired.states`
- `operations.hired_non_owned.non_owned.states`

### 6.6 Cap List Entries (runner.py:285-301)

**Hard cap on inline list extraction (runner.py:281-282):**
```python
_MAX_INLINE_LIST_ENTRIES = int(os.environ.get("ACCORD_LIST_CAP", "3"))
_CAPPED_LISTS = ("vehicles", "drivers", "loss_history")
```

Default 3 in production (set high in benchmarks to avoid scoring penalties). If LLM returns >3 vehicles, trim to first 3:
```python
def _cap_list_entries(delta: dict[str, Any]) -> None:
    for key in _CAPPED_LISTS:
        lst = delta.get(key)
        if isinstance(lst, list) and len(lst) > _MAX_INLINE_LIST_ENTRIES:
            dropped = len(lst) - _MAX_INLINE_LIST_ENTRIES
            logger.info(
                "Capping %s at %d entries (dropped %d inline, use upload for overflow)",
                key, _MAX_INLINE_LIST_ENTRIES, dropped,
            )
            delta[key] = lst[:_MAX_INLINE_LIST_ENTRIES]
```

**Rationale:** The extraction prompt already instructs the LLM to emit max 3, but this is a defence-in-depth safety net. On generated/bulk scenarios the LLM sometimes ignores the instruction. The overflow path is document upload / one-at-a-time entry.

### 6.7 NER Post-Validation (runner.py:496 calls ner.py:227-443)

**`validate_extraction_with_ner()`** applies learned NER classifications to fix LLM mistakes:

**Fix 1: Contact name is actually an ORG**
If `contact.full_name` matches a detected ORG (and NOT a detected PERSON) → remove it:
```python
if contact_lower in detected_orgs and contact_lower not in detected_persons:
    logger.info("NER: Removing contact_name '%s' — classified as ORG, not PERSON", contact_name)
    contact.pop("full_name", None)
```

**Fix 1b: Contact name has ORG suffix**
If `contact.full_name` has a legal suffix (LLC, Inc, Corp, etc.) → remove it:
```python
elif _ORG_SUFFIXES.search(contact_name):
    logger.info("NER: Removing contact_name '%s' — has ORG suffix", contact_name)
    contact.pop("full_name", None)
```

**Fix 2: No contact name extracted, but NER found a PERSON**
If delta has no `contact.full_name` but NER found person names:
```python
if isinstance(contact, dict) and not contact.get("full_name"):
    if detected_persons:
        best_person = max(ner_tags.get("persons", []), key=len, default=None)
        if best_person and best_person.lower() != biz_name.strip().lower():
            contact["full_name"] = best_person
            logger.info("NER: Suggested contact_name '%s' from NER PERSON entity", best_person)
            if "contact" not in named:
                named["contact"] = contact
```

**Fix 3: No business_name ANYWHERE (delta AND session), but NER found an ORG**

This is the most aggressive fix. Injects business_name from NER ORG entity **only if**:
- Delta has no business_name
- Session state has no business_name (prevents overwriting prior correct extraction with NER false positive)
- NER detected ORG entities

Then applies aggressive junk filtering (ner.py:351-418):

**Rejection rules:**
- **_BARE_REJECT set:** eine, fein, ssn, vin, dot, cdl, naics, sic, etc. (identity codes, not business names)
- **_VEHICLE_MAKES:** ford, chevy, gmc, kenworth, etc. — spaCy frequently tags these as ORG
- **_JARGON_TOKENS:** cov, coverage, bi, pd, um, uim, policy, claim, driver, vehicle, etc.
- **Digit ratio >30%:** Likely a doc code, not a business name
- **Single-word without legal suffix:** Too ambiguous (city? family name? product?)
- **ID-like pattern:** "CDL-A W123", "DL# 12345", "VIN 1HGB", "MC# 98765"
- **Address-shaped:** Contains 5-digit ZIP or "MI 49684" pattern
- **Pure state codes:** "MO KS CO TX" (territory list, not a business name)
- **All caps + digits + no legal suffix >40% chars:** Likely a license plate

**Best ORG selection:**
```python
candidates = [o for o in ner_tags.get("orgs", []) if _looks_like_business(o)]
best_org = max(candidates, key=len, default=None) if candidates else None
if best_org:
    named["business_name"] = best_org
    logger.info("NER: Suggested business_name '%s' from NER ORG entity", best_org)
```

Uses longest surviving candidate (more likely a full name vs. abbreviation).

**Fix 4: Website detected but not extracted**
If NER regex found URLs but delta has no website:
```python
detected_websites = ner_tags.get("websites", [])
if detected_websites and not named.get("website"):
    best = detected_websites[0]
    biz_key = re.sub(r"[^a-z0-9]", "", biz_name.lower()) if biz_name else ""
    if biz_key:
        for w in detected_websites:
            wk = re.sub(r"[^a-z0-9]", "", w.lower())
            if biz_key[:6] and biz_key[:6] in wk:
                best = w
                break
    named["website"] = best
    logger.info("NER: Injected website '%s' from URL regex", best)
    delta["named_insured"] = named
```

Prefers a domain matching the business name when multiple websites detected.

### 6.8 Format Validation (runner.py:572-586)

**`_validate_formats()`** applies field-level format validators (validators.py).

```python
def _validate_formats(delta: dict) -> dict:
    from .validators import validate_fields
    from ..core.types import Severity
    
    issues = validate_fields(delta)
    for issue in issues:
        if issue.severity == Severity.ERROR:
            _strip_by_path(delta, issue.field_path)
            logger.info("Format validation stripped %s: %s", issue.field_path, issue.message)
    return delta
```

Only strips **ERROR**-level issues (hard schema violations). **WARNING**-level issues are logged but not stripped (e.g., malformed email, suspicious phone).

**Field validators (validators.py:400-418):**
```python
_FIELD_VALIDATORS = {
    "email": validate_email,
    "phone": validate_phone,
    "state": validate_state_code,
    "zip_code": validate_zip,
    "tax_id": validate_fein,
    "effective_date": validate_date_format,
    "expiration_date": validate_date_format,
    "business_start_date": validate_date_format,
    "dob": validate_dob,
    "hire_date": validate_date_format,
    "occurrence_date": validate_date_format,
    "vin": validate_vin,
    "naics": validate_naics,
    "amount": validate_loss_amount,
    "paid_amount": validate_loss_amount,
    "reserve_amount": validate_loss_amount,
}
```

**Key validators:**

**Phone (validators.py:47-79):**
- Extract 10 digits (strip country code if 11 digits starting with 1)
- Reject if ≠10 digits
- Reject all-same-digit (1111111111) or reserved codes (000, 911)
- WARNING severity (doesn't strip, just flags)

**State (validators.py:96-115):**
- Must be 2-letter USPS code
- Case-insensitive match against US_STATES
- WARNING severity

**ZIP (validators.py:126-154):**
- Must be 5-digit or ZIP+4 format
- Reject placeholder prefixes (000, 099, 999) or all-same-digit
- WARNING severity

**FEIN (validators.py:175-217):**
- Format: XX-XXXXXXX
- Auto-insert hyphen if missing (e.g., "123456789" → "12-3456789")
- Reject prefix 00 (IRS never assigns this) — **ERROR**
- Reject patterns like 123456789, 999999999, 000000000 — **ERROR**
- Reject all-same-digit — **WARNING**
- WARNING on format issues

**Date (validators.py:227-271):**
- Format: MM/DD/YYYY or bare YYYY (accepted for business_start_date)
- Auto-swap DD/MM if month > 12 and day ≤ 12 (European format detection)
- Validate as real calendar date
- WARNING severity

**DOB (validators.py:278-317):**
- Must be valid date (delegates to validate_date_format)
- Must be 18+ (ERROR if <18)
- Flag if >100 years old (WARNING)
- ERROR for underage drivers

**VIN (validators.py:327-347):**
- Must be 17 characters
- Cannot contain I, O, Q (invalid in VINs)
- WARNING severity

**NAICS (validators.py:354-365):**
- Must be 2-6 digit numeric
- WARNING severity

**Loss amount (validators.py:372-393):**
- Must be numeric (strip $ and commas)
- Must be non-negative
- WARNING severity

---

## STAGE 7: JUDGE/REFINER ASYNC (Cold Path)

### 7.1 LOB Inference (runner.py:501-569)

**`_infer_active_lobs()`** detects which LOBs are active based on merged state + delta:

```python
def _infer_active_lobs(
    current_state: dict[str, Any],
    delta: dict[str, Any],
) -> list[str]:
    lobs: list[str] = []
    
    # Merge current_state + delta for inference
    merged: dict[str, Any] = {}
    for k, v in (current_state or {}).items():
        merged[k] = v
    for k, v in (delta or {}).items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = {**merged[k], **v}
        else:
            merged[k] = v
    
    coverages = merged.get("coverages") if isinstance(merged.get("coverages"), dict) else {}
    
    # Commercial Auto
    if merged.get("vehicles") or merged.get("drivers") or merged.get("operations") \
            or any(k in coverages for k in (
                "auto_liability", "physical_damage", "cargo",
                "hired_auto", "non_owned_auto", "medical_payments",
                "uninsured_motorist",
            )):
        lobs.append("commercial_auto")
    
    # General Liability
    if any(k in coverages for k in (
            "general_liability", "products_liability",
            "personal_injury", "completed_operations",
    )):
        lobs.append("general_liability")
    
    # Workers Compensation
    if any(k in coverages for k in (
            "workers_compensation", "employers_liability",
    )) or merged.get("class_codes") or merged.get("payroll"):
        lobs.append("workers_compensation")
    
    # Commercial Property
    if any(k in coverages for k in (
            "property", "building", "bpp",
            "business_personal_property", "business_income",
    )) or merged.get("locations") or merged.get("construction_info"):
        lobs.append("commercial_property")
    
    # Umbrella
    if any(k in coverages for k in ("umbrella", "excess_liability")):
        lobs.append("commercial_umbrella")
    
    # Cyber
    if "cyber_info" in merged or any(k in coverages for k in (
            "cyber", "data_breach", "privacy_liability", "network_security",
    )):
        lobs.append("cyber")
    
    # Directors & Officers
    if any(k in coverages for k in (
            "directors_officers", "management_liability",
            "employment_practices", "fiduciary_liability",
    )):
        lobs.append("directors_officers")
    
    # BOP
    if any(k in coverages for k in ("bop", "business_owners")):
        lobs.append("bop")
    
    return lobs
```

Inference is **shape-based**, not by explicit LOB list. Looks for:
- Presence of vehicles/drivers/operations → commercial_auto
- Coverage keys → respective LOB
- Supporting data (class_codes → workers_comp, locations → property, cyber_info → cyber)

### 7.2 Async Judge/Refiner (runner.py:611-636)

**Fired in background thread after delta is returned (cold path):**
```python
def _fire_judge_async(
    user_message: str,
    delta: dict,
    current_state: dict,
    harness_manager: Any,
    engine: VLLMToolEngine,
    active_lobs: list[str] | None = None,
) -> None:
    from ..harness.refiner import judge_and_refine

    def _run():
        try:
            judge_and_refine(
                user_message=user_message,
                delta=delta,
                current_state=current_state,
                harness_manager=harness_manager,
                engine=engine,
                active_lobs=active_lobs or [],
            )
        except Exception as e:
            logger.error("Async judge/refine failed: %s", e)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
```

Judge/Refiner runs **asynchronously** — it does NOT block the return of the delta. The main extraction thread returns the delta immediately (line 482), and the judge/refiner starts in the background. If the judge finds issues (schema violations, hallucinations, contradictions with harness), it updates the harness for future turns (a learning mechanism).

---

## STAGE 8: INTEGRATION INTO SESSION STATE

After `run_extraction()` returns delta, the **orchestrator** (not shown here, lives in higher layer) merges the delta into session.entities.

**Merge strategy** (outside v3 extraction, but informed by v3's phantom merge logic):
- Fields in delta replace corresponding fields in current_state
- Phantom items with identity inheritance are merged into existing single items (handled by _drop_phantom_list_items)
- Empty values are stripped by _strip_empty before delta returns, so no null pollution

---

## COMPARISON WITH V4

V4 (`accord_v4/accord_ai/extraction/extractor.py`) restructures around:

1. **Unified schema:** Uses Pydantic `CustomerSubmission` model as single source of truth (not string-based schema sections)
2. **Split postprocess:** Separates `postprocess.py` module (runner doesn't include all steps inline)
3. **Explicit correction handling:** `correction.py` module dedicated to correction detection + prompt building
4. **Context-aware extraction:** `ExtractionContext` object (flow, expected fields, etc.) passed to extraction
5. **Adaptive max_tokens:** Same logic as v3 (lines 84-106 of v4 extractor.py match v3 runner.py:368-372)
6. **Async judge/refiner:** Same architecture (called after delta return, not blocking)
7. **NER:** Still uses spaCy but module imports suggest it was **disabled in production** (comment on line 39-41: "NER imports: kept only in tests")

**V3 → V4 migration notes:**
- All postprocessing steps remain (unfold, strip empty, phantom merge, coerce lists, cap, validate)
- Harness integration is identical (system prompt includes self-improving ruleset)
- vLLM engine parameters are the same (temperature=0, structured_json=True, repetition_penalty=1.05)
- RAG retrieval logic is preserved (same query generation, TTL caching)
- Correction prompt is slightly reformatted but logic is unchanged

---

## LEARNED LESSONS & NON-OBVIOUS TRICKS

### 1. Scan end-to-end reminder in system prompt
The extraction system prompt explicitly reminds the LLM: "Scan the ENTIRE message end-to-end: dates, radius, or policy info often appear at the END after vehicle/driver details. Do not stop early."

**Why:** Early v3 production observed the LLM stopping after extracting the first few vehicles, missing loss history / prior insurance mentioned at the end of multi-paragraph messages.

### 2. Correction field hints disabled for entity type keywords
Keywords like "llc", "corporation", "s-corp" were **removed** from `_CORRECTION_FIELD_HINTS` (prompts.py:478-480) because they false-positive when embedded in a business name (e.g., "Johnson LLC" shouldn't trigger entity_type correction).

### 3. ALWAYS_INCLUDE_SECTIONS prevents wipeout
Keeping vehicles/drivers/loss_history/coverages/hired_non_owned in the schema **even after they have data** prevents later turns from emitting empty lists that would wipe previously extracted entities. Without this anchor, a turn saying "no changes" could accidentally clear all vehicles.

### 4. Phantom merge logic for 1-item case
If the LLM returns a phantom item (no identity) and the session has exactly 1 existing item of that type, the phantom is merged back into the existing item (inheriting its identity). This rescues corrections like "the year is 2022" that omit VIN/make/model.

**Implementation:** _drop_phantom_list_items() checks `len(current_items) == 1` before merging (line 157).

### 5. NER is conservative on business name injection
NER's business name suggestion (_looks_like_business function) applies **aggressive junk filtering** to reject spaCy ORG false positives. The function checks for vehicle makes, identity codes (CDL, VIN, etc.), address patterns, state-code lists, and jargon tokens. This prevents "VIN 1HGBH41" or "MO KS CO" from polluting business_name.

### 6. State normalization handles multi-word names
_normalize_state_list() splits on delimiters (comma, slash, semicolon) then further splits on spaces within each token. This preserves "new york" → "NY" and "new jersey" → "NJ" in mixed input like "New York, new jersey, ne".

### 7. JSON parse has 7 fallbacks
The vLLM engine doesn't assume well-formed JSON. It tries direct parse, then regex extraction, json_repair, ast.literal_eval, trailing comma removal, truncated repair, and finally empty dict. This handles markdown wrapping, Qwen3 thinking tags, and partial output.

### 8. Prefix caching exploited for session-stable system message
The system message includes the **full** schema and harness (same across all turns). vLLM caches this entire message on turn 1, so turns 2+ reuse the KV cache. Only mutable data (state summary, RAG results, latest user message) rides on the final user message to keep everything upstream static.

### 9. RAG cache TTL = 5 minutes
RAG retrieval is cached for 5 minutes per (query, tenant, message-prefix). Deterministic query generation means the same unpopulated section + user message always generates the same query. 5-minute TTL balances freshness against embedding/search cost (~10-30ms on CPU).

### 10. Adaptive max_tokens prevents hallucination & truncation
Short inputs (<30 chars) get 512 tokens (acknowledgements don't need verbose output). Bulk inputs (>1500 chars) get 4096 tokens (multi-vehicle/driver JSON is large). Normal turns get 2048. This prevents both over-generation on short turns and truncation on long turns.

### 11. Format validation is schema enforcement only, not a guard
Format validation strips fields with ERROR severity (invalid FEIN prefix, underage driver DOB, etc.), but does NOT block the delta. The harness handles business logic guards (e.g., "if no named_insured, reject"). Extraction format validation is just "dates are dates, phones have 10 digits, states are 2-letter codes."

### 12. Judge/refiner runs async, doesn't block extraction return
The judge/refiner is fired in a background daemon thread (line 635 of runner.py). It does NOT block the extraction function's return. If the judge finds issues, it updates the harness for future turns (self-improvement mechanism). The extraction API returns the delta immediately, unaware of judge/refiner results.

### 13. No retry on invalid JSON in extraction layer
If JSON parsing fails on the first LLM call, the extraction layer returns `{}` (empty delta) — it does NOT retry. Retry semantics are owned by the orchestrator layer, not the extraction layer.

### 14. Temperature is hardcoded at 0.0
All vLLM calls from extraction use `temperature=0.0` for deterministic output. The only exception is the cold-path judge/refiner (enable_thinking=True), which may use higher temp for multi-step reasoning.

### 15. Chat template kwargs enables thinking mode only on cold path
The payload includes `"chat_template_kwargs": {"enable_thinking": enable_thinking}`. This is False for hot-path extraction (speed) and True for judge/refiner (reasoning). When thinking is enabled, response_format masking is skipped (line 128) because `<think>` prefix would be suppressed.

---

## FILES & LINE NUMBERS QUICK REFERENCE

| Function | File | Lines |
|----------|------|-------|
| `run_extraction()` | runner.py | 339-482 |
| `_postprocess_delta()` | runner.py | 485-498 |
| `_unfold_dot_keys()` | runner.py | 54-75 |
| `_strip_empty()` | runner.py | 78-105 |
| `_drop_phantom_list_items()` | runner.py | 108-191 |
| `_normalize_state_list()` | runner.py | 213-273 |
| `_coerce_list_fields()` | runner.py | 304-337 |
| `_cap_list_entries()` | runner.py | 285-301 |
| `_validate_formats()` | runner.py | 572-586 |
| `_infer_active_lobs()` | runner.py | 501-569 |
| `_fire_judge_async()` | runner.py | 611-636 |
| `tag_entities()` | ner.py | 99-167 |
| `format_ner_hints()` | ner.py | 192-220 |
| `validate_extraction_with_ner()` | ner.py | 227-443 |
| `build_extraction_messages()` | prompts.py | 404-461 |
| `build_correction_prompt()` | prompts.py | 513-559 |
| `is_correction()` | prompts.py | 182-186 |
| `_build_system_stable()` | prompts.py | 358-373 |
| `_build_full_schema()` | prompts.py | 251-253 |
| `get_rag_queries_for_unpopulated_fields()` | prompts.py | 293-319 |
| `VLLMToolEngine.generate()` | engine.py | 68-139 |
| `VLLMToolEngine.parse_json()` | engine.py | 141-227 |
| `validate_fields()` | validators.py | 421-445 |
| `validate_phone()` | validators.py | 47-79 |
| `validate_fein()` | validators.py | 175-217 |
| `validate_dob()` | validators.py | 278-317 |

