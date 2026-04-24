# LIBRARIES_EXTRACTION.md

Evaluation of 2026 production-ready LLM structured-extraction libraries for accord_v4. Research date: 2026-04-23.

---

## Executive Summary

**Recommendation: Keep v3's custom pipeline as the skeleton. Bolt on two libraries for the parts it does poorly.**

1. **For constrained decoding at the vLLM layer**, use **vLLM's native `structured_outputs` / `guided_json`** (already in the inference server you're running — no new dependency, schema enforced at token-sampling time, works with multi-LoRA per-request). [vLLM docs](https://docs.vllm.ai/en/latest/features/structured_outputs/)
2. **For permissive JSON parse + repair**, use **BAML's Schema-Aligned Parsing (SAP)** as a drop-in replacement for v3's 7-strategy parse chain. SAP handles markdown fences, trailing commas, chain-of-thought prefixes, and type coercion in <10 ms without an LLM retry. [BAML repo](https://github.com/BoundaryML/baml), Apache-2.0. [SAP explanation](https://medium.com/@rajkundalia/how-baml-brings-engineering-discipline-to-llm-powered-systems-983c06d31bf8)

**Do NOT adopt** Instructor, Outlines, DSPy, Marvin, Mirascope, or pydantic-ai wholesale as the pipeline driver. They impose their own control flow and prompt composition that will collide with v3's harness injection, RAG injection, state summarization, and correction branching. The 71% regression in v4 is a symptom of a library (whichever one it is) capturing the control flow; more library is not the answer.

**The single highest-leverage change**: switch extraction calls from open-ended JSON-mode to vLLM `response_format: {"type": "json_schema", "json_schema": {...}}` with the CustomerSubmission Pydantic schema. That alone eliminates most of the "LLM returned a string where a list was expected" class of bugs that v3's postprocess chain patches up after the fact.

---

## Per-Library Evaluation

### 1. Instructor (jxnl / 567-labs)

- **Repo / metrics**: 12.8k stars, v1.15.1 (2026-04-03), MIT, 3 M monthly PyPI downloads, actively maintained ([GitHub](https://github.com/567-labs/instructor), [PyPI](https://pypi.org/project/instructor/))
- **What it does**: Patches an OpenAI-compatible client so `client.chat.completions.create(response_model=MyPydantic, max_retries=3)` returns a validated Pydantic instance. On validation failure, it re-prompts the model with the Pydantic error.
- **vLLM**: Yes — via OpenAI-compatible transport; `Mode.JSON` works with any vLLM endpoint. Not explicitly featured in the main integrations page but known-working in the wild ([integrations list](https://python.useinstructor.com/integrations/) mentions Ollama + llama-cpp; vLLM piggybacks on the OpenAI client patch).
- **Pydantic v2**: Yes, native ([repo topics](https://github.com/567-labs/instructor) include `pydantic-v2`).
- **Does it solve v3's problems?**
  - Permissive JSON parse: **Partial.** Relies on the model's own JSON mode + one retry round; no SAP-style recovery.
  - Prompt composition: **No.** It *owns* the message list. Injecting the harness + RAG + correction branch means either (a) stuffing them all into the system prompt before Instructor runs, or (b) patching Instructor. v3's layered prompt composition fights Instructor's "we handle the prompt" philosophy.
  - State summarization: No.
- **Per-tenant LoRA**: Neutral. Instructor doesn't care what `model=` you pass, so tenant routing via `model=tenant_lora_name` works. No interference.
- **License**: MIT — no SaaS concerns.
- **Verdict**: Good for greenfield code. **Wrong abstraction level for v4** because v3's accuracy comes from controlling the full prompt envelope (harness → schema → dynamic turn with prefix caching), not from a single `response_model=` decorator. Adopting Instructor means ceding that control. Skip.

### 2. Outlines (dottxt-ai)

- **Repo / metrics**: 13.7k stars, v1.2.12 (2026-03-03), Apache-2.0 ([GitHub](https://github.com/dottxt-ai/outlines))
- **What it does**: Grammar/FSM-constrained generation at the token-sampling level. Guarantees valid JSON/regex/CFG output by masking illegal tokens during decoding.
- **vLLM**: **Yes — but redundantly for our use case.** vLLM ships Outlines as one of three built-in structured-output backends (alongside `xgrammar` and `lm-format-enforcer`). You already get Outlines' constraint logic by passing `extra_body={"guided_decoding_backend": "outlines"}` to the OpenAI client hitting vLLM. ([vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/), [Outlines vLLM integration](https://dottxt-ai.github.io/outlines/latest/features/models/vllm/))
- **Pydantic v2**: Yes, `BaseModel.model_validate_json()` is the canonical path.
- **Does it solve v3's problems?** Solves *one*: malformed JSON disappears because it's physically unrepresentable. Doesn't help with composition, RAG, corrections, or summarization.
- **Per-tenant LoRA**: Neutral when used via vLLM's OpenAI endpoint. Direct `outlines.models.vllm_offline` would fight LoRA routing, but nobody serious uses that path in prod.
- **License**: Apache-2.0 — clean.
- **Verdict**: **Use it via vLLM, don't import the Python package.** The `xgrammar` backend is now vLLM's default and is the recommended choice per Red Hat (up to 100× faster than pre-v1 outlines). ([Red Hat article](https://developers.redhat.com/articles/2025/06/03/structured-outputs-vllm-guiding-ai-responses), [BentoML explainer](https://www.bentoml.com/blog/structured-decoding-in-vllm-a-gentle-introduction))

### 3. DSPy (stanfordnlp)

- **Repo / metrics**: 34k stars, v3.2.0 (2026-04-21), MIT ([GitHub](https://github.com/stanfordnlp/dspy))
- **What it does**: Declarative "signatures" (`input_fields -> output_fields`) compiled via optimizers (MIPRO, GEPA, BootstrapFewShot) into prompts. Its `JSONAdapter` / `ChatAdapter` handle structured output; `ChatAdapter` falls back to `JSONAdapter` on parse failure. ([DSPy adapters](https://dspy.ai/learn/programming/adapters/))
- **vLLM**: Yes via the OpenAI-compatible transport. DSPy's default LM client uses LiteLLM under the hood.
- **Pydantic v2**: Yes, Pydantic models work as output-field types.
- **Does it solve v3's problems?**
  - DSPy is a *prompt compiler*. It's orthogonal to extraction plumbing. The win is if you let it optimize the extraction prompt against a labeled dev set (which you have — the 51-scenario eval). GEPA reportedly lifts structured extraction by ~20 pp in some domains ([Madura write-up](https://kmad.ai/DSPy-Optimization)).
  - Does not solve permissive parse (uses vanilla JSON load + retry).
  - Directly replaces *manual* prompt composition — which is both a win (no more harness hand-tuning) and a loss (you lose harness/RAG/correction branching unless you model them as inputs).
- **Per-tenant LoRA**: Neutral — DSPy calls through LiteLLM with any `model=` string.
- **License**: MIT.
- **Verdict**: **Do not use DSPy as the extraction runtime.** Consider it as an *offline* tool to re-optimize the harness prompt against your eval set (this is exactly the Judge/Refiner loop, but with gradient-free search instead of a refiner LLM). Worth a 1-week spike after v4 is unstuck, not now.

### 4. LangChain `with_structured_output`

- **Repo / metrics**: ~100 k stars (langchain), actively maintained ([new docs URL](https://docs.langchain.com/oss/python/langchain/structured-output))
- **What it does**: `ChatModel.with_structured_output(MySchema)` picks between `ProviderStrategy` (native structured output on OpenAI/Anthropic/xAI/Gemini) and `ToolStrategy` (tool-calling fallback) automatically. Retries on validation error via `handle_errors=True`.
- **vLLM**: Works as an OpenAI-compatible client — same as Instructor. Gotcha: a known bug has vLLM 0.11.x ignoring Pydantic `Field(description=...)` when rendering the JSON schema, so any prompt-level instructions smuggled in field descriptions (a trick v3 may rely on) will be dropped ([vllm#31804](https://github.com/vllm-project/vllm/issues/31804)).
- **Pydantic v2**: Yes.
- **Does it solve v3's problems?** Same tradeoff as Instructor — it owns the message construction. LangChain also drags in a heavy dependency graph (langchain-core, langsmith, etc.) and changes its API roughly every 6 months.
- **Per-tenant LoRA**: Neutral — model name is configurable.
- **License**: MIT.
- **Verdict**: **Skip.** Framework weight vastly exceeds the value of `with_structured_output` alone. If you want just that feature, pull `response_format=` directly from the OpenAI SDK — it's 3 lines and you don't inherit LangChain's upgrade treadmill.

### 5. LlamaIndex Extraction

- **Repo / metrics**: ~40 k stars, active.
- **What it does**: Retrieval-first pipelines with `StructuredLLM` / `PydanticProgram` wrappers.
- **vLLM / Pydantic**: Supported via OpenAI-compatible transport.
- **Does it solve v3's problems?** LlamaIndex's strength is *retrieval over long docs* — a good fit for PDF/document ingestion, not for turn-by-turn conversational extraction into a 30-field CustomerSubmission.
- **Verdict**: **Skip** for the main extractor. Consider only for the PDF/DOCX ingest path under `extraction/document_ingestion.py` if your current implementation is weak there.

### 6. Marvin (PrefectHQ)

- **Repo / metrics**: 6.1 k stars, v3.2.7 (2026-03-04), Apache-2.0 ([GitHub](https://github.com/PrefectHQ/marvin))
- **What it does**: `marvin.extract(text, target=MyModel)`, `marvin.cast`, `marvin.classify`. In v3 it's a thin wrapper over pydantic-ai.
- **vLLM**: Inherits pydantic-ai's OpenAI compatibility — yes.
- **Verdict**: **Skip.** Too minimal — it's an ergonomic one-liner for prototypes, not a pipeline driver. Doesn't solve any v3 problem pydantic-ai doesn't already solve.

### 7. OpenAI Structured Outputs (`response_format`)

- **Native**: OpenAI API, mirrored by vLLM.
- **What it does**: Pass `response_format={"type": "json_schema", "json_schema": {"schema": MySchema.model_json_schema(), "strict": True}}`. Server guarantees schema-valid JSON at decode time.
- **vLLM**: **Supported — same parameter name, same semantics.** vLLM exposes `guided_json` / `response_format` / `structured_outputs` (the name changed across versions — all three map to the same underlying constraint). ([vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/))
- **Pydantic v2**: Via `MySchema.model_json_schema()`.
- **Per-tenant LoRA**: **Combines cleanly per-request.** `model=` selects the LoRA, `response_format=` constrains output. Both are per-request parameters on the vLLM OpenAI endpoint. ([vLLM LoRA blog](https://blog.vllm.ai/2026/02/26/multi-lora.html))
- **Verdict**: **USE THIS.** It is the single biggest leverage move and requires no new dependency. See recommendation in executive summary.

### 8. Anthropic tool use / structured output

- Not applicable to the primary path — you're running Qwen on vLLM. Only relevant for the Judge/Refiner cold path where you already use Claude.

### 9. Mirascope

- **Repo / metrics**: 1.5 k stars, v2.4.0 (2026-03-08), MIT ([GitHub](https://github.com/Mirascope/mirascope))
- **What it does**: Decorator-based LLM calls (`@llm.call(provider=..., response_model=...)`); "anti-framework" positioning.
- **vLLM**: Not explicitly documented; works via OpenAI-compatible transport (same as Instructor).
- **Verdict**: **Skip.** Smaller community than Instructor with no meaningful differentiation for our use case.

### 10. Pydantic-AI (pydantic org)

- **Repo / metrics**: 16.6 k stars, v1.86.0 (2026-04-23 — today), MIT ([GitHub](https://github.com/pydantic/pydantic-ai))
- **What it does**: Agent framework with `output_type=MyModel`, tool calling, streaming, durable execution, Logfire observability.
- **vLLM**: Yes via `OpenAIChatModel` + `OpenAIProvider(base_url=..., api_key=...)`. Tutorials exist for Qwen on vLLM via this path ([AMD ROCm tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/build_airbnb_agent_mcp.html)).
- **Caveats**: Multiple open issues for vLLM compatibility edge cases ([#224](https://github.com/pydantic/pydantic-ai/issues/224), [#728](https://github.com/pydantic/pydantic-ai/issues/728), [#1326](https://github.com/pydantic/pydantic-ai/issues/1326)).
- **Does it solve v3's problems?** It's an agent framework first, extractor second. Same "owns the control flow" concern as Instructor — harder to inject harness/RAG/corrections at the right point.
- **License**: MIT.
- **Verdict**: **Skip for now.** If v4 ever grows into a multi-step agentic flow (tool calling, code execution, retrieval loops) this is the strongest contender among the frameworks. For conversational extraction it's overkill.

### 11. BAML (BoundaryML)

- **Repo / metrics**: 8.1 k stars, v0.221.0 (2026-04-15), Apache-2.0 ([GitHub](https://github.com/BoundaryML/baml))
- **What it does**: `.baml` DSL files → Rust compiler → generated typed clients in Python/TS/Go/Ruby/Rust. Each BAML function definition includes its prompt, schema, and tests co-located.
- **Crown jewel — Schema-Aligned Parsing (SAP)**: Rust parser recovers structured data from malformed LLM output (markdown fences, trailing commas, chain-of-thought preambles, type coercion) in <10 ms with no retry round-trip. Directly replaces v3's 7-strategy parse chain. ([SAP explanation](https://medium.com/@rajkundalia/how-baml-brings-engineering-discipline-to-llm-powered-systems-983c06d31bf8), [BAML vs Pydantic](https://docs.boundaryml.com/guide/comparisons/baml-vs-pydantic))
- **vLLM**: Yes — lists "Anything OpenAI Compatible" incl. vLLM.
- **Pydantic**: Compiler can *generate* Pydantic models from BAML types. You still write Pydantic elsewhere; BAML-emitted classes plug in.
- **License**: Apache-2.0 on the compiler and runtime.
- **Concern**: "Launched late 2025, ecosystem maturing" ([techsy review](https://techsy.io/en/blog/best-llm-structured-output-libraries)) — smaller production base than Instructor. But the parsing engine alone is genuinely novel.
- **Verdict**: **Use the parsing engine as a library, optionally without adopting the DSL.** BAML's standalone SAP parser (exposed in Python bindings) can ingest raw LLM output + a JSON schema and emit a validated dict — the exact contract v3's postprocess chain provides, but in Rust and without writing 7 fallback strategies. This is the only library that offers something v3 genuinely lacks.

### 12. LiteLLM (BerriAI)

- Mentioned for completeness. Proxy layer that unifies 100+ provider APIs. Useful if you want a tenant-aware router in front of vLLM (could map `tenant_id` → LoRA adapter automatically). Not an extraction library. ([repo](https://github.com/BerriAI/litellm))

### 13. XGrammar

- Constrained-decoding engine, now vLLM's default guided-decoding backend. 100× faster than pre-v1 Outlines. You get it for free by using vLLM's `response_format` — no separate install needed. ([techsy ranking](https://techsy.io/en/blog/best-llm-structured-output-libraries) #5)

---

## Decision Table

| If you need… | Use… | Notes |
|---|---|---|
| Guaranteed schema-valid JSON from every LLM call | **vLLM `response_format: json_schema`** (native) | Composes with multi-LoRA per-request |
| Recovery from malformed JSON without an LLM retry | **BAML SAP parser** | Apache-2.0, Rust, <10 ms |
| Offline prompt optimization against labeled eval set | **DSPy** (cold path only) | Run against the 51-scenario eval, export the optimized prompt string, deploy it like any other harness |
| Prompt composition (harness + RAG + schema + dynamic turn with prefix caching) | **v3's custom code** | No library does 4-segment prefix-cacheable composition |
| Correction branching / state summarization | **v3's custom code** | No library models this out of the box |
| Document/PDF retrieval-first extraction (fleet rosters, policy PDFs) | Consider **LlamaIndex** for that path only | Keep it out of the conversational extractor |
| Per-tenant LoRA routing | **vLLM native** (pass `model=tenant-adapter-name`) + LiteLLM if you want a smart router | Orthogonal to all libraries above |
| Quick single-shot extraction in a script | Instructor or pydantic-ai | Both fine; do not adopt as pipeline driver |
| Cross-language clients (if you ever need TS/Go) | **BAML** | Its strongest differentiator vs Instructor |

---

## Code v4 Could Delete

Adopting the two recommended pieces lets you delete:

1. **The 7-strategy JSON parse chain** → BAML SAP (one function call: `baml_py.parse(raw, schema)`).
2. **Most of the postprocess chain that fixes "string where list was expected" class of bugs** → vLLM `response_format: json_schema` prevents them at the source. Field-level coercions (e.g., normalize phone format) stay in Python.
3. **Retry-on-parse-error logic** → SAP makes the first response parseable; retries become a rare correctness concern, not a common one.

What you **cannot** delete:
- Prompt composition layers (harness injection, RAG injection, state summarization, QUESTION-JUST-ASKED annotation).
- Correction branching.
- Deterministic post-extraction rules (`driver_count = len(drivers[])`).
- Per-tenant cache keys and LoRA routing glue.

---

## Why v4 Is Stuck at 71%

Hypothesis based on this research (not a direct v4 code audit): when a library captures the control flow, the layered prompt composition that gets v3 to 99% ends up either (a) crammed into a single monolithic system prompt where the model can't distinguish harness rules from the current turn's context, or (b) partly expressed in library primitives (e.g., Instructor's retry prompt, LangChain's tool-calling metadata) that don't compose with each other. The fix is not a different library; it's to keep composition explicit and delegate only the two narrow concerns where libraries genuinely beat hand-rolled code: **decode-time schema enforcement** and **malformed-JSON recovery**.

---

## Sources

- [Instructor GitHub](https://github.com/567-labs/instructor) · [PyPI](https://pypi.org/project/instructor/) · [integrations](https://python.useinstructor.com/integrations/)
- [Outlines GitHub](https://github.com/dottxt-ai/outlines) · [vLLM integration](https://dottxt-ai.github.io/outlines/latest/features/models/vllm/)
- [DSPy GitHub](https://github.com/stanfordnlp/dspy) · [Adapters docs](https://dspy.ai/learn/programming/adapters/) · [GEPA optimization writeup](https://kmad.ai/DSPy-Optimization)
- [LangChain structured output](https://docs.langchain.com/oss/python/langchain/structured-output)
- [Marvin GitHub](https://github.com/PrefectHQ/marvin)
- [Mirascope GitHub](https://github.com/Mirascope/mirascope)
- [Pydantic-AI GitHub](https://github.com/pydantic/pydantic-ai) · [OpenAI-compat docs](https://pydantic.dev/docs/ai/models/openai/) · issues [#224](https://github.com/pydantic/pydantic-ai/issues/224), [#728](https://github.com/pydantic/pydantic-ai/issues/728), [#1326](https://github.com/pydantic/pydantic-ai/issues/1326)
- [BAML GitHub](https://github.com/BoundaryML/baml) · [BAML vs Pydantic](https://docs.boundaryml.com/guide/comparisons/baml-vs-pydantic) · [Schema-Aligned Parsing explainer](https://medium.com/@rajkundalia/how-baml-brings-engineering-discipline-to-llm-powered-systems-983c06d31bf8)
- [vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/) · [vLLM multi-LoRA blog Feb 2026](https://blog.vllm.ai/2026/02/26/multi-lora.html) · [vLLM#31804 Pydantic description bug](https://github.com/vllm-project/vllm/issues/31804)
- [Red Hat: Structured outputs in vLLM](https://developers.redhat.com/articles/2025/06/03/structured-outputs-vllm-guiding-ai-responses) · [BentoML: Structured Decoding in vLLM](https://www.bentoml.com/blog/structured-decoding-in-vllm-a-gentle-introduction)
- [Techsy: 8 best structured output libraries 2026](https://techsy.io/en/blog/best-llm-structured-output-libraries) · [techsy: vLLM vs SGLang](https://techsy.io/en/blog/vllm-vs-sglang)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
