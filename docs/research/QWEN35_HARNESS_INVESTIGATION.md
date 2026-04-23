# Qwen3.5-9B Harness Collapse Investigation

**Status:** Research findings for v4 rewrite-pivot decision
**Trigger:** Step 25 matrix test — adding a 200-token "light harness" as a SECOND system message drops bulk-scenario F1 from 0.730 → 0.195 on v4, while v3 achieves 99% F1 with 2000 tokens of harness **inside** its single system message.
**Date:** 2026-04-23

---

## TL;DR (up front)

Strong converging evidence points to one root cause and a companion factor:

1. **Primary — Prompt layout, not prompt length.** Qwen's chat template emits TWO distinct `<|im_start|>system ... <|im_end|>` blocks when it sees two system messages. v3 succeeds because it concatenates base + schema + harness into ONE system message (verified in source). Multiple public reports echo the pattern "X didn't work until I put X into the same system prompt."
2. **Companion — Qwen3-family guided-decoding quirks.** With `enable_thinking=False` + xgrammar, Qwen3/3.5 is documented to ignore field descriptions and even emit gibberish. Schema validity is preserved but semantic coverage collapses — exactly matching v4's symptom (structure OK, field recall tanked).

**Recommended Variant E (detailed in §6):** Concatenate the light harness into the same system string as `SYSTEM_V2 + schema`, keep `enable_thinking=False`, hold all other variables (temperature, schema, guided_json backend, tokenizer) constant. If F1 recovers to ≥0.70, the hypothesis is confirmed and v3's prompt geometry is the path forward for v4.

---

## Baseline context: what v3 actually does

`accord_ai/extraction/prompts.py:358-373` (`_build_system_stable`) builds the full system message via a single f-string:

```python
EXTRACTION_SYSTEM_PROMPT_BASE.format(
    harness=harness_section,   # "\n═══ EXTRACTION HARNESS ═══\n{content}"
    schema=_build_full_schema(),
    **date_ctx,
)
```

`build_extraction_messages` (line 404-461) then emits a messages list shaped as:

```
[ {role: system, content: <base + schema + harness + date>},  # ONE block
  {role: user/assistant, content: …},                          # history
  {role: user, content: <state summary + RAG + latest turn>} ]
```

**Key observation:** v3 never emits a separate system message for the harness. The harness is interpolated INTO the same system string. This matches what the Qwen3 prompt-engineering guide recommends: "`system` role clearly" with rules and format constraints inside one system block. ([qwen3lm.com](https://qwen3lm.com/qwen3-prompt-engineering-structured-output/))

v4's current "broken" variants put the harness as a **separate** `{role: system}` message BEFORE `SYSTEM_V2`. That is a fundamentally different shape after the chat template renders.

---

## 1. Qwen chat template: how multiple system messages are handled

**What I found.** I pulled the raw `tokenizer_config.json` for `Qwen/Qwen2.5-7B-Instruct` (same chat-template lineage as 3.5) and read the Jinja template.

Template source: <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/tokenizer_config.json>

Relevant section:
```jinja
{%- if messages[0]['role'] == 'system' %}
    {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user")
         or (message.role == "system" and not loop.first)
         or (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
```

**Behavior for `[{system: A}, {system: B}, {user: ...}]`:**
- Message 0 (system A) → wrapped in `<|im_start|>system\nA<|im_end|>\n` by the preamble.
- Message 1 (system B) → caught by the `(system and not loop.first)` branch → wrapped in its OWN `<|im_start|>system\nB<|im_end|>\n`.
- Result: **two separate system blocks** in the rendered prompt stream, not a merge, not a drop.

Corroborating commentary from Altsoph's chat-template analysis: "Qwen gives full freedom in role alternation … In contrast, SmolLM3 scheme allows arbitrary system messages, but ignores all except the first." ([altsoph.substack.com](https://altsoph.substack.com/p/whats-wrong-with-chat-templates-format))

**Why this matters.** Qwen2.5/3.5 was instruction-tuned with ONE system block as the dominant training distribution. Two back-to-back `<|im_start|>system ... <|im_end|>` blocks are OOD prompt geometry. The model sees a structure it rarely saw during SFT/RLHF, and the training signal for "follow the first block precisely then still respect the second" is weak-to-absent.

**Confidence: HIGH** — template behavior is verified from source; training-distribution claim is inference but well-supported by the SmolLM3 comparison (which explicitly collapses to first-only) and by the fact that Qwen's official examples only ever show one system message.

Sources:
- [Qwen2.5-7B-Instruct tokenizer_config.json (raw)](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/tokenizer_config.json)
- [Qwen-3 Chat Template Deep Dive (HF blog)](https://huggingface.co/blog/qwen-3-chat-template-deep-dive)
- [Altsoph, "What's Wrong with Chat-Templates Format"](https://altsoph.substack.com/p/whats-wrong-with-chat-templates-format)
- [Qwen 3 Prompt Engineering Guide — structured output](https://qwen3lm.com/qwen3-prompt-engineering-structured-output/) (all examples use a single system message)

---

## 2. Qwen3.5 instruction-following degradation with long system prompts

**What I found.** Qwen's own model cards and the 2.5 blog post emphasize "significant improvements in instruction following … generating structured outputs especially JSON" and "more resilient to the diversity of system prompts." No published benchmark distinguishes short vs long system prompts at equal content.

The "lost in the middle" phenomenon (RoPE-induced U-shape attention) is confirmed broadly for transformer LLMs including Qwen derivatives, but is primarily a long-CONTEXT issue (tens of thousands of tokens), not a 2000-token system-prompt issue.

IFScale (arXiv:2507.11538) quantifies instruction-density degradation across 20 frontier models: "even the best frontier models only achieve 68% accuracy at the max density of 500 instructions" with "3 distinct performance degradation patterns correlated with model size" and explicit evidence of "bias towards earlier instructions." Qwen 9B-class models aren't in the list, but the general shape — earlier instructions crowd out later ones — directly predicts what we see: SYSTEM_V2's "extract all fields" instruction arrives AFTER the harness in v4, so it gets deprioritized.

**Confidence: MEDIUM** for length-alone hurting Qwen3.5-9B. The 53.5-point F1 collapse from just 200 added tokens is far too large to be explained by prompt length alone in the 2K-token regime. Length is not the primary factor.

Sources:
- [Qwen2.5 model card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [Qwen2.5 blog](https://qwenlm.github.io/blog/qwen2.5/)
- [IFScale — How Many Instructions Can LLMs Follow at Once? (arXiv 2507.11538)](https://arxiv.org/abs/2507.11538)
- [Morph, "Lost in the Middle LLM"](https://www.morphllm.com/lost-in-the-middle-llm) (RoPE U-shape, not small-prompt relevant)

---

## 3. vLLM guided_json + system prompt interaction

**What I found.** Several vLLM issues document structured-output failures that correlate with prompt content but NOT prompt length per se. The most relevant:

### Issue #23404 — "Qwen3 vLLM Structured Output Ignores Field Descriptions" (Aug 2025)
> "It doesn't follow the description until I put the schema in system prompt"
> ([vllm-project/vllm#23404](https://github.com/vllm-project/vllm/issues/23404))

This is a DIRECT mirror of our finding. A Qwen3 user reports that field-level semantics are ignored when schema is outside the system prompt, and FIXED when folded into the system prompt. This is the exact Variant E hypothesis, confirmed on a sibling model.

### Issue #39348 — Qwen3.5-9B-AWQ endless `!` inside JSON
> "starts generating valid JSON and then abruptly degenerates into an endless stream of `!` characters"
> ([vllm-project/vllm#39348](https://github.com/vllm-project/vllm/issues/39348))

Structure valid, content corrupt. Prompt was 4.9K-char system + 5.5K-char user. Matches the "xgrammar keeps shape valid, semantics collapse" pattern we're seeing.

### Issue #18819 — Broken structured output with Qwen3 + `enable_thinking=False`
> "The output json will most likely not a valid json. It can have an extra '{' or '[' or have '```' in the beginning, and can even be complete gibberish"
> ([vllm-project/vllm#18819](https://github.com/vllm-project/vllm/issues/18819))

Workarounds that worked: enable thinking, append `/no_think` in USER message, or disable the reasoning parser entirely. Highly relevant: the Qwen3/3.5 "no-think" code path is where guided decoding breaks.

### Issue #15236 — Widespread guided-generation failures v0.6.3–0.8.1
Mass breakage of documented examples; xgrammar rejects common schema features; fallback to outlines doesn't happen. ([vllm-project/vllm#15236](https://github.com/vllm-project/vllm/issues/15236))

**Confidence: HIGH** that the v4 collapse is in the same failure family. The symptom profile (valid structure + dropped semantics) matches #23404 almost exactly.

Sources:
- [vLLM #23404 — Qwen3 Structured Output Ignores Field Descriptions](https://github.com/vllm-project/vllm/issues/23404)
- [vLLM #39348 — Qwen3.5-9B-AWQ endless `!` in JSON](https://github.com/vllm-project/vllm/issues/39348)
- [vLLM #18819 — Broken guided decoding with `enable_thinking=False`](https://github.com/vllm-project/vllm/issues/18819)
- [vLLM #15236 — Major issues with guided generation](https://github.com/vllm-project/vllm/issues/15236)

---

## 4. Qwen3.5-9B specific quirks

**What I found.** Directly 9B-specific evidence:

- **AWQ quant of 9B specifically has a decoding pathology** (endless `!`) on ROCm/vLLM 0.19.0 (#39348). If v4 is running a quant variant and v3 is running bf16, that alone could explain some drop — but the 53-point collapse is too large for quant noise alone.
- **No-think path is the weakest.** #18819 shows Qwen3/3.5 guided-decoding bugs trigger specifically when `enable_thinking=False`. Both v3 and v4 use this setting (per v3 CLAUDE.md: "All vLLM calls use `chat_template_kwargs: {enable_thinking: False}`"), so this is a shared risk, but it becomes exploitable when prompt geometry is borderline.
- **9B vs 14B/32B:** No public head-to-head benchmark comparing long-system-prompt instruction following. IFScale does not include 9B Qwen. Anecdotally, vLLM's Qwen3 issue threads show 32B and 30B-A3B hitting the same class of bugs, so it's NOT uniquely a 9B-size problem.

**Confidence: MEDIUM** that 9B is particularly fragile here. More likely the 9B model amplifies a prompt-geometry problem that would be partially masked on 32B's larger instruction-following headroom.

Sources:
- [vLLM #39348 (9B-AWQ specific)](https://github.com/vllm-project/vllm/issues/39348)
- [vLLM #18819 (no-think + guided decoding)](https://github.com/vllm-project/vllm/issues/18819)
- [Qwen3.5 Unsloth guide](https://unsloth.ai/docs/models/qwen3.5)

---

## 5. Production examples: Qwen3.5-9B with long system prompts that succeed

**What I found.** I could not locate a public, production-grade example of Qwen3.5-9B with a ~2000-token system prompt that includes external benchmarks of accuracy. What IS clear from the evidence:

- **Every Qwen3 official structured-output example uses ONE system message.** The Qwen3 structured-output prompt guide and the Alibaba Cloud Model Studio JSON-mode docs consistently place all rules inside a single system block, then user content. ([qwen3lm.com](https://qwen3lm.com/qwen3-prompt-engineering-structured-output/), [Alibaba Model Studio JSON](https://www.alibabacloud.com/help/en/model-studio/qwen-structured-output))
- **v3 itself is the strongest production reference we have.** 99% F1 on 51 scenarios, 97.2% on 5-customer deep test, 2000-token harness inline — proves the single-system-message pattern works.
- **IBM's "JSON prompting for LLMs" guide** and every major structured-prompting blog recommend "define schema, show example, add strict formatting rules, include validation instruction" as a **four-layer SINGLE prompt**, not as separate messages. ([IBM Developer](https://developer.ibm.com/articles/json-prompting-llms/))

**Confidence: HIGH** that single-concat is the dominant production pattern. The absence of multi-system-message success stories is itself signal.

Sources:
- [Qwen3 prompt engineering for structured output](https://qwen3lm.com/qwen3-prompt-engineering-structured-output/)
- [Alibaba Model Studio — Qwen structured output](https://www.alibabacloud.com/help/en/model-studio/qwen-structured-output)
- [IBM Developer — JSON prompting for LLMs](https://developer.ibm.com/articles/json-prompting-llms/)
- v3 prompts.py `_build_system_stable()` at `/home/inevoai/Development/Accord-Model-Building/Custom_model_fa_pf/accord_ai_v3/accord_ai/extraction/prompts.py:358`

---

## 6. RECOMMENDED EXPERIMENTS

The goal is to localize the failure to ONE variable at a time. Budget-ordered (cheapest first):

### Experiment E1 — **Single concat'd system message (primary Variant E)**
**Hypothesis:** Prompt geometry, not prompt content. Folding harness into the same system string fixes it.
**Setup:**
- Take v4's current `SYSTEM_V2 + schema` system message.
- Append the 200-token light harness to that exact string (just string concat, same block).
- Keep everything else identical: same harness text, same user message shape, same xgrammar schema, same sampling, `enable_thinking=False`.
- Run the same bulk-scenario eval that produced the 0.730 → 0.195 drop.
**Decision rule:**
- F1 ≥ 0.70 → confirmed. Adopt single-concat layout for v4. This is the v3-proven geometry.
- F1 in [0.40, 0.70) → layout helps but there's a second factor. Proceed to E2.
- F1 < 0.40 → layout is not the root cause. Harness content itself is harmful on v4. Investigate content differences vs v3's harness (wording, negations, rule style).

### Experiment E2 — **Harness in USER tail, not system**
**Hypothesis:** If E1 is partial, the issue may be SYSTEM_V2's instruction getting crowded out by ANY extra system content. Move the harness to the very end of the final user message, right before the content to extract.
**Setup:**
- System message = base SYSTEM_V2 + schema (no harness), identical to baseline.
- Final user message: `<state><RAG><harness as "Extraction tips:" block><latest user content>`.
- This mirrors IFScale's "bias towards earlier instructions" finding inverted: the harness at the END of the prompt is freshest/most-attended for the output.
**Decision rule:** F1 recovery ≥ 0.65 → harness belongs at the tail on Qwen3.5-9B. Compare to E1 for best-of.

### Experiment E3 — **xgrammar vs outlines backend swap**
**Hypothesis:** v4's field-dropping is xgrammar-specific (per vLLM #23404, #18819, #15236 patterns).
**Setup:**
- Keep v4's TWO-system-message layout (the known-bad case).
- Switch vLLM `guided_decoding_backend` from `xgrammar` to `outlines` (or `guidance`).
- Same schema, same prompts, same model.
**Decision rule:**
- F1 recovery with outlines → xgrammar + long-prompt is the culprit. File upstream or pin to outlines.
- No recovery → rules out backend. E1's prompt-geometry hypothesis stands alone.

This experiment is cheap (one flag change) and its result orthogonally disambiguates layout from backend. Run it in parallel with E1 if machine time allows.

### Single most important test to run: **E1**

E1 directly tests the hypothesis best-supported by the evidence: v3's success is NOT because it's magical, it's because it concatenates into one system block, which is what Qwen was trained to follow. The bug is v4's multi-system-message layout, not the harness content. E1 either confirms this for one flag-flip's worth of effort, or rules it out cleanly.

---

## Appendix — key quotes for the decision memo

> "It doesn't follow the description until I put the schema in system prompt"
> — vLLM #23404, Qwen3 user, Aug 2025

> "Qwen gives full freedom in role alternation … In contrast, SmolLM3 scheme allows arbitrary system messages, but ignores all except the first."
> — Altsoph, chat-template analysis

> Qwen2.5 chat template — the second-system-message branch:
> `{%- if (message.role == "system" and not loop.first) %} {{- '<|im_start|>system\n' + message.content + '<|im_end|>\n' }}`
> → two separate system blocks, not merged, not dropped.

> "Even the best frontier models only achieve 68% accuracy at the max density of 500 instructions … bias towards earlier instructions."
> — IFScale, arXiv:2507.11538
