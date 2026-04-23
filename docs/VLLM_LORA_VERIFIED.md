# vLLM LoRA API verification — DOCS-LEVEL VERIFIED

**Date:** 2026-04-20
**Status:** ✅ DOCS-LEVEL VERIFIED. Empirical verification deferred to Phase 4 step 4.6 (first synthetic broker training).
**Fallback if the LoRA API breaks at that point:** per-broker vLLM process model (one serve process per adapter, swap via DNS/reverse-proxy rather than in-process hot-swap). Heavier on memory, simpler to reason about.

## Why this isn't a tri-state yet

The 1B instructions specified "a second terminal, on a disposable box (not production)". I don't have access to one — this machine has a single RTX 3090, and the vLLM instance currently running on `:8000` is the one every test + eval in the v4 loop is talking to. Restarting it with `--enable-lora` flags for the verification would take:

- vLLM cold-start: ~60-120s
- Any in-flight eval: interrupted (currently none)
- Return to the current serving config (for v4 testing continuity): another ~60-120s
- Plus a base-model swap if the test LoRA's base differs from `Qwen/Qwen3.5-9B`

Total: ~15-30 min window where v4 testing is unavailable. Not destructive, but crosses the "affects shared systems" line in the sandbox's action guide. Flagging for explicit authorization before I do it.

## What I verified without restarting vLLM

### 1. vLLM 0.18.1 supports the LoRA API surface Phase 4 needs

- `vllm serve ... --enable-lora --max-loras N --max-lora-rank R` startup flags: **confirmed in vllm 0.18.1** (per docs + `vllm --help`)
- Dynamic adapter load/unload HTTP endpoints: **`POST /v1/load_lora_adapter` and `POST /v1/unload_lora_adapter`** are the documented endpoints in vllm ≥ 0.5. Still present in 0.18.1.
- OpenAI-compat routing via `served-model-name` alias: confirmed working today — our current vLLM is serving as `Qwen/Qwen3.5-9B` and v3 calls `--served-model-name insurance-agent` to route through the same alias mechanism.

### 2. Adapter assets inventory (for the future real test)

On this machine at `/home/inevoai/Development/Accord-Model-Building/finetune/output/`:

| Adapter path | Base model | Rank/Alpha |
|---|---|---|
| `final/` | unsloth/qwen3-vl-8b-instruct-bnb-4bit | 32 / 64 |
| `agent/final/` | unsloth/qwen3-8b-bnb-4bit | 32 / 64 |
| `agent/phase_0_agent_foundation/final/` | unsloth/qwen3-8b-bnb-4bit | 32 / 64 |
| `agent/phase_1_agent_complex/final/` | unsloth/qwen3-8b-bnb-4bit | 32 / 64 |
| `agent/phase_2_agent_error_specific/final/` | unsloth/qwen3-8b-bnb-4bit | 32 / 64 |

**Note the base-model mismatch.** None of these target `Qwen/Qwen3.5-9B` (the one we serve). Four target `unsloth/qwen3-8b-bnb-4bit` (Qwen3-8B, 4-bit). One targets `Qwen3-VL-8B` (vision-language variant).

Implication for Phase 4: if the plan is to load per-broker adapters against the currently-served `Qwen/Qwen3.5-9B`, we need new adapters trained against **that** base. The existing adapters are for an earlier training run targeting a different base model.

## What the empirical test would add

- Confirm `load_lora_adapter` returns 200 on a fresh vLLM and the adapter appears in the next `/v1/models` list (tests Phase 4's model-registry assumption)
- Confirm `unload_lora_adapter` returns 200 AND GPU memory drops (tests Phase 4's swap-without-restart assumption)
- Measure the load time (seconds or minutes matters for per-broker hot-swap UX)
- Verify a request with `"model": "<adapter_name>"` routes through the adapter (tests Phase 4's multi-tenant routing)

## tl;dr for status reply

The LoRA APIs exist in vllm 0.18.1 per docs; the current v4 vLLM doesn't have `--enable-lora` on and restarting it to verify would take v4 testing offline for 15-30 min. Adapters on disk are for Qwen3-8B or Qwen3-VL, **not for the Qwen/Qwen3.5-9B we serve** — so Phase 4's "per-broker adapter" plan needs new adapters trained against the current base model regardless of LoRA-API availability. Awaiting explicit authorization to restart vLLM for empirical test; if the plan pivots to "keep base model, train new adapters," that verification can fold into the first adapter-training run.
