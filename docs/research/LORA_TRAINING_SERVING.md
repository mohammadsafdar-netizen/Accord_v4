# Phase L — Per-Tenant LoRA Training + Multi-Adapter Serving

**Date:** 2026-04-23
**Base model:** `Qwen/Qwen3.5-9B` (vLLM)
**Hardware:** RTX 3090 (dev) → A100 40/80GB or L40S (prod)
**Scale:** 50–500 brokers × 1 adapter each, retrained weekly
**Existing code:** `finetune/train.py` (Unsloth SFT), `finetune/train_dpo.py` (TRL DPO)

---

## Executive summary

**Adopt this stack for Phase L:**

| Layer | Choice | Rationale |
|---|---|---|
| Training framework | **Unsloth** (primary) + **TRL** (DPO/ORPO trainers) | Already in-tree; 2–5× faster than TRL baseline on single-GPU, 24% faster than Torchtune on RTX 4090, Liger-Kernel-equivalent throughput with lower memory. |
| Algorithm | **SFT first → ORPO for corrections** (fall back to DPO if ORPO unstable) | ORPO fuses SFT + preference in one loss — no reference model, ~50% memory + time vs SFT+DPO. DPO stays as the well-trodden fallback; KTO is for unpaired binary thumbs-up/down data which we don't have. |
| Adapter config | LoRA r=16, α=32, targets = all linear, dropout=0.05 | Matches existing `train.py` defaults; keeps per-adapter size ≈ 80–120 MB for Qwen 9B; vLLM `--max-lora-rank 16` is the cheapest. |
| Serving (phase-L v1) | **vLLM ≥ 0.15** with `--enable-lora --max-loras 8 --max-lora-rank 16 --max-cpu-loras 64` + `VLLM_ALLOW_RUNTIME_LORA_UPDATING=true` behind an internal control-plane; **disable prefix caching while LoRA is live** on our serving engine to avoid the known cross-adapter cache bug. |
| Serving (phase-L v2, 50+ brokers) | Switch to **LoRAX** (Predibase, Apache 2.0) or **SGLang ≥ 0.5.9** — both batch heterogeneous adapters in one batch with negligible per-adapter overhead. |
| Adapter registry | **S3 (or MinIO) + SQLite manifest** keyed by `(tenant_id, version, base_model_sha, algo)` — HF Hub only if we want the UI for free. |
| Canary | Shadow-route 10% of tenant traffic to the new adapter, gate on `judge_score_delta ≥ 0` and `hallucination_rate ≤ baseline` for 24h before flipping the pointer. |
| Lifecycle target | Correction → deployed adapter in **< 24h** (weekly is fine for v1; build the plumbing to support daily). |

**Expected cost envelope:** ≈ 0.5–1.5 A100-hours per broker per week (30–50 MB of training data, 1–3 epochs, r=16); ≈ 100 MB storage per adapter; 50 brokers @ weekly ≈ 25–75 A100-hours/week ≈ **$50–$150/wk cloud** or ~4–10 hours/week saturating a local RTX 3090 fleet.

---

## 1. vLLM multi-LoRA serving (2026 state)

**Flag set we should run on A100:**
```
vllm serve Qwen/Qwen3.5-9B \
  --served-model-name insurance-agent \
  --enable-lora \
  --max-loras 8 \
  --max-lora-rank 16 \
  --max-cpu-loras 64 \
  --kv-cache-dtype fp8 --enforce-eager
# and for dynamic registry behind a trusted control-plane:
VLLM_ALLOW_RUNTIME_LORA_UPDATING=true
```

- `max_loras` = max concurrent adapters in one batch (GPU-resident). 8 is enough for our traffic pattern; each slot costs ≈ (rank × model_hidden × 2 × 2bytes) ≈ tens of MB.
- `max_cpu_loras` = CPU-pinned pool swappable into GPU. Set to 64 so we don't hit disk on tenant switch. ([vLLM LoRA docs](https://docs.vllm.ai/en/latest/features/lora/))
- `--max-lora-rank 16` is the cheapest tier. Going to 64 wastes memory for all adapters even when most are r=16. ([vLLM LoRA config](https://docs.vllm.ai/en/latest/features/lora/))
- On **RTX 3090** we'll keep `--max-loras 2 --max-cpu-loras 16` because VRAM is the constraint once the 9B base is loaded (~18 GB weights + KV cache).

**Known production gotchas (must internalize):**

1. **Dynamic load/unload is explicitly NOT recommended for production multi-replica deployments** — `VLLM_ALLOW_RUNTIME_LORA_UPDATING=true` exposes `POST /v1/load_lora_adapter` and `/v1/unload_lora_adapter`, but vLLM docs flag that it does not coordinate across replicas and "should ONLY be used for local development." Our mitigation: single-replica per region in v1, or put LoRAX in front of vLLM in v2. ([vLLM forum discussion](https://discuss.vllm.ai/t/lora-adapter-enabling-with-vllm-is-not-working/468), [RFC #12174](https://github.com/vllm-project/vllm/issues/12174))
2. **Prefix caching + multi-LoRA bug still open.** Issue [#30931](https://github.com/vllm-project/vllm/issues/30931) (2026): "different lora_int_id values incorrectly share KV cache blocks → outputs generated with wrong LoRA weights." Academic benchmark ([arxiv 2505.03756](https://arxiv.org/html/2505.03756v1)) measures **48.1% invalid KV caches on average** across multi-LoRA workloads. **Decision: disable `--enable-prefix-caching` in the LoRA serving instance** until the fix lands. Lose ~30% TPS on repeated prefixes but gain correctness.
3. **Qwen3.5 LoRA target-module edge cases.** [#38085](https://github.com/vllm-project/vllm/issues/38085) and [#36372](https://github.com/vllm-project/vllm/issues/36372) document mismatched module names between Unsloth-trained adapters and vLLM's Qwen3.5 model def. **Mitigation:** pin adapters to the same Unsloth base-model repo we use at serving time, and run a smoke-test (load + single inference) in CI before publishing.
4. **Up to 50% throughput drop with LoRA loaded** vs plain base on A100 40GB ([issue #10062](https://github.com/vllm-project/vllm/issues/10062)). Budget capacity assuming 2× headroom.
5. `vLLM ≥ 0.15` is required for the 2026 multi-LoRA optimizations (454% OTPS gain, 87% lower TTFT vs 0.11.1) per the [vLLM + SageMaker blog](https://blog.vllm.ai/2026/02/26/multi-lora.html). Our current pin in `v3` needs to roll forward.

**Alternatives — when to switch:**

| Engine | Verdict for Phase L |
|---|---|
| **SGLang ≥ 0.5.9** | Multi-LoRA is first-class; v0.5.9 adds LoRA weight load overlap → 78% lower TTFT. Switch worth considering at 50+ adapters. ([particula](https://particula.tech/blog/sglang-vs-vllm-inference-engine-comparison)) |
| **LoRAX** (Predibase OSS) | Purpose-built for the 1000s-of-adapters case; heterogeneous continuous batching; S3/HF adapter pull; Prometheus/OTel out of the box. **Top recommendation if vLLM gets painful.** ([LoRAX GitHub](https://github.com/predibase/lorax), [AWS case study](https://aws.amazon.com/blogs/machine-learning/host-concurrent-llms-with-lorax/)) |
| **TGI** | LoRA support exists but has lagged vLLM; no reason to switch. |
| **LMDeploy / MLC-LLM** | Strong on throughput but multi-LoRA is less polished; not worth retooling for. |
| **TensorRT-LLM** | Peak throughput champion but Nvidia-only + build friction; don't touch unless we hit real capacity pain. |

---

## 2. Training framework comparison

| Framework | Speed (RTX 4090, 7B LoRA) | Memory | Models | DX | Verdict |
|---|---|---|---|---|---|
| **Unsloth** | 2–5× TRL; 24% > Torchtune | 80% less VRAM | 150+ | Python API, script-shaped | **Keep** — already in-tree. ([Spheron comparison](https://www.spheron.network/blog/axolotl-vs-unsloth-vs-torchtune/)) |
| **Axolotl** | ~TRL baseline (slower than Unsloth) | Standard | 100+ | YAML config, battle-tested | Use if we need multi-node or exotic schedules. Not needed for 9B weekly retrains. |
| **LLaMA-Factory** | Slower (GUI overhead) | Standard | 100+ | Web UI + CLI | Useful for broker-facing self-serve, not for automated pipelines. ([VoltAgent](https://voltagent.dev/blog/llama-factory/)) |
| **Liger Kernel + TRL** | +20% throughput, −60% mem vs vanilla TRL | −60% | any HF | Drop-in | **Use as a fallback** — if Unsloth regresses on a Qwen3.5 release, Liger + TRL DPOTrainer gives us ~equivalent throughput. Benchmark ([arxiv 2410.10989](https://arxiv.org/pdf/2410.10989)) shows Unsloth and Liger tie on DPO quality. |
| **DeepSpeed ZeRO** | — | — | — | Heavy | **Overkill** for 9B on a 3090/A100 with LoRA. Only needed for full FT of 70B+. |
| **TRL native** | Baseline | Standard | any HF | Good abstractions | Use for the DPO/ORPO trainer classes **on top of** an Unsloth-prepped model. |

**PEFT 2025–2026 features worth using:** DoRA (Weight-Decomposed LoRA), rsLoRA (scaled LoRA), VeRA (shared random projections). DoRA gives +0.5–1.5 pp on preference benchmarks at ~same memory — enable via Unsloth's `use_dora=True`. For 500-adapter scale, VeRA can cut per-tenant storage 5–10× but needs validation on our extraction task first.

---

## 3. DPO vs SFT vs IPO vs KTO vs ORPO for corrections

Our signal is **"broker corrected field X from value A to value B"** — a structured paired preference (A = rejected, B = chosen), anchored on the same session context. Some sessions yield only positive labels (broker said "looks good, finalize") which is a KTO-shaped signal.

| Algorithm | Fit for our signal | Recommendation |
|---|---|---|
| **SFT** | Direct, stable, ignores "what was wrong." | Use for the **clean session baseline** (sessions the broker finalized without edits). |
| **DPO** | Classical preference pair → model. Proven, stable, needs reference model (2× memory). | **Fallback** if ORPO is noisy. |
| **IPO** | DPO variant that fixes overfitting on small pref sets. | Relevant if we have <500 pairs/broker — worth A/B-ing. |
| **KTO** | Unpaired ±1 labels. ([aman.ai](https://aman.ai/primers/ai/preference-optimization/)) | Use for "broker approved finalize" (positive-only) batches. Mix with ORPO on paired data. |
| **ORPO** | **Combines SFT + preference in one loss, no reference model** → ~50% time+memory vs SFT+DPO. ([arxiv 2403.07691](https://arxiv.org/abs/2403.07691), [HF blog](https://huggingface.co/blog/mlabonne/orpo-llama-3)) | **Primary choice.** Outperforms SFT+DPO on Phi-2/Llama3 at smaller data scales. Built into TRL as `ORPOTrainer`. |
| **SimPO** | Reference-free, up to +7.5 pp on Arena-Hard vs DPO. ([princeton-nlp/SimPO](https://github.com/princeton-nlp/SimPO)) | Strong for broad alignment, less clearly better for narrow field-extraction corrections. Keep in reserve. |

**Data-size rule of thumb (from the DPO literature + our use case):**
- Under 100 pairs/broker: don't train — keep appending to the pool and retrain when threshold crossed, else you'll overfit and regress general capability.
- 100–500 pairs: ORPO with early stopping, 1 epoch, β=0.1.
- 500+ pairs: ORPO 2–3 epochs; add SFT on neutral data to stop mode-collapse.
- Always keep ≥5% of pairs as a held-out per-broker eval set.

**Mixing SFT + preference:** ORPO handles this natively (that's the whole point). If falling back to DPO, run **SFT on clean sessions for 1 epoch → DPO on corrections for 1 epoch with β=0.1** ([OpenAI cookbook](https://cookbook.openai.com/examples/fine_tuning_direct_preference_optimization_guide) confirms SFT-before-DPO measurably improves domain-shifted data).

---

## 4. Adapter lifecycle management

**Storage layout (recommended):**
```
s3://accord-adapters/
  {tenant_id}/
    manifest.json                        # pointer to active version
    v{N}_{sha}/
      adapter_config.json
      adapter_model.safetensors          # 80-120 MB
      training_card.json                 # data hash, algo, hyperparams, eval scores
      eval_report.json                   # judge score, hallucination rate, field accuracy
```

`manifest.json` holds `{"active": "v7_abc123", "candidate": "v8_def456", "rollback": "v6_xyz789"}` so promotion and rollback are atomic S3 object writes.

**Registry options (pick one):**

| Option | Fit |
|---|---|
| **HuggingFace Hub private repos per tenant** | Free, has UI, handles hashing + versioning. **Good for v1.** API-key-gate per tenant. |
| **S3 + SQLite manifest** | Full control, no external dependency, same data plane as everything else we run. **Best for v2.** |
| **MLflow Model Registry** | Nice "Production/Staging" semantics but overkill for flat adapter artifacts. Skip. |
| **KServe InferenceService + ModelMesh** | Strong if we're on K8s. Overkill for a 3-GPU deployment. |
| **BentoML** | Good if we want to package the base model + adapters + API into a single deploy unit. Add later if we sell on-prem. |

**Versioning + canary workflow:**
1. Trainer writes `v{N}_{sha}/` + `training_card.json` + `eval_report.json` to S3.
2. Evaluator runs judge + held-out eval; if `judge_score < 7.5` or `hallucination_rate > 0.01`, mark `quarantined` and stop.
3. Control-plane shadow-routes 10% of that tenant's traffic to the new adapter for 24h (send same request to old + new, compare outputs, judge).
4. If shadow metrics ≥ baseline, flip `manifest.active` → new version.
5. Keep last 3 versions warm for rollback (< 30s to flip `manifest.active` back).

**Auto-deploy gates (mandatory):**
- Judge score ≥ 7.5 on held-out set (same threshold as `harness/judge.py`).
- Hallucination rate ≤ production baseline.
- Zero regressions on the 51-scenario L3 eval (99.0% current).
- Training data passes PII redaction lint (must match our `PIIRedactionFilter` output exactly).

**Existing tooling for auto-pipelines:** Predibase (commercial), Fireworks.ai (commercial), ZenML (OSS orchestrator), Argo Workflows on K8s. For our scale, a **cron + a single Python orchestrator** that calls `unsloth → trl → s3-upload → evaluator → manifest-flip` is enough; don't pick up Argo before we have 20+ brokers in the pipeline.

---

## 5. Continuous improvement patterns — production case studies

**Convirza (Llama-3-8b on Predibase/LoRAX):** 60 concurrent broker-like adapters, one GPU, p50 < 2s. 10× cost reduction vs GPT-4, +8% F1 vs baseline Longformer, +80% throughput. Adapter lifecycle: corrections → nightly retrain → shadow eval → auto-promote. ([Predibase case study](https://predibase.com/blog/lora-exchange-lorax-serve-100s-of-fine-tuned-llms-for-the-cost-of-one))

**Checkr (Llama-3-8b-instruct, background-check classification):** 90% accuracy on hard cases, 5× cost cut, 30× speedup vs GPT-4. Same pattern: per-customer adapter, correction loop, nightly retrain.

**AWS SageMaker multi-tenant LoRA reference architecture:** atomic add/delete/update of adapters across endpoint instances without redeploy; adapters loaded from GPU/CPU/disk in milliseconds. Uses S3 as the adapter registry, inference-component APIs for lifecycle. ([AWS ML blog](https://aws.amazon.com/blogs/machine-learning/easily-deploy-and-manage-hundreds-of-lora-adapters-with-sagemaker-efficient-multi-adapter-inference/))

**Realistic latency for "correction to deployed adapter":**
- Weekly batch (our v1 target): acceptable, matches broker cadence.
- Daily: feasible with a nightly retrain cron. Plumbing-wise: same.
- <1h ("every correction becomes training"): requires streaming DPO / online fine-tuning — not production-mature. Skip.

---

## Recommended training pipeline (pseudocode)

```python
# nightly_train.py — one broker at a time
for tenant in brokers_with_fresh_corrections():
    clean = load_clean_sessions(tenant)          # SFT data
    corrections = load_correction_pairs(tenant)  # ORPO preference data
    if len(corrections) < 100: continue          # defer

    model = unsloth.FastLanguageModel.from_pretrained("Qwen/Qwen3.5-9B")
    model = unsloth.apply_lora(model,
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules="all-linear", use_dora=True)

    trainer = trl.ORPOTrainer(
        model=model,
        args=ORPOConfig(
            beta=0.1, num_train_epochs=2,
            per_device_train_batch_size=1, gradient_accumulation_steps=8,
            learning_rate=5e-5, warmup_ratio=0.05,
        ),
        train_dataset=merge_sft_and_pairs(clean, corrections),
    )
    trainer.train()

    adapter_dir = f"s3://accord-adapters/{tenant}/v{next_version}_{sha}/"
    model.save_adapter(adapter_dir)

    report = run_evaluator(adapter_dir, held_out=corrections[-5%:])
    if report["judge_score"] >= 7.5 and report["hallucination_rate"] <= baseline:
        shadow_route(tenant, adapter_dir, pct=0.1, duration_h=24)
    else:
        mark_quarantined(adapter_dir, report)
```

---

## Adapter lifecycle workflow

```
 [broker corrections]          [clean finalized sessions]
        │                               │
        └──────────► pair builder ◄─────┘
                         │
                   [ORPO training]   (Unsloth + TRL, A100 or 3090)
                         │
               adapter_v{N}.safetensors  +  training_card.json
                         │
                 [evaluator: judge + held-out]
                         │
                ┌────────┴────────┐
             reject           accept → [shadow route 10% for 24h]
                │                         │
              quarantine              metrics OK?
                                     ┌────┴────┐
                                  yes          no
                                   │            │
                             manifest.flip    rollback keep old
                                   │
                             [vLLM / LoRAX pulls new adapter]
                                   │
                               serving live
```

---

## Cost + hardware sizing

**Per-adapter training (Qwen 9B, r=16, DoRA, 500 ORPO pairs, 2 epochs):**
- RTX 3090 (24 GB): 2–4 hours, ≈ 18–22 GB VRAM peak (Unsloth 4-bit load + LoRA).
- A100 40 GB: 30–60 min.
- A100 80 GB: 20–40 min.
- H100 80 GB: 12–25 min.

Sources: RunPod LoRA budget guide confirms 24 GB handles 7–9B at r=16 with QLoRA comfortably; A100 40 GB halves wall time. ([RunPod guide](https://www.runpod.io/articles/guides/how-to-fine-tune-large-language-models-on-a-budget))

**Per-adapter storage:** 80–120 MB (safetensors, r=16, all-linear targets). 500 brokers = ~50 GB — trivial.

**Serving capacity (A100 40GB, vLLM ≥ 0.15, base+LoRA):**
- ~1,500–2,000 tok/s aggregate with `--max-loras 8`, vs ~3,000 base-only (≈ 50% degradation ceiling; see [issue #10062](https://github.com/vllm-project/vllm/issues/10062)).
- Matches our current v3 budget (2,728 tok/s at 10 concurrent on 3090 base-only → ~1,300 with LoRA). Plan production on A100 80 GB to recover headroom.

**Weekly cost for 50 brokers:**
- 50 × 1 hour × A100 40 GB RunPod spot (~$1.50/h) = **~$75/week**.
- Storage (S3 std): 50 × 100 MB × 4 versions = 20 GB ≈ $0.50/month.
- Serving: unchanged from current v3 serving bill (+ optional A100 80 GB for headroom).

**Weekly cost for 500 brokers:**
- 500 × 0.5 hour × A100 80 GB (~$2/h) = **~$500/week** for training.
- Storage: 200 GB ≈ $5/month.
- Serving: need 2× A100 80 GB minimum for headroom and redundancy; consider LoRAX orchestration.

---

## Open questions / follow-ups

1. **Base-model alignment.** `VLLM_LORA_VERIFIED.md` flags that on-disk adapters were trained against `qwen3-8b-bnb-4bit` and `qwen3-vl-8b`, not `Qwen/Qwen3.5-9B`. First Phase-L training run must target the served base exactly — expect to discard existing adapters.
2. **Empirical verification** of `load_lora_adapter` / `unload_lora_adapter` on our serving instance (time-to-load, GPU memory swap). Deferred in the VLLM_LORA_VERIFIED doc; schedule for the first Phase-L dry run.
3. **Privacy:** broker corrections likely contain PII → training data must run through `PIIRedactionFilter` before hitting S3; never send raw corrections to Predibase/Fireworks managed services without explicit broker consent.
4. **Evaluator reuse:** our existing `harness/judge.py` is the obvious auto-gate — just score adapter outputs on a fixed per-tenant eval set and compare to baseline.

---

## Sources

- vLLM LoRA docs — https://docs.vllm.ai/en/latest/features/lora/
- vLLM multi-LoRA + SageMaker blog (Feb 2026) — https://blog.vllm.ai/2026/02/26/multi-lora.html
- vLLM issue #30931 — prefix cache corruption with LoRA — https://github.com/vllm-project/vllm/issues/30931
- vLLM issue #5475 — prefix caching + multi-LoRA — https://github.com/vllm-project/vllm/issues/5475
- vLLM issue #10062 — LoRA throughput drop on A100 — https://github.com/vllm-project/vllm/issues/10062
- vLLM issue #38085 — Qwen3.5 LoRA target-module mismatch — https://github.com/vllm-project/vllm/issues/38085
- vLLM RFC #12174 — distribute LoRA across replicas — https://github.com/vllm-project/vllm/issues/12174
- arxiv 2505.03756 — multi-LoRA KV cache management — https://arxiv.org/html/2505.03756v1
- SGLang vs vLLM (Particula, 2026) — https://particula.tech/blog/sglang-vs-vllm-inference-engine-comparison
- Spheron fine-tuning framework comparison (2026) — https://www.spheron.network/blog/axolotl-vs-unsloth-vs-torchtune/
- DEV Community "EVAL #003" (2026) — https://dev.to/ultraduneai/eval-003-fine-tuning-in-2026-axolotl-vs-unsloth-vs-trl-vs-llama-factory-2ohg
- Liger Kernel paper — https://arxiv.org/pdf/2410.10989
- ORPO paper — https://arxiv.org/abs/2403.07691
- ORPO on Llama-3 (HF blog) — https://huggingface.co/blog/mlabonne/orpo-llama-3
- SimPO (NeurIPS 2024) — https://github.com/princeton-nlp/SimPO
- DPO post-training stack (Fahey, 2025) — https://medium.com/@fahey_james/dpo-isnt-enough-the-modern-post-training-stack-simpo-orpo-kto-and-beyond-d82e52a1ee6c
- Preference optimization primer — https://aman.ai/primers/ai/preference-optimization/
- OpenAI cookbook DPO guide — https://cookbook.openai.com/examples/fine_tuning_direct_preference_optimization_guide
- Predibase LoRAX (GitHub) — https://github.com/predibase/lorax
- Predibase LoRA Exchange case studies — https://predibase.com/blog/lora-exchange-lorax-serve-100s-of-fine-tuned-llms-for-the-cost-of-one
- Predibase vs Fireworks vs vLLM benchmark — https://predibase.com/blog/llm-inference-benchmarks-predibase-fireworks-vllm
- AWS multi-tenant LoRA on SageMaker — https://aws.amazon.com/blogs/machine-learning/easily-deploy-and-manage-hundreds-of-lora-adapters-with-sagemaker-efficient-multi-adapter-inference/
- AWS LoRAX on SageMaker — https://aws.amazon.com/blogs/machine-learning/host-concurrent-llms-with-lorax/
- RunPod LoRA budget guide — https://www.runpod.io/articles/guides/how-to-fine-tune-large-language-models-on-a-budget
- Unsloth 3× faster training — https://docs.unsloth.ai/new/3x-faster-training-packing
- ZenML LLMOps case-study collection — https://www.zenml.io/blog/llmops-in-production-457-case-studies-of-what-actually-works
