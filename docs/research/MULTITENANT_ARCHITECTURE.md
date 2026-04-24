# Multi-Tenant Architecture Research — accord_v4

Scope: architectural patterns for per-tenant customization in production LLM systems, with
specific recommendations for accord_v4 (50–500 brokers, 1K–5K intakes/broker/month, Qwen3.5-9B
base + per-tenant LoRA adapters via vLLM).

---

## Executive Summary

**Current design verdict: structurally correct, tactically incomplete.** The decisions already
made in v3 — per-tenant ChromaDB directories, per-tenant SQLite sessions, per-tenant harness
overlay files — match the mainstream 2025/2026 pattern (namespaced RAG, tenant-scoped state,
layered prompts). The gaps are in four specific areas:

1. **LoRA delivery** — static `--lora-modules` at vLLM boot does not scale to 50–500
   adapters with churn. You need the **LoRA Resolver Plugin** (S3 or FS backend) or
   a **LoRAX** sidecar; both exist and are production-proven ([Convirza/Predibase case
   study](https://predibase.com/blog/convirza-case-study): 60+ adapters, sub-2s latency,
   10× cost reduction vs OpenAI).
2. **Prefix-cache salting** — your harness+schema preamble is currently shared across
   tenants. vLLM's prefix cache creates a [cross-tenant timing side channel](https://www.ndss-symposium.org/wp-content/uploads/2025-1772-paper.pdf)
   (NDSS 2025) unless you set `cache_salt` per-tenant ([vLLM RFC #16016](https://github.com/vllm-project/vllm/issues/16016)).
3. **Prompt management** — ad-hoc `harness/{tenant}.md` files work at 5 tenants; at 500
   you need versions, labels (staging/prod), rollback, and audit. **Langfuse** (open source,
   self-hostable, has explicit multi-tenant label pattern) is the low-effort win.
4. **GDPR erasure for adapters** — no current plan. An adapter fine-tuned on a tenant's
   data encodes their data in its weights. Deletion = delete adapter artifact + retain
   anonymized audit trail. This must be designed in, not bolted on.

Below, per-question findings, concrete patterns to adopt and avoid, and a final stack diagram.

---

## 1. How Production Multi-Tenant LLM Apps Structure Customization

The industry has converged on a **three-layer customization stack** — the same pattern
whether the platform is Replit, Cursor, LiteLLM, or AWS Bedrock:

| Layer | What varies per tenant | What's shared |
|---|---|---|
| **Retrieval** | Vector namespace / collection | Base embeddings, reranker |
| **Prompt** | System prompt / instructions overlay | Base chat template, tool definitions |
| **Weights** | LoRA adapter (optional) | Base model, KV cache fabric |

**Concrete examples found:**

- **LiteLLM multi-tenant proxy** ([docs](https://docs.litellm.ai/docs/proxy/multi_tenant_architecture))
  uses per-tenant API keys with metadata, per-key rate limits, and per-key model/prompt
  routing — no per-tenant fine-tuning; customization is prompt + RAG only. This is the
  default pattern for 90%+ of SaaS LLM apps.
- **Cursor** ([blog](https://cursor.com/blog/agent-best-practices)) uses workspace-level
  `.cursor/plans/` and `.cursor/rules/` files as a *persistent context overlay* — prompt-level
  customization per workspace, no per-workspace fine-tuning. Hooks extend agent behavior
  organization-wide.
- **Convirza on Predibase LoRAX** ([case study](https://predibase.com/blog/convirza-case-study))
  — the clearest production analog to our plan: 60+ LoRA adapters on a single Llama-3-8B,
  <2s p50 latency, 10× cheaper than OpenAI, 8% F1 improvement. Each adapter was a separate
  "performance indicator" rather than a tenant, but the serving topology is identical.
- **Paragon's "RAG vs fine-tuning for SaaS" analysis](https://www.useparagon.com/blog/rag-vs-finetuning-saas)**
  argues: start with RAG + per-tenant prompts, only fine-tune when the tenant has enough
  data and a clear measurable task-quality delta. Do not fine-tune by default per tenant.

**Adopt:**
- Three-layer stack (retrieval / prompt / weights) with clear precedence.
- RAG-first: LoRA is opt-in per broker when they've supplied enough labeled intakes.

**Avoid:**
- Per-tenant fine-tuning as the default — economically and operationally a trap at 500
  tenants if only a subset will ever have usable training data.
- Metadata-filter-only isolation (scanning a shared vector index with a `tenant_id` filter)
  — Pinecone docs call this out explicitly: latency scales with total tenants, and a
  filter-bug is a cross-tenant leak ([Pinecone multi-tenancy guide](https://www.pinecone.io/learn/series/vector-databases-in-production-for-busy-engineers/vector-database-multi-tenancy/)).
  Our per-directory ChromaDB is correct.

---

## 2. vLLM Multi-LoRA Serving, 2026 State

**What vLLM can do today:**

- **Static declaration at launch** via `--lora-modules`, `--max-loras N`, `--max-lora-rank R`
  ([vLLM LoRA docs](https://docs.vllm.ai/en/latest/features/lora/)). This is the default
  but requires redeploy to add adapters.
- **Runtime load/unload** via `/v1/load_lora_adapter` and `/v1/unload_lora_adapter` endpoints,
  gated behind `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`. vLLM docs explicitly warn this is
  **unsafe for production**: loads are not replicated across replicas, and there's no RBAC
  ([issue #6275](https://github.com/vllm-project/vllm/issues/6275)).
- **LoRA Resolver Plugins** (newer, [docs](https://docs.vllm.ai/en/stable/design/lora_resolver_plugins/),
  [PR #15733](https://github.com/vllm-project/vllm/pull/15733)): on first request for a
  new adapter name, a resolver fetches the adapter from FS/S3/HF and loads it. Built-in
  resolvers for local-dir and HF Hub; you can ship a custom S3 resolver in ~50 LOC. Still
  flagged by vLLM as risky in *untrusted* environments — fine for our trusted-tenant case.
- **Memory model:** adapters live in CPU memory (LRU-cached, `--max-cpu-loras`) and are
  paged into GPU only for active requests (`--max-loras`, the per-batch concurrent adapter
  count). Memory-per-adapter ≈ `2 × rank × Σ(hidden_dims)` × dtype bytes — for Qwen3.5-9B
  at rank=16 bf16 this is roughly 30–60 MB per adapter on disk and in CPU RAM; GPU
  incremental cost is negligible because of S-LoRA-style unified paging
  ([LMSYS S-LoRA blog](https://www.lmsys.org/blog/2023-11-15-slora/), arxiv
  [2311.03285](https://arxiv.org/abs/2311.03285)).

**Throughput reality check:**
- S-LoRA/LoRAX demonstrate 1000+ adapters on one GPU with SGMV kernels.
- Recent 2025 research ([arxiv 2505.03756](https://arxiv.org/html/2505.03756v1)) measured
  vLLM at **46.5% invalid KV cache blocks** under multi-LoRA load — the KV cache gets
  fragmented when adapters switch. Mitigation: batch requests by adapter (Predibase's
  "Continuous Multi-Adapter Batching") or accept the hit and oversize KV cache.
- [GitHub issue #10062](https://github.com/vllm-project/vllm/issues/10062) — throughput
  degradation on A100-40GB with even a single adapter if `--max-lora-rank` is too high.
  Set to the max rank you actually use, not a ceiling.

**Qwen3.5-9B + LoRA specifics:**
- [Issue #5298](https://github.com/vllm-project/vllm/issues/5298): Qwen LoRA adapters
  produce different results under vLLM vs HF `generate()`. Root cause in most reports:
  wrong chat template or tokenizer mismatch at inference. Lock your training chat template
  into the adapter directory (`chat_template.jinja`) and load it via `--chat-template`.
- [Issue #40307](https://github.com/vllm-project/vllm/issues/40307): some Qwen3.5 MoE
  variants have LoRA loading bugs; we're on the dense 9B, not affected.
- vLLM ≤ 0.16.0 has known Qwen3.5 support gaps. Pin to **≥ 0.17** for v4.

**Adopt for v4:**
```bash
vllm serve Qwen/Qwen3.5-9B \
  --served-model-name insurance-agent \
  --enable-lora \
  --max-loras 8 \
  --max-cpu-loras 128 \
  --max-lora-rank 16 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching
# Plus: custom S3 LoRA resolver plugin pointing at s3://accord-adapters/{tenant_slug}/latest/
```

- `--max-loras 8`: 8 concurrent adapters in a batch. Brokers have heavy-tail usage;
  anything higher hurts throughput without helping latency.
- `--max-cpu-loras 128`: keeps 128 adapters hot in CPU RAM. At ~60 MB each = ~8 GB RAM,
  trivial. Cold fetches from S3 add ~1–2s TTFT for the first request of an unused adapter.
- **Decision to revisit:** if we hit the multi-LoRA KV-cache fragmentation issue, switch
  the inference layer to **LoRAX** ([GitHub](https://github.com/predibase/lorax)). LoRAX
  is vLLM's commercial/open-source competitor *specifically* tuned for high-adapter-count
  serving, ships Helm charts + Prometheus, and was built on exactly the Convirza pattern
  we're planning. Keep this as a fallback, not a default.

**Avoid:**
- `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` with the default load/unload endpoints exposed —
  no RBAC, no replication. Wrap it behind our own internal control plane or use Resolver
  Plugins.
- Merging adapters into the base model at deploy time ("flattening") — loses multi-tenancy,
  only valid if exactly one tenant will ever use a given replica.

---

## 3. Per-Tenant Prompt Management

At 5 tenants, files work. At 50, you need versioning. At 500, you need a system.

**Open-source options surveyed:**

| Tool | Multi-tenant model | Self-host | Fit |
|---|---|---|---|
| **Langfuse** | Labels (`tenant-1`, `tenant-2`, `production`, `staging`) + RBAC + protected labels | Yes (OSS) | Strong — explicit pattern in [discussion #4169](https://github.com/orgs/langfuse/discussions/4169) |
| Humanloop | SaaS only, project-based | No | Wrong licensing model for PII-heavy workload |
| PromptLayer | SaaS-first, tag-based | Partial | Less explicit multi-tenant support |
| Promptfoo | Testing/eval, not runtime management | — | Complementary, not a replacement |

**Langfuse pattern for us** ([docs](https://langfuse.com/docs/prompt-management/features/prompt-version-control)):
- Each tenant gets a label `tenant:{slug}`.
- Each environment gets a label `env:production`, `env:staging`.
- Runtime: SDK fetches prompt by name + label set. Cached client-side.
- Rollback = flip the `production` label back to a previous version in the UI.
- **Protected labels** (enterprise RBAC) prevent non-admins from changing prod prompts.
- All changes audited (who, when, diff).

**Prompt-leakage defense** ([OWASP LLM01:2025](https://genai.owasp.org/llmrisk/llm01-prompt-injection/),
[Obsidian Security 2025](https://www.obsidiansecurity.com/blog/prompt-injection)):
- Never concatenate user input into the system prompt — use fixed slots.
- Harness/prompt is server-side only, never returned in API responses (we already do this).
- Tenant-scoped prompts must be fetched with a tenant ID that's validated against the
  authenticated principal, not taken from the request body.

**Cross-tenant cache leak — the sleeper bug:**
vLLM's automatic prefix caching creates a TTFT timing oracle: if tenant A's preamble and
tenant B's preamble differ only in one token, attacker B can detect whether tenant A
recently ran a query ([NDSS 2025 paper](https://www.ndss-symposium.org/wp-content/uploads/2025-1772-paper.pdf),
[RFC #16016](https://github.com/vllm-project/vllm/issues/16016)).

**Fix:** set `cache_salt` per tenant on every request (supported as of vLLM 0.8+). Same
tenant → cache hits; different tenant → separate cache lineage. Tiny throughput cost, closes
the oracle.

**Adopt:**
- Langfuse self-hosted for harness + prompt storage. Cache client-side with TTL.
- `cache_salt=tenant_slug` on every vLLM request.
- Protected `production` labels; all non-admin edits require review.

**Avoid:**
- Storing per-tenant prompts in the same SQLite row as session data — couples customization
  changes with conversation history and bloats sessions DB.
- Scattering `harness-{tenant}.md` in the repo. Works at 5, unmanageable at 50.

---

## 4. Adapter + Prompt Composition Order

The question: when a request has both a tenant-specific LoRA and a tenant-specific prompt
overlay, does the LoRA "see" the overlay in training?

**The rule, from the PEFT/vLLM serving chain:**

1. Prompt is assembled **first** (outside the model): `<base_preamble><harness_overlay><schema><turn>`.
2. Tokenizer converts to tokens.
3. Model forward pass runs with LoRA adapters fused into linear layers at each attention/MLP
   block — the LoRA sees the **already-tokenized prompt** as input, exactly like any text.
4. LoRA weight updates during training were conditioned on whatever prompt was in the training
   data. So: **the LoRA learned behavior relative to the prompt distribution it saw in
   training.**

**Practical implication:**
- If you train the LoRA with the overlay in the prompt, then serve with the overlay, behavior
  is consistent.
- If you train the LoRA without the overlay, then serve with the overlay, you're asking the
  model to generalize to an unseen prompt distribution — works for small deltas, fails for
  large ones.

**Production pattern ([Together AI docs](https://docs.together.ai/docs/lora-training-and-inference),
[NVIDIA NIM PEFT](https://docs.nvidia.com/nim/large-language-models/latest/peft.html)):**

> Inference prompt format MUST match training prompt format exactly, including the system
> prompt.

**For accord_v4:** freeze the harness structure during LoRA training. The training pipeline
should emit `(base_preamble, harness_at_training_time, schema, turn) -> target` tuples, and
the tenant's harness at inference should only add tenant-specific *facts* (e.g., "our agency
is Acme Brokerage, NY ZIP codes start with 1"), not *instruction* changes. Instruction-level
changes require adapter retraining.

**Recommendation: two-tier harness.**

```
harness = base_instructions (versioned, shared, changes trigger adapter retrain)
        + tenant_facts     (per-tenant, append-only, safe to change anytime)
```

Base is in Langfuse with label `env:production`. Tenant facts are in Langfuse with label
`tenant:{slug}`. The LoRA is trained against `base_instructions` + a representative
`tenant_facts` distribution; changes to tenant_facts at inference are safe; changes to
base_instructions mean we need to retrain every live LoRA.

**Adopt:**
- Two-tier harness with explicit instruction/facts split.
- Adapter version metadata records `base_instructions_version` — refuse to load an adapter
  if the base has drifted incompatibly.

**Avoid:**
- Editing base_instructions without a coordinated adapter rebuild cycle.
- Letting tenants edit instruction-level harness text — only fact-level overrides.

---

## 5. Data Isolation + Right to Delete

**The core tension:** an adapter fine-tuned on tenant data *is* tenant data. Weights encode
the training set. GDPR Article 17 erasure applies.

**Published guidance** ([TechGDPR](https://techgdpr.com/blog/reconciling-the-regulatory-clock/),
[Relyance](https://www.relyance.ai/blog/llm-gdpr-compliance), [MDPI 2025](https://www.mdpi.com/1999-5903/17/4/151)):

- Full retraining for every erasure request is economically infeasible.
- **Modular architectures (per-tenant adapters) are explicitly called out as the
  compliant pattern** — deletion = drop the adapter artifact + purge training data.
- Audit trails must survive deletion, so they must be built from *anonymized/pseudonymized*
  records, not from the raw PII.

**Deletion surfaces in accord_v4 — enumerate and delete atomically:**

| Surface | Path | Action on tenant delete |
|---|---|---|
| Sessions DB | SQLite `accord_ai.db` | `DELETE WHERE tenant=?`, VACUUM |
| Vector store | `chroma_data/{slug}/` | `rm -rf` directory |
| LoRA adapter | `s3://accord-adapters/{slug}/` | Delete prefix + invalidate resolver cache |
| Training data | `s3://accord-training/{slug}/` | Delete prefix |
| Harness/prompts | Langfuse prompts with label `tenant:{slug}` | Soft-delete in UI (label archived) |
| Filled PDFs | Drive folder per tenant | Revoke + delete folder |
| Logs | `logs/*.log` | PII already redacted (existing filter); retain for audit |
| KV cache | vLLM in-memory | Automatic eviction; `cache_salt` isolation means no cross-tenant residue |
| Checkpoints / backups | DB snapshots | Schedule rolling purge; document retention window |

**Audit-trail design (anonymized):**

```
tenant_events table:
  event_id, tenant_hash (sha256), event_type, timestamp, metadata_hash
  — NO raw tenant_slug, NO PII. Hash is irreversibly broken on key rotation.
```

On delete: flip a "tombstoned" flag in a separate `deletion_audit` table (who requested,
when, which surfaces purged), keep the tombstone indefinitely, rotate the hashing key so
that even the hash can no longer be linked back to the customer after key rotation —
this is the ["architectural separation" pattern](https://techgdpr.com/blog/reconciling-the-regulatory-clock/)
(raw data deleted, audit trail reconstructed from anonymized assets).

**Adopt:**
- Single `delete_tenant(slug)` orchestrator that walks all surfaces above transactionally.
- Deletion test suite that creates a tenant, exercises every surface, deletes, and asserts
  zero residue.
- Hashed audit log separate from session log.

**Avoid:**
- Relying on "eventual deletion" from backups. Document a retention window (e.g., 30 days)
  in the DPA and purge deterministically.
- Keeping the adapter "just in case" after deletion — it's a regulatory liability.

---

## Recommended Architecture Diagram

```
                  ┌─────────────────────────────────────────────────┐
                  │            Tenant Router (FastAPI)              │
                  │  - Auth: INTAKE_API_KEY → tenant_slug           │
                  │  - Rate limit per tenant                        │
                  │  - request_context.tenant = slug                │
                  └───────────────┬─────────────────────────────────┘
                                  │
          ┌───────────────────────┼──────────────────────────────┐
          │                       │                              │
          ▼                       ▼                              ▼
  ┌──────────────┐      ┌──────────────────┐         ┌────────────────────┐
  │ Prompt Fetch │      │ Retrieval        │         │ LLM Inference      │
  │ (Langfuse)   │      │ (ChromaDB)       │         │ (vLLM)             │
  │              │      │                  │         │                    │
  │ base (env:   │      │ chroma_data/     │         │ base: Qwen3.5-9B   │
  │   production)│      │   _shared/       │         │ lora: resolver     │
  │ + tenant:{s} │      │   {tenant}/      │         │   (S3 backend)     │
  │              │      │                  │         │   max_loras=8      │
  │ Cached 60s   │      │ bge-small embed  │         │   max_cpu_loras=128│
  └──────┬───────┘      └────────┬─────────┘         │ cache_salt={slug}  │
         │                       │                   └─────────┬──────────┘
         │    prompt + retrieved_context + lora_name={slug}    │
         └───────────────────────┴────────────────────┬────────┘
                                                      ▼
                                          ┌────────────────────┐
                                          │ Response           │
                                          └─────────┬──────────┘
                                                    │
                                                    ▼
                                          ┌────────────────────┐
                                          │ Session Store      │
                                          │ (SQLite, per-thread│
                                          │  conn, WAL)        │
                                          │ tenant indexed     │
                                          └────────────────────┘

  ──── storage backends ──────────────────────────────────────────────────

    Adapters:    s3://accord-adapters/{tenant}/v{N}/adapter_model.safetensors
    Training:    s3://accord-training/{tenant}/     (source for adapter retrain)
    Prompts:     Langfuse (self-hosted, Postgres)
    Sessions:    SQLite (existing pattern)
    Vectors:     ChromaDB per-tenant directories (existing pattern)
    Audit:       Postgres, anonymized hashes only
    Deletion:    delete_tenant(slug) orchestrator → all surfaces

  ──── cold paths (async, not in request flow) ──────────────────────────

    Judge/Refiner:      cold-path, disabled in prod by default (PII)
    Adapter retrain:    nightly batch on training_data/{tenant}/
    Harness promotion:  manual approval → Langfuse label flip
```

---

## Summary: Adopt / Avoid

**Adopt:**
1. vLLM LoRA Resolver Plugin with S3 backend, `--max-loras 8 --max-cpu-loras 128`.
2. Langfuse (self-hosted) for prompt+harness with per-tenant labels and protected prod label.
3. Two-tier harness: shared `base_instructions` (coupled to adapter version) + per-tenant
   `tenant_facts` (freely editable).
4. `cache_salt=tenant_slug` on every vLLM request to close the prefix-cache timing oracle.
5. Single `delete_tenant()` orchestrator covering DB, vectors, adapters, prompts, drive,
   training data. Anonymized audit trail survives deletion.
6. RAG-first; LoRA is opt-in per broker once they have enough labeled data.
7. Pin vLLM ≥ 0.17 for stable Qwen3.5-9B + LoRA behavior; lock chat template into
   adapter directories.

**Avoid:**
1. Per-tenant fine-tuning as the default — economics and ops don't work at 500 tenants.
2. `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` with raw endpoints exposed.
3. Metadata-filter-only vector isolation (shared index + `tenant_id` filter).
4. Editing `base_instructions` without a coordinated adapter retrain cycle.
5. Concatenating user input into the system prompt (OWASP LLM01).
6. Keeping a tenant's adapter after a deletion request — it is their data under GDPR.
7. Relying on eventual consistency / backup TTL for "deletion" — document and enforce
   a fixed retention window.

---

## Sources

- [vLLM LoRA Adapters docs](https://docs.vllm.ai/en/latest/features/lora/)
- [vLLM LoRA Resolver Plugins](https://docs.vllm.ai/en/stable/design/lora_resolver_plugins/)
- [vLLM Resolver Plugin PR #15733](https://github.com/vllm-project/vllm/pull/15733)
- [vLLM RFC: Distribute LoRA across deployment #12174](https://github.com/vllm-project/vllm/issues/12174)
- [vLLM RFC: Enhancing LoRA for Production #6275](https://github.com/vllm-project/vllm/issues/6275)
- [vLLM RFC: Cache Salting #16016](https://github.com/vllm-project/vllm/issues/16016)
- [vLLM Automatic Prefix Caching docs](https://docs.vllm.ai/en/latest/design/v1/prefix_caching.html)
- [vLLM Issue #10062: A100 LoRA throughput degradation](https://github.com/vllm-project/vllm/issues/10062)
- [vLLM Issue #5298: Qwen LoRA HF/vLLM divergence](https://github.com/vllm-project/vllm/issues/5298)
- [S-LoRA paper (arxiv 2311.03285)](https://arxiv.org/abs/2311.03285) — serving 1000s of adapters
- [LMSYS S-LoRA blog](https://www.lmsys.org/blog/2023-11-15-slora/)
- [arxiv 2505.03756](https://arxiv.org/html/2505.03756v1) — multi-LoRA KV cache fragmentation
- [NDSS 2025: Prompt leakage via KV-cache sharing](https://www.ndss-symposium.org/wp-content/uploads/2025-1772-paper.pdf)
- [Predibase LoRAX GitHub](https://github.com/predibase/lorax)
- [Predibase Convirza case study](https://predibase.com/blog/convirza-case-study) — 60+ adapters in production
- [LiteLLM multi-tenant architecture](https://docs.litellm.ai/docs/proxy/multi_tenant_architecture)
- [Cursor agent best practices](https://cursor.com/blog/agent-best-practices)
- [Paragon: RAG vs fine-tuning for SaaS](https://www.useparagon.com/blog/rag-vs-finetuning-saas)
- [Langfuse prompt version control](https://langfuse.com/docs/prompt-management/features/prompt-version-control)
- [Langfuse multi-tenant discussion #4169](https://github.com/orgs/langfuse/discussions/4169)
- [Pinecone multi-tenancy guide](https://www.pinecone.io/learn/series/vector-databases-in-production-for-busy-engineers/vector-database-multi-tenancy/)
- [OWASP LLM01:2025 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)
- [Obsidian Security: Prompt Injection 2025](https://www.obsidiansecurity.com/blog/prompt-injection)
- [Blaxel: Multi-tenant isolation for AI agents](https://blaxel.ai/blog/multi-tenant-isolation-ai-agents)
- [TechGDPR: AI Data Retention](https://techgdpr.com/blog/reconciling-the-regulatory-clock/)
- [Relyance: LLM GDPR Compliance](https://www.relyance.ai/blog/llm-gdpr-compliance)
- [MDPI 2025: GDPR and LLMs — Technical and Legal Obstacles](https://www.mdpi.com/1999-5903/17/4/151)
- [Together AI LoRA training/inference docs](https://docs.together.ai/docs/lora-training-and-inference)
- [NVIDIA NIM PEFT docs](https://docs.nvidia.com/nim/large-language-models/latest/peft.html)
