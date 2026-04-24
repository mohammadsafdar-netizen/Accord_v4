"""Extractor — user message + current submission state → diff submission.


Output shape is a CustomerSubmission whose `model_fields_set` contains
only the fields the LLM explicitly produced. That makes it directly
compatible with apply_diff() — fields the LLM didn't mention are not in
model_fields_set and won't touch current state during the merge.

Reuses extraction.SYSTEM_V1 + USER_TEMPLATE_V1 from Phase 4's prompts
module (pinned in the 5.a decision). Uses parse_submission_output for
wire-level robustness against markdown-wrapped JSON output.

Error model (parallel to Refiner):
  - Engine exceptions propagate — callers own retry semantics.
  - ExtractionOutputError wraps non-JSON / schema-invalid output. The LLM
    emitted garbage for this prompt shape; retrying at this layer won't help.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import List, Optional

from accord_ai.extraction.context import EMPTY_CONTEXT, ExtractionContext
from accord_ai.extraction.harness_content.v3_harness_snapshot import (
    V3_CA_LOB_HARNESS,
    V3_CORE_HARNESS,
    V3_GL_LOB_HARNESS,
    V3_HARNESS_FULL,
    V3_HARNESS_LIGHT,
    compose_harness_for_lobs,
)
from accord_ai.extraction.correction import (
    SYSTEM_CORRECTION_V1,
    detect_correction_target,
    is_correction,
)
# NER imports: kept only in tests (see extraction/ner.py). The
# production extract() path doesn't call them — see comment in extract()
# for rollback context.
from accord_ai.extraction.postprocess import run_postprocess
from accord_ai.harness.rules import apply_negation_rule
from accord_ai.llm.engine import Engine, Message
from accord_ai.llm.prompts import extraction as extraction_prompts
from accord_ai.llm.prompts import render
from accord_ai.llm.prompts.parsing import parse_submission_output
from accord_ai.llm.json_validity_tracker import TRACKER as _VALIDITY_TRACKER
from accord_ai.logging_config import get_logger
from accord_ai.schema import CustomerSubmission

_logger = get_logger("extractor")

# Schema is static — compute once at import, not per extract() call.
# The full schema includes $defs for Vehicle/Driver/Address/etc. so the LLM
# sees the exact nested shapes it must emit. This dump is ~7 200 tokens;
# paired with max_model_len=16384 there's ample room for prompt + output.
# Previously this was schema-stripped to fit an 8 192 window — confirmed
# that caused total extraction collapse on multi-vehicle/multi-driver
# input (2026-04-19 live test).
_FULL_SCHEMA_DICT = CustomerSubmission.model_json_schema()
# Remove runtime-only fields from the extraction schema — the LLM should never
# try to emit `conflicts` (it's populated by enrichers, not user input).
_SCHEMA_DICT = {
    **_FULL_SCHEMA_DICT,
    "properties": {
        k: v for k, v in _FULL_SCHEMA_DICT.get("properties", {}).items()
        if k != "conflicts"
    },
}
_SCHEMA_JSON = json.dumps(_SCHEMA_DICT)

# Sizing bucket boundaries — ported from accord_ai_v3 runner.py:368-372.
# v3 production uses these exact thresholds after 12 months of tuning;
# they balance "don't waste tokens on acknowledgement turns" against
# "don't truncate bulk extraction output".
_SHORT_MSG_CHARS = 30        # below → 512-token output (acknowledgements)
_LONG_MSG_CHARS  = 1500      # above → 4096-token output (bulk dumps)
_SHORT_OUT       = 512
_DEFAULT_OUT     = 2048      # v4 default — covers normal-turn diffs cleanly
_LONG_OUT        = 4096      # headroom for multi-driver / multi-vehicle JSON


def _adaptive_max_tokens(user_message: str) -> int:
    """Adaptive output budget matched to input length.

    Ported from v3's extraction/runner.py:368-372. The rule:
      * len < 30 chars   → 512   (e.g. "yes", "continue", "actually 7800")
      * 30 ≤ len ≤ 1500  → 2048  (normal conversational turn)
      * len > 1500       → 4096  (bulk dump — business + fleet + drivers)

    Why this beats a static default:
      - Short inputs don't need much output room; giving the LLM 2048 encourages
        over-generation (hallucinated adjacent fields).
      - Bulk inputs produce JSON that often exceeds 2048 tokens when fully
        populated (observed ~1200-1800 on realistic CA scenarios); 4096 gives
        comfortable headroom without wasting budget on shorter turns.

    Pure function; unit-testable without the engine.
    """
    n = len(user_message)
    if n < _SHORT_MSG_CHARS:
        return _SHORT_OUT
    if n > _LONG_MSG_CHARS:
        return _LONG_OUT
    return _DEFAULT_OUT


class ExtractionOutputError(ValueError):
    """Extractor produced non-JSON or schema-invalid output.

    Wraps json.JSONDecodeError or pydantic.ValidationError. Non-retryable
    at this layer — the LLM emitted garbage for this prompt shape.
    """


def _build_corrections_block(
    tenant: Optional[str],
    memory: Optional["CorrectionMemory"],
    enabled: bool,
) -> str:
    """Return a formatted corrections header for prompt injection, or empty string."""
    if not enabled or not tenant or memory is None:
        return ""
    try:
        entries = memory.get_relevant(tenant=tenant)
    except Exception as exc:
        _logger.warning(
            "correction_memory_query_failed tenant=%s error=%s", tenant, exc
        )
        return ""
    if not entries:
        return ""
    header = (
        "RECENT CORRECTIONS "
        "(this broker's prior fixes — avoid re-making these mistakes):"
    )
    lines = [e.as_prompt_line() for e in entries]
    return header + "\n" + "\n".join(lines) + "\n\n"


def _build_context_block(ctx: ExtractionContext) -> str:
    """Render a flow-context prefix for the extraction prompt, or ''."""
    if ctx.is_empty:
        return ""
    lines = []
    if ctx.current_flow:
        lines.append(f"FLOW: {ctx.current_flow}")
    if ctx.expected_fields:
        lines.append(f"FOCUS FIELDS: {', '.join(ctx.expected_fields)}")
    if ctx.question_text:
        lines.append(f"QUESTION ASKED: {ctx.question_text}")
    # rag_snippets rendered here in Phase 3.4
    return "\n".join(lines) + "\n\n"


_HARNESS_BLOCKS: dict[str, str] = {
    "none": "",
    # Legacy v1.0 content — retained for Step 25 A/B/C/D matrix comparisons.
    "light": V3_HARNESS_LIGHT.strip() + "\n\n",
    "full": V3_HARNESS_FULL.strip() + "\n\n",
    # Curated v6.1 — v3's actual production harness (not harness.md).
    # Research 2026-04-22 identified we were porting the wrong file.
    "core": V3_CORE_HARNESS.strip() + "\n\n" if V3_CORE_HARNESS.strip() else "",
    # core + commercial_auto LOB rules.
    "core_ca": (
        compose_harness_for_lobs(["commercial_auto"]).strip() + "\n\n"
        if compose_harness_for_lobs(["commercial_auto"]).strip()
        else ""
    ),
    # core + general_liability LOB rules.
    "core_gl": (
        compose_harness_for_lobs(["general_liability"]).strip() + "\n\n"
        if compose_harness_for_lobs(["general_liability"]).strip()
        else ""
    ),
}


class Extractor:
    """Turn a user message + current state into a CustomerSubmission diff."""

    def __init__(
        self,
        engine: Engine,
        memory: Optional["CorrectionMemory"] = None,
        memory_enabled: bool = True,
        experiment_harness: str = "none",
        extraction_mode: str = "xgrammar",
        harness_position: str = "before",
        ner_postprocess: bool = False,
    ) -> None:
        """Build an Extractor.

        harness_position controls where the harness block sits in the
        system message:

        - "before" (legacy default): harness + SYSTEM_V2. This is what
          Step 25 matrix tested — placement research identified it as
          the likely cause of harness regression (Qwen3.5-9B is
          sensitive to pre-SYSTEM content per vLLM issue #23404).

        - "after" (v3's pattern): SYSTEM_V2 + harness. v3 places harness
          at the END of the system message after the schema reminder.
          Research 2026-04-22 indicates this is the placement that makes
          harness text productive on Qwen.

        harness_position is a no-op when experiment_harness="none".
        """
        if harness_position not in ("before", "after"):
            raise ValueError(
                f"harness_position must be 'before' or 'after', got "
                f"{harness_position!r}"
            )
        self._engine = engine
        self._memory = memory
        self._memory_enabled = memory_enabled and memory is not None
        self._experiment_harness = experiment_harness
        self._extraction_mode = extraction_mode
        self._harness_position = harness_position
        # Static fallback — used when the experiment_harness flag doesn't
        # support dynamic composition (e.g. "none", "light", "full"). For
        # the "core*" family, we compose per-request with tenant overlay.
        self._static_harness_block = _HARNESS_BLOCKS.get(experiment_harness, "")
        # Per-(mode, tenant) composition cache — avoids re-reading tenant
        # overlay files on every extraction call while still supporting
        # per-broker customization. Invalidates on process restart.
        self._harness_cache: dict[tuple[str, str | None], str] = {}
        # NER post-extraction validation (Thread 3 port). Off by default
        # until an eval confirms productive composition with the harness
        # stack; enable via NER_POSTPROCESS=true.
        self._ner_postprocess = ner_postprocess

    @property
    def _harness_block(self) -> str:
        """Back-compat property — older tests read this attribute directly.

        For static modes, returns the precomputed block.
        For dynamic modes ("core*"), returns an empty string when no tenant
        is set (since composition requires request context).
        """
        return self._static_harness_block

    def _get_harness_block(self) -> str:
        """Return the harness block for the current request.

        For static modes (none/light/full): returns the precomputed block.
        For dynamic modes (core/core_ca/core_gl): composes core.md + LOB
        overlay + per-tenant overlay based on current request_context tenant.

        Caches per (mode, tenant) pair to avoid re-reading files every call.
        """
        # Static modes: precomputed at __init__
        if self._experiment_harness not in ("core", "core_ca", "core_gl"):
            return self._static_harness_block

        # Dynamic: compose per-request with tenant overlay
        from accord_ai.request_context import get_tenant
        tenant = get_tenant()
        cache_key = (self._experiment_harness, tenant)
        cached = self._harness_cache.get(cache_key)
        if cached is not None:
            return cached

        from accord_ai.extraction.harness_content.v3_harness_snapshot import (
            compose_harness_for_lobs,
        )
        lobs_for_mode = {
            "core": None,
            "core_ca": ["commercial_auto"],
            "core_gl": ["general_liability"],
        }
        active_lobs = lobs_for_mode.get(self._experiment_harness)
        composed = compose_harness_for_lobs(active_lobs=active_lobs, tenant=tenant)
        block = composed.strip() + "\n\n" if composed.strip() else ""
        self._harness_cache[cache_key] = block
        return block

    async def extract(
        self,
        *,
        user_message: str,
        current_submission: CustomerSubmission,
        context: ExtractionContext = EMPTY_CONTEXT,
    ) -> CustomerSubmission:
        """Extract a diff from user_message given current_submission as context.

        Returns a CustomerSubmission whose model_fields_set contains only
        the fields the LLM explicitly set — suitable as a diff argument to
        apply_diff(current_submission, diff).
        """
        # Extractor sees only the non-null "what we already know" block —
        # empty-submission case drops from 750 B of nulls to ~75 B.
        current_json = current_submission.model_dump_json(exclude_none=True)

        # Correction memory injection (Phase 2.4): prepend recent broker
        # corrections to the user message slot.  Failure-isolated — if the
        # query fails or returns nothing, user_message is used unchanged.
        from accord_ai.request_context import get_tenant
        _tenant = get_tenant()
        corrections_block = _build_corrections_block(
            _tenant, self._memory, self._memory_enabled
        )
        context_block = _build_context_block(context)

        # Compose the volatile user-message slot:
        # corrections first, flow context second, user text last.
        # When either prefix is present we label the user text so the
        # model can distinguish the blocks. The anti-narrowing instruction
        # ("also extract any other fields") is the guardrail that keeps
        # the model from only emitting focus fields.
        if corrections_block or context_block:
            user_text_slot = (
                corrections_block
                + context_block
                + f"USER MESSAGE: {user_message}\n\n"
                "Extract all relevant fields as JSON. "
                "Prioritize focus fields if listed; "
                "also extract any other fields the user provides."
            )
        else:
            user_text_slot = user_message
        effective_message = user_text_slot

        # NER pre-extraction (Phase A step 5) — DISABLED in the
        # production path after live tests showed it regressing
        # multi-vehicle-fleet and negation scenarios. The code path
        # (tag_entities, format_ner_hints, validate_extraction_with_ner)
        # remains in accord_ai/extraction/ner.py with its unit tests
        # so a narrower re-enablement (e.g. only VIN/EIN regex hints,
        # or only when LLM output is schema-invalid) is a feature-
        # flag-away. See memory note on NER regressions.
        user_content = render(
            extraction_prompts.USER_TEMPLATE_V1,
            schema=_SCHEMA_JSON,
            current_state=current_json,
            user_message=effective_message,
        )

        # Correction-turn detection (Phase A step 4). If the user's
        # message is a correction ("actually the EIN is ...", "wait, 12
        # trucks not 10"), swap to a focused system prompt that tells
        # the LLM to emit ONLY the changed field — not re-extract the
        # whole submission. Optionally surfaces the v4 schema path we
        # think is being corrected so the LLM has a target.
        is_corr = is_correction(user_message)
        target = detect_correction_target(user_message) if is_corr else None
        # Step 3A: when is_corr=True but target=None, the narrow
        # SYSTEM_CORRECTION_V1 prompt with full-schema guided_json
        # produces flat output (e.g. {"year": "2023"}) that fails schema
        # validation — the nested path (lob_details.vehicles[0].year) is
        # lost. Diagnosed on correction-vehicle-year: turn 2 extraction
        # failed → year stayed at prior value. Fix: the correction branch
        # only fires when we actually know which field to hint. Targetless
        # "corrections" fall through to SYSTEM_V2 so the LLM keeps its
        # nested-schema posture and produces valid JSON.
        if is_corr and target is not None:
            system_content = SYSTEM_CORRECTION_V1
            # Append the hint to the user content (not the system
            # prompt) so prefix cache for SYSTEM_CORRECTION_V1 stays
            # hot across correction turns.
            user_content = (
                user_content
                + f"\n\nFIELD TO CORRECT: {target}\n"
                f"Output only the corrected value(s) at that path."
            )
        else:
            # Normal extraction — SYSTEM_V2 (LOB routing rules, no harness).
            # Postmortem 1A showed SYSTEM_V3's expanded harness competing
            # with complex-object emission under guided_json on
            # vehicle/driver middle turns — extraction failures in 3 of
            # 4 regressed scenarios. Harness is quarantined to the
            # refiner path (see harness/refiner.py gating).
            #
            # Step 25 + 2026-04-23 placement research:
            # - harness_position="before" → harness + SYSTEM_V2 (Step 25 default)
            # - harness_position="after" → SYSTEM_V2 + harness (v3's pattern)
            # The research strongly suggests "after" matches Qwen3.5-9B's
            # training distribution; "before" may degrade because Qwen weights
            # the first system content most heavily.
            if not self._get_harness_block():
                # No harness → both positions produce identical output.
                system_content = extraction_prompts.SYSTEM_V2
            elif self._harness_position == "after":
                system_content = (
                    extraction_prompts.SYSTEM_V2.rstrip()
                    + "\n\n"
                    + self._get_harness_block()
                )
            else:
                system_content = self._get_harness_block() + extraction_prompts.SYSTEM_V2

        # --- Structured trace for 1A postmortem diagnostic ---
        # Emits at INFO so the default log level captures it. Grep for
        # `extraction_route=` in logs/app.log to reconstruct per-turn
        # routing. The v4 extractor NEVER narrows the schema (both paths
        # pass the full CustomerSubmission schema via guided_json), so
        # schema_paths_kept="full" is a constant — logged explicitly to
        # rule out failure mode (d) from the postmortem checklist.
        msg_digest = hashlib.md5(user_message.encode("utf-8")).hexdigest()[:8]
        _logger.info(
            "extraction_route msg=%s is_correction=%s prompt=%s "
            "field_hint=%s msg_len=%d schema_paths_kept=full",
            msg_digest, is_corr,
            "SYSTEM_CORRECTION_V1" if (is_corr and target is not None)
            else "SYSTEM_V2",
            target, len(user_message),
        )

        messages: List[Message] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        # Adaptive sizing — see _adaptive_max_tokens docstring.
        # Short turns get 512 (prevents over-generation hallucination);
        # bulk turns get 4096 (prevents mid-JSON truncation on realistic
        # full-submission dumps).
        max_tokens = _adaptive_max_tokens(user_message)

        # Harness-aware cap: when harness is active, the system message
        # grows by ~3000-4000 tokens. On 16k context models, this collides
        # with the output budget on bulk turns (observed: 12.3k input +
        # 4k output = 16.4k > 16.3k ceiling). Cap output at 2048 when
        # harness is active to preserve ~2k of input budget headroom for
        # bulk fleet dumps.
        if self._get_harness_block():
            max_tokens = min(max_tokens, 2048)

        # json_schema: vLLM 0.18+ constrains output tokens to the schema via
        # xgrammar. Guarantees valid JSON — eliminates the
        # ExtractionOutputError-from-malformed-output path entirely. On
        # providers without structured-output support the kwarg is ignored
        # and parse_submission_output still catches any malformed result.
        response = await self._engine.generate(
            messages,
            max_tokens=max_tokens,
            json_schema=_SCHEMA_DICT,
        )

        # Post-processing (Phase A step 2): 5-step cleanup
        # (unfold/strip/phantom/coerce/cap). Runs between JSON parse
        # and pydantic validation.
        #
        # Step 6 (validate_extraction_with_ner) INTENTIONALLY NOT WIRED
        # in the production path right now. Live 5-scenario test showed
        # multi-five-vehicle-fleet regressing from F1=0.324 to F1=0.121
        # with the validator enabled — the fleet-driver PERSON entities
        # interact with the validator's fixes in ways that strip real
        # extracted data. The function + tests remain in accord_ai/
        # extraction/ner.py for future targeted use (e.g. only applying
        # fix 4 website-injection, or behind a feature flag).
        current_state_dict = current_submission.model_dump(
            mode="python", exclude_none=False,
        )

        def _postprocess(delta: dict) -> dict:
            # 5-step cleanup (unfold/strip/phantom/coerce/cap), then
            # deterministic harness rules (negation fires when user
            # text contains "no hired auto" / "no hazmat" / etc.), then
            # NER post-extraction validation if enabled.
            #
            # Rules sit AFTER run_postprocess so they see a clean
            # dict but BEFORE pydantic-validate so they can add fields
            # that would otherwise be missing.
            delta = run_postprocess(delta, current_state_dict)
            delta = apply_negation_rule(user_message, delta)
            if self._ner_postprocess:
                from accord_ai.extraction.ner import (
                    tag_entities,
                    validate_extraction_with_ner,
                )
                try:
                    ner_tags = tag_entities(user_message)
                    delta = validate_extraction_with_ner(
                        delta, ner_tags, current_state=current_state_dict,
                    )
                except Exception as exc:  # spaCy model missing, etc.
                    _logger.warning(
                        "NER postprocess failed (silently skipped): %s", exc,
                    )
            return delta

        _parse_ok = True
        try:
            diff = parse_submission_output(
                response.text,
                error_cls=ExtractionOutputError,
                postprocess=_postprocess,
            )
        except ExtractionOutputError:
            _parse_ok = False
            raise

        # --- Extraction output trace (1A postmortem) ---
        # Logs the top-level keys that survived parse+postprocess+validate,
        # correlated with the earlier extraction_route line via msg_digest.
        # Sorted for deterministic diffing across runs.
        extracted_keys = sorted(
            k for k in diff.model_fields_set
            if getattr(diff, k, None) not in (None, [], {})
        )
        _logger.info(
            "extraction_output msg=%s extracted_fields=%s",
            msg_digest, extracted_keys,
        )

        # Step 25: record JSON validity. Placed after extracted_keys so the
        # field count is accurate. Mode is stored on the extractor at
        # construction time to avoid piercing RetryingEngine's wrapping.
        _VALIDITY_TRACKER.record(
            valid_first_try=_parse_ok,
            mode=self._extraction_mode,
            harness=self._experiment_harness,
            extracted_field_count=len(extracted_keys),
        )

        return diff
