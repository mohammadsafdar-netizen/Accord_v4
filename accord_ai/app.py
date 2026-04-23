"""IntakeApp — top-level wired application.

One call (`build_intake_app(settings)`) constructs the full stack:
SessionStore + LLM engines (main + refiner) + Extractor + HarnessManager
+ Responder + ConversationController. Phase 8 CLI and Phase 9 API both
bootstrap through this single factory.

Dataclass is frozen; internal state (session rows, vector DB) lives
inside the components themselves. Callers reach into the fields directly
— stable access paths so the reach-in pattern doesn't hurt us.

Test injection: pass `engine=` and/or `refiner_engine=` to override the
real client construction with FakeEngines or stubs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from accord_ai.config import Settings
from accord_ai.conversation.controller import ConversationController
from accord_ai.conversation.flow_engine import FlowEngine
from accord_ai.conversation.flow_loader import load_flows
from accord_ai.conversation.responder import Responder
from accord_ai.core.store import SessionStore
from accord_ai.extraction.extractor import Extractor
from accord_ai.feedback.transcript_capture import TranscriptCapture, TranscriptCaptureConfig
from accord_ai.forms import FilledPdfStore
from accord_ai.harness.judge import SchemaJudge
from accord_ai.harness.manager import HarnessManager
from accord_ai.harness.refiner import Refiner, build_refiner
from accord_ai.integrations.backend import BackendClient, build_backend_client
from accord_ai.integrations.drive import DriveClient, build_drive_client
from accord_ai.llm import Engine, build_engine, build_refiner_engine
from accord_ai.validation.census_naics import NaicsValidator
from accord_ai.validation.inline import build_inline_runner
from accord_ai.validation.nhtsa_vpic import NhtsaVpicValidator
from accord_ai.validation.zippopotam import ZippopotamValidator


@dataclass(frozen=True)
class IntakeApp:
    """Wired intake application. Frozen — fields are stable access paths."""
    settings: Settings
    store: SessionStore
    controller: ConversationController
    # Exposed for CLI / API consumers that need to generate greetings,
    # render verdicts independent of process_turn, etc. Reach-in pattern.
    responder: Responder
    judge: SchemaJudge
    filled_pdf_store: FilledPdfStore
    # Exposed for OCR endpoint (Phase 1.5) — the main engine is shared with
    # the extractor. None in legacy test fixtures that don't set it.
    engine: Optional[Engine] = None
    # Optional outbound integrations — each factory returns None when its
    # *_enabled flag is False, so IntakeApp built from default settings
    # still works in tests / offline environments.
    backend_client: Optional[BackendClient] = None
    drive_client: Optional[DriveClient] = None
    transcript_capture: Optional[TranscriptCapture] = None


def build_intake_app(
    settings: Settings,
    *,
    engine: Optional[Engine] = None,
    refiner_engine: Optional[Engine] = None,
) -> IntakeApp:
    """Construct the full intake stack from Settings.

    Args:
        settings: the application config.
        engine: override for the main LLM engine (extractor + responder share
            this). Default: build_engine(settings) → real OpenAI-compat stack.
        refiner_engine: override for the refiner's LLM engine. Default:
            build_refiner_engine(settings) → may point at a different provider
            per harness_refiner_* settings; privacy-gate-checked.
    """
    main = engine if engine is not None else build_engine(settings)
    refiner_llm = (
        refiner_engine if refiner_engine is not None
        else build_refiner_engine(settings)
    )

    store = SessionStore(settings.db_path)
    filled_pdf_store = FilledPdfStore(settings.filled_pdf_dir)

    judge = SchemaJudge()
    responder = Responder(main)

    cascade = build_refiner(settings)
    refiner_instance = cascade if cascade is not None else Refiner(refiner_llm)

    inline_runner = build_inline_runner(
        validators=[NhtsaVpicValidator(), ZippopotamValidator(), NaicsValidator()],
        timeout_s=2.0,
    )

    from accord_ai.feedback.memory import CorrectionMemory
    correction_memory = CorrectionMemory(db_path=settings.db_path)

    from accord_ai.feedback.transcript_capture import TranscriptCapture, TranscriptCaptureConfig
    transcript_capture = TranscriptCapture(
        TranscriptCaptureConfig(
            output_dir=settings.training_data_dir,
            enabled=settings.enable_transcript_capture,
        )
    )

    flow_engine = FlowEngine(load_flows()) if settings.use_flow_engine else None

    controller = ConversationController(
        store=store,
        extractor=Extractor(
            main,
            memory=correction_memory,
            memory_enabled=settings.enable_correction_memory,
            experiment_harness=settings.experiment_harness,
            extraction_mode=settings.extraction_mode.value,
            harness_position=settings.harness_position,
        ),
        harness=HarnessManager(
            judge=judge,
            refiner=refiner_instance,
            max_refines=settings.harness_max_refines,
        ),
        responder=responder,
        inline_runner=inline_runner,
        flow_engine=flow_engine,
        extraction_context_enabled=settings.extraction_context,
    )

    backend_client = build_backend_client(settings, audit_store=store)
    drive_client = build_drive_client(settings, audit_store=store)

    return IntakeApp(
        settings=settings,
        store=store,
        controller=controller,
        responder=responder,
        transcript_capture=transcript_capture,
        judge=judge,
        filled_pdf_store=filled_pdf_store,
        engine=main,
        backend_client=backend_client,
        drive_client=drive_client,
    )
