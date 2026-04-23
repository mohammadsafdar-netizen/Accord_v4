"""Accord AI — insurance intake library.

Top-level imports for library consumers:

    from accord_ai import build_intake_app, Settings

    app = build_intake_app(Settings())
    sid = app.store.create_session()
    result = await app.controller.process_turn(
        session_id=sid, user_message="we are Acme",
    )

For test doubles, use the testing subpackage:

    from accord_ai.testing import FakeEngine, FakeEmbedder, FakeVectorStore
"""
from accord_ai.app import IntakeApp, build_intake_app
from accord_ai.audit import record_audit_event
from accord_ai.config import Settings
from accord_ai.forms import forms_for_lob, list_all_forms, load_form_spec

__version__ = "0.1.0"

__all__ = [
    "IntakeApp",
    "Settings",
    "build_intake_app",
    "forms_for_lob",
    "list_all_forms",
    "load_form_spec",
    "record_audit_event",
    "__version__",
]
