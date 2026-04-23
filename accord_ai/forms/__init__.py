"""Form registry — ACORD blank templates + field specs + LOB→form mapping.

Read-only in P10.A.1. PDF filling lands in P10.A.2; field mapping in P10.A.3.
"""
from accord_ai.forms.filler import FillResult, fill_form
from accord_ai.forms.mapper import map_submission, map_submission_to_form
from accord_ai.forms.pipeline import FilledForm, fill_submission
from accord_ai.forms.registry import (
    FormField,
    FormSpec,
    UnknownFormError,
    forms_for_lob,
    list_all_forms,
    load_form_spec,
    template_path,
)
from accord_ai.forms.storage import FilledPdfStore

__all__ = [
    "FillResult",
    "FilledForm",
    "FilledPdfStore",
    "FormField",
    "FormSpec",
    "UnknownFormError",
    "fill_form",
    "fill_submission",
    "forms_for_lob",
    "list_all_forms",
    "load_form_spec",
    "map_submission",
    "map_submission_to_form",
    "template_path",
]
