"""Extract widget ground truth from blank ACORD PDFs.

Output: accord_ai/form_fields_enriched/form_<N>_widgets.json

Run whenever a blank PDF changes:
    uv run python scripts/extract_pdf_widgets.py

This script reads the actual PDF bytes (not v3's stale metadata) so mapping
decisions in accord_ai/forms/mapper.py are always validated against PDF truth.
The test tests/test_forms_widget_ground_truth.py consumes these files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import fitz  # PyMuPDF


ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = ROOT / "accord_ai" / "form_templates"
OUT_DIR = ROOT / "accord_ai" / "form_fields_enriched"


# PyMuPDF widget type code → human label
WIDGET_TYPE_NAMES = {
    0: "unknown",
    1: "pushbutton",
    2: "checkbox",
    3: "combobox",
    4: "listbox",
    5: "radio",
    6: "signature",
    7: "text",
}


def extract(pdf_path: Path) -> dict:
    doc = fitz.open(pdf_path)
    try:
        widgets = []
        for page_idx, page in enumerate(doc):
            for w in page.widgets():
                widgets.append({
                    "name":        w.field_name,
                    "type":        WIDGET_TYPE_NAMES.get(w.field_type, "unknown"),
                    "type_code":   w.field_type,
                    "tooltip":     (w.field_label or "").strip(),
                    "page":        page_idx,
                    "rect":        [round(c, 2) for c in w.rect],
                    "flags":       w.field_flags,
                    "is_readonly": bool(w.field_flags & 1),
                    "choices":     getattr(w, "choice_values", None),
                })
        return {
            "form":           pdf_path.stem.split("_")[1],
            "pdf_path":       str(pdf_path.relative_to(ROOT)),
            "widget_count":   len(widgets),
            "unique_names":   len({w["name"] for w in widgets}),
            "widgets":        widgets,
        }
    finally:
        doc.close()


def main() -> int:
    OUT_DIR.mkdir(exist_ok=True)
    pdfs = sorted(TEMPLATES.glob("acord_*_blank.pdf"))
    if not pdfs:
        print(f"no PDFs found in {TEMPLATES}", file=sys.stderr)
        return 1
    for pdf in pdfs:
        data = extract(pdf)
        out_path = OUT_DIR / f"form_{data['form']}_widgets.json"
        out_path.write_text(json.dumps(data, indent=2, sort_keys=True))
        print(f"{data['form']}: {data['widget_count']:>5} widgets "
              f"({data['unique_names']:>4} unique) → {out_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
