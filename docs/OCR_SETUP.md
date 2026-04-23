# OCR Setup — Phase 1.5

## System Dependency

Tesseract OCR must be installed on the host machine:

```bash
sudo apt install tesseract-ocr tesseract-ocr-eng
```

Verify installation:

```bash
which tesseract
tesseract --version
```

If tesseract is missing, the `/upload-image` endpoint returns **503** with:
```json
{"detail": {"error": "ocr_unavailable", "reason": "tesseract binary not found — run: sudo apt install tesseract-ocr tesseract-ocr-eng"}}
```

## Python Dependencies

`pytesseract` and `Pillow` are required. Pillow ships with the project.
Install pytesseract:

```bash
uv pip install "pytesseract>=0.3.10"
```

Or add to your `pyproject.toml` dependencies:
```
"pytesseract>=0.3.10",
"Pillow>=10.0",
```

## Supported Document Types

| `kind` value            | Description                        |
|-------------------------|------------------------------------|
| `drivers_license`       | State-issued driver's license      |
| `insurance_card`        | Auto/commercial insurance card     |
| `vehicle_registration`  | Vehicle registration certificate   |

## Pipeline

1. Tesseract extracts raw text (local, CPU, no GPU load)
2. Qwen3.5-9B (existing vLLM) parses raw text → structured fields

## Privacy

All processing is local. No image data or OCR text leaves the host.
Log output is PII-redacted via `redact_pii_text()` before any debug logging.

## LOB Rules

| Document type           | Allowed LOB         | Rejection (422) for  |
|-------------------------|---------------------|----------------------|
| `drivers_license`       | `commercial_auto`   | other LOBs (no-op)   |
| `vehicle_registration`  | `commercial_auto`   | GL, WC, any other    |
| `insurance_card`        | `workers_comp`      | other LOBs (no-op)   |

## Upgrade Path

- Phase 2: Replace Tesseract with EasyOCR/PaddleOCR for better photo accuracy
- Future: Qwen2.5-VL when dedicated vision GPU available
