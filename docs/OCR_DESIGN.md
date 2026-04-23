# OCR Design — Phase 1.5

## v3 Reference Analysis

v3 (`accord_ai_v3/accord_ai/extraction/ocr.py`) supported three engines:
1. **Mindee API** — cloud, structured output, 250 pages/month free
2. **Tesseract** — local, raw text only, regex post-processing
3. **Mock** — for CI

v3 defaulted to Mindee when `MINDEE_API_KEY` was set, Tesseract otherwise.

## v4 Deliberate Divergence

**Privacy constraint (hard):** No cloud OCR in v4. OCR content is raw PII
(names, DOBs, SSNs on some DLs, VINs, policy numbers). Sending this to an
external API requires explicit per-tenant approval that doesn't exist yet.
→ Local-only. Mindee path dropped entirely.

## Engine Choice: Tesseract + Qwen3.5-9B Two-Stage

1. **Tesseract** extracts raw text from the image (CPU, no GPU load).
2. **Qwen3.5-9B** (existing vLLM) parses raw text → structured fields via
   `guided_json`.

Why not regex (v3 approach): regex DL parsing is fragile; state formats vary
widely. Qwen3.5-9B with guided_json produces higher-quality structured output
with the same local constraint.

Upgrade path: swap Tesseract for EasyOCR/PaddleOCR for better photo-quality
accuracy, or move to Qwen2.5-VL when a dedicated vision GPU is available.

## Public API

```python
from accord_ai.extraction.ocr import ocr_document, DocKind

fields = await ocr_document(image_bytes, kind="drivers_license", engine=llm_engine)
# returns DriverLicenseFields | InsuranceCardFields | RegistrationFields
```

Single dispatcher on `kind`. Three concrete Pydantic output types (v3 used
flat `dict[str, Any]`).

## Preprocessing (new in v4, absent in v3)

1. Convert to RGB (handles HEIC/CMYK edge cases)
2. Upscale if longest edge < 1500px (phone photos are often 800–1200px)
3. Convert to grayscale (reduces noise for Tesseract)

## Error Handling (exceptions, not error-field)

| Condition | Exception |
|-----------|-----------|
| Non-image bytes | `OCRReadError` |
| Tesseract returns empty/whitespace | `OCRReadError` |
| `tesseract` binary not installed | `OCRConfigError` (hint: `sudo apt install tesseract-ocr`) |

`OCRConfigError` → API returns 503 (dependency unavailable)
`OCRReadError` → API returns 422 (bad input)

## Merge Strategy

- `drivers_license` → `merge_drivers()` (Phase 1.4), keyed on `license_number`
- `vehicle_registration` → `merge_vehicles()` (Phase 1.4), keyed on VIN
- `insurance_card` → prepend to `CustomerSubmission.prior_insurance[]`

## LOB Guard

Vehicle registration implies commercial_auto. If the session's active LOB is
not `commercial_auto`, reject with 422 ("vehicle_registration requires CA LOB").

## PII Logging

OCR raw text is passed through `redact_pii_text()` before any debug logging.
Structured fields are logged at DEBUG only — never at INFO or above.

## System Dependency

```bash
sudo apt install tesseract-ocr tesseract-ocr-eng
```

Python packages: `pytesseract>=0.3.10`, `Pillow>=10.0` (Pillow already in deps).
