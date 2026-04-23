"""Generate synthetic OCR test fixtures.

All data is entirely fake — no real customer PII.
Run once: uv run python tests/fixtures/ocr/generate.py

Output files (all synthetic):
  dl_clean.png           — readable driver license
  dl_rotated.png         — same DL rotated 90° (tests preprocessing robustness)
  insurance_card_clean.png
  registration_clean.png
  unreadable.png         — near-blank / white image (triggers OCRReadError)
  not_image.txt          — plain text bytes (triggers OCRReadError)
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


_OUT = Path(__file__).parent


def _make_text_image(lines: list[str], size=(600, 300), bg="white", fg="black") -> Image.Image:
    img = Image.new("RGB", size, color=bg)
    draw = ImageDraw.Draw(img)
    y = 10
    for line in lines:
        draw.text((10, y), line, fill=fg)
        y += 22
    return img


def generate():
    # DL clean
    dl = _make_text_image([
        "DRIVER LICENSE",
        "STATE OF TEXAS",
        "LN: DOE   FN: JOHN",
        "DOB: 01/15/1985",
        "DL: TX12345678",
        "CLASS: C",
        "EXP: 01/15/2029",
        "123 MAIN ST AUSTIN TX 78701",
    ], size=(640, 200))
    dl.save(_OUT / "dl_clean.png")

    # DL rotated 90°
    dl.rotate(90, expand=True).save(_OUT / "dl_rotated.png")

    # Insurance card
    ic = _make_text_image([
        "INSURANCE CARD",
        "CARRIER: ACME INSURANCE CO",
        "POLICY: POL-9876543",
        "NAMED INSURED: JOHN DOE",
        "EFF: 01/01/2025",
        "EXP: 01/01/2026",
    ], size=(640, 180))
    ic.save(_OUT / "insurance_card_clean.png")

    # Registration
    reg = _make_text_image([
        "VEHICLE REGISTRATION",
        "VIN: 1HGCM82633A004352",
        "YEAR: 2020  MAKE: HONDA  MODEL: CIVIC",
        "STATE: TX",
        "OWNER: JOHN DOE",
        "EXP: 12/31/2025",
    ], size=(640, 180))
    reg.save(_OUT / "registration_clean.png")

    # Unreadable — very low contrast (nearly white text on white)
    unreadable = Image.new("RGB", (200, 100), color=(250, 250, 250))
    draw = ImageDraw.Draw(unreadable)
    draw.text((10, 40), ".", fill=(255, 255, 255))
    unreadable.save(_OUT / "unreadable.png")

    # Not an image — plain text bytes
    (_OUT / "not_image.txt").write_bytes(b"This is not an image file.\n")

    print("Generated fixtures in", _OUT)


if __name__ == "__main__":
    generate()
