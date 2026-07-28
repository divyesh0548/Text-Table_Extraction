"""Load shared OCR settings from OCR/.env."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

OCR_DIR = Path(__file__).resolve().parent
load_dotenv(OCR_DIR / ".env")


def get_input_pdf() -> Path:
    """
    Return the PDF path from INPUT_PDF in OCR/.env.

    Relative paths are resolved against the OCR folder.
    """
    raw = os.getenv("INPUT_PDF", "").strip().strip('"').strip("'")
    if not raw:
        raise ValueError(
            "INPUT_PDF is not set in OCR/.env. "
            r'Example: INPUT_PDF=Current-test\MBP 1 JSD 2026.pdf'
        )

    path = Path(raw)
    if not path.is_absolute():
        path = (OCR_DIR / path).resolve()
    else:
        path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(f"INPUT_PDF not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(
            f"INPUT_PDF must be a PDF file, got: {path.suffix or '(no extension)'}"
        )

    return path
