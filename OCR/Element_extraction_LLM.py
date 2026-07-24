"""
Extract structured invoice fields from PDF/image using EasyOCR + local Ollama LLM.

Flow:
  1. If input is an image -> convert to a temporary PDF
  2. Run EasyOCR (EOCR.extract_with_easyocr) on the PDF
  3. Send OCR text to a local Ollama model
  4. Parse and save structured JSON fields
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

# Avoid OpenMP duplicate-runtime crash common with EasyOCR/numpy on Windows
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import requests
from PIL import Image

from EOCR import extract_with_easyocr

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "llama3.1:latest"
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
PDF_EXTENSIONS = {".pdf"}

# If True: save intermediate OCR text (.txt) and extracted fields (.json).
# If False: do not write output files (results are still returned/printed).
SAVE_INTERMEDIATE = False

FIELDS_SCHEMA = {
    "invoice_number": "Invoice / bill / tax invoice number",
    "invoice_date": "Invoice date (keep original format if possible)",
    "seller_name": "Vendor / supplier / from company name",
    "seller_gstin": "Seller GSTIN / GST number",
    "buyer_name": "Customer / bill-to / ship-to company name",
    "buyer_gstin": "Buyer GSTIN / GST number",
    "po_number": "Purchase order / PO number if present",
    "subtotal": "Taxable / subtotal amount before tax",
    "cgst": "CGST amount if present",
    "sgst": "SGST amount if present",
    "igst": "IGST amount if present",
    "total_tax": "Total tax amount if present",
    "total_amount": "Grand total / amount payable",
    "currency": "Currency symbol or code if present (e.g. INR, Rs)",
    "payment_terms": "Payment terms if present",
    "due_date": "Due date if present",
}


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def resolve_input_path(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return path


def image_to_pdf(image_path: Path, output_pdf: Path) -> Path:
    """Convert a single image to a 1-page PDF (EasyOCR expects PDF)."""
    with Image.open(image_path) as img:
        # PDF encoder expects RGB
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")
        img.save(output_pdf, "PDF", resolution=300.0)
    return output_pdf


def prepare_pdf(input_path: Path) -> tuple[Path, Path | None]:
    """
    Return (pdf_path, temp_dir_or_None).

    Images are converted to a temporary PDF that the caller must delete.
    PDFs are used as-is (temp_dir is None).
    """
    suffix = input_path.suffix.lower()

    if suffix in PDF_EXTENSIONS:
        return input_path, None

    if suffix in IMAGE_EXTENSIONS:
        temp_dir = Path(tempfile.mkdtemp(prefix="element_llm_"))
        pdf_path = temp_dir / f"{input_path.stem}.pdf"
        print(f"Converting image -> temporary PDF:\n  {pdf_path}")
        image_to_pdf(input_path, pdf_path)
        return pdf_path, temp_dir

    raise ValueError(
        f"Unsupported file type '{suffix}'. "
        f"Use PDF or image ({', '.join(sorted(IMAGE_EXTENSIONS))})."
    )


def cleanup_temp_pdf(temp_dir: Path | None) -> None:
    """Remove temporary directory created for image->PDF conversion."""
    if not temp_dir:
        return
    try:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"Removed temporary PDF folder: {temp_dir}")
    except OSError as exc:
        print(f"Warning: could not remove temp folder {temp_dir}: {exc}")


def parse_page_numbers(page_arg: str | None) -> list[int] | None:
    """Parse '1,3-5' into [1, 3, 4, 5]. None means all pages."""
    if not page_arg or not page_arg.strip():
        return None

    pages: set[int] = set()
    for part in page_arg.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if start > end:
                start, end = end, start
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part))
    return sorted(pages) or None


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

def list_ollama_models(base_url: str = OLLAMA_URL) -> list[str]:
    resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=10)
    resp.raise_for_status()
    return [m["name"] for m in resp.json().get("models", [])]


def build_extraction_prompt(ocr_text: str) -> str:
    fields_block = "\n".join(f'  "{k}": {v}' for k, v in FIELDS_SCHEMA.items())
    return f"""You are an expert at reading OCR text from invoices and extracting structured data.

            OCR text (may contain noise, broken lines, or OCR errors):
            \"\"\"
            {ocr_text}
            \"\"\"
            
            Extract these fields. Use null when a value is not found. Do not invent values.
            Return ONLY a valid JSON object with exactly these keys:
            {{
            {fields_block}
            }}
            
            Rules:
            - Prefer exact values as they appear in the text.
            - For amounts, keep digits and decimal points; strip currency symbols into "currency".
            - GSTIN is typically 15 characters (e.g. 22AAAAA0000A1Z5).
            - If multiple GSTINs appear, assign seller vs buyer using context (From/Supplier vs Bill To/Customer).
            - Output JSON only — no markdown, no explanation.
            """


def call_ollama(
    prompt: str,
    model: str = DEFAULT_MODEL,
    base_url: str = OLLAMA_URL,
    temperature: float = 0.1,
    ) -> str:
    """
    Call Ollama /api/generate and return the full model text.

    Uses non-streaming requests so the HTTP body is a single JSON object.
    (stream=True returns NDJSON lines; resp.json() then raises Extra data.)
    """
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": temperature,
        },
    }
    print(f"Querying Ollama model '{model}' ...")
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()

    try:
        data = resp.json()
    except json.JSONDecodeError:
        # Safety net if a proxy/server ever returns NDJSON anyway
        chunks: list[str] = []
        for line in resp.text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                part = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(part, dict) and part.get("response"):
                chunks.append(str(part["response"]))
        return "".join(chunks)

    return data.get("response", "") or ""


def _strip_code_fences(text: str) -> str:
    """Remove markdown ``` / ```json wrappers if present."""
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return text.strip()


def _strip_non_json_noise(text: str) -> str:
    """Remove common LLM chatter outside JSON (thinking tags, preamble)."""
    # Drop <think>...</think> / <thinking>...</thinking> blocks
    text = re.sub(
        r"<(?:think|thinking)>[\s\S]*?</(?:think|thinking)>",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip()


def _iter_json_candidates(text: str):
    """
    Yield successive JSON values from text using raw_decode.
    Skips junk before each value; stops when nothing more can be parsed.
    """
    decoder = json.JSONDecoder()
    idx = 0
    length = len(text)

    while idx < length:
        # Skip whitespace / non-JSON prefix until a value could start
        while idx < length and text[idx] not in "{[\"tfnTFN0123456789+-":
            idx += 1
        if idx >= length:
            break
        try:
            value, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            idx += 1
            continue
        yield value
        idx = end


def _brace_slice_first_object(text: str) -> str | None:
    """Fallback: extract the first balanced {...} substring."""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_json_from_response(raw: str) -> dict[str, Any]:
    """
    Parse a JSON object from noisy LLM output.

    Strips markdown fences / thinking tags, ignores characters outside the
    JSON, and tolerates trailing extra data or multiple JSON blobs
    (first dict wins; later dicts are shallow-merged for missing keys).
    """
    if not raw or not str(raw).strip():
        raise ValueError("Empty response from LLM")

    text = _strip_non_json_noise(_strip_code_fences(str(raw)))

    dicts: list[dict[str, Any]] = []

    # Prefer streaming decode — correctly ignores trailing / interleaved junk
    for value in _iter_json_candidates(text):
        if isinstance(value, dict):
            dicts.append(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    dicts.append(item)

    # Brace-matching fallback if raw_decode found nothing
    if not dicts:
        sliced = _brace_slice_first_object(text)
        if sliced:
            try:
                parsed = json.loads(sliced)
                if isinstance(parsed, dict):
                    dicts.append(parsed)
            except json.JSONDecodeError:
                pass

    if not dicts:
        preview = text[:300].replace("\n", "\\n")
        raise ValueError(
            "Could not find a JSON object in LLM response. "
            f"Preview: {preview}"
        )

    # Merge later objects only to fill keys still missing/null
    merged: dict[str, Any] = dict(dicts[0])
    for extra in dicts[1:]:
        for key, value in extra.items():
            if key not in merged or merged[key] in (None, "", []):
                merged[key] = value
    return merged


def normalize_fields(data: dict[str, Any]) -> dict[str, Any]:
    """Ensure all expected keys exist; coerce empty strings to null."""
    out: dict[str, Any] = {}
    for key in FIELDS_SCHEMA:
        value = data.get(key, None)
        if isinstance(value, str):
            value = value.strip()
            if value.lower() in ("", "null", "none", "n/a", "na", "-"):
                value = None
        out[key] = value
    return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def extract_elements(
    input_path: str | Path,
    page_numbers: list[int] | None = None,
    model: str = DEFAULT_MODEL,
    ollama_url: str = OLLAMA_URL,
    save_intermediate: bool = SAVE_INTERMEDIATE,
    output_path: str | Path | None = None,
    ) -> dict[str, Any]:
    input_path = resolve_input_path(str(input_path))
    pdf_path, temp_dir = prepare_pdf(input_path)

    try:
        print(f"Running EasyOCR on: {pdf_path}")
        pages_label = (
            ",".join(str(p) for p in page_numbers) if page_numbers else "all"
        )
        print(f"Pages: {pages_label}")
        ocr_text = extract_with_easyocr(str(pdf_path), page_numbers)

        if not ocr_text or not ocr_text.strip():
            raise RuntimeError("OCR returned empty text. Check the input file.")

        print(f"OCR extracted {len(ocr_text)} characters.")

        prompt = build_extraction_prompt(ocr_text)
        raw_response = call_ollama(prompt, model=model, base_url=ollama_url)
        fields = normalize_fields(extract_json_from_response(raw_response))

        result: dict[str, Any] = {
            "source_file": str(input_path),
            "model": model,
            "pages": page_numbers,
            "fields": fields,
        }

        if save_intermediate:
            result["ocr_text"] = ocr_text

            if output_path is None:
                json_path = input_path.with_name(f"{input_path.stem}_elements.json")
            else:
                json_path = Path(output_path)

            text_path = input_path.with_name(f"{input_path.stem}_ocr.txt")

            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(ocr_text)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            result["ocr_text_file"] = str(text_path)
            result["output_file"] = str(json_path)
            print(f"Saved OCR text: {text_path}")
            print(f"Saved JSON:     {json_path}")
        else:
            result["ocr_text_file"] = None
            result["output_file"] = None

        return result
    finally:
        # Always delete temp PDF created from an image input
        cleanup_temp_pdf(temp_dir)


def print_fields(fields: dict[str, Any]) -> None:
    print("\n=== Extracted Elements ===")
    width = max(len(k) for k in fields) if fields else 10
    for key, value in fields.items():
        display = value if value is not None else "(not found)"
        print(f"  {key.ljust(width)} : {display}")
    print("=" * 28)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract invoice elements from PDF/image via EasyOCR + Ollama."
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Path to PDF or image file",
    )
    parser.add_argument(
        "-p",
        "--pages",
        default=None,
        help="Pages to OCR, e.g. '1' or '1,3-5' (default: all)",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--ollama-url",
        default=OLLAMA_URL,
        help=f"Ollama base URL (default: {OLLAMA_URL})",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output JSON path (default: <input>_elements.json)",
    )
    parser.add_argument(
        "--save-intermediate",
        action=argparse.BooleanOptionalAction,
        default=SAVE_INTERMEDIATE,
        help=(
            "If true, save intermediate OCR text (.txt) and extracted JSON. "
            f"(default: {SAVE_INTERMEDIATE})"
        ),
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available Ollama models and exit",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.list_models:
        try:
            models = list_ollama_models(args.ollama_url)
        except requests.RequestException as exc:
            print(f"Could not reach Ollama at {args.ollama_url}: {exc}", file=sys.stderr)
            return 1
        if not models:
            print("No models found. Pull one with: ollama pull llama3.1")
            return 0
        print("Available Ollama models:")
        for name in models:
            print(f"  - {name}")
        return 0

    # input_path = Path("current_test/CH-012026-0559-INVOICE.pdf")
    input_path = Path("current_test/CH-012026-0559-INVOICE.jpeg")
    try:
        page_numbers = parse_page_numbers(args.pages)
        result = extract_elements(
            input_path=input_path,
            page_numbers=page_numbers,
            model=args.model,
            ollama_url=args.ollama_url,
            save_intermediate=args.save_intermediate,
            output_path=args.output,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print_fields(result["fields"])
    if result.get("output_file"):
        print(f"\nSaved OCR text: {result['ocr_text_file']}")
        print(f"Saved JSON:     {result['output_file']}")
    else:
        print("\nNo files saved (SAVE_INTERMEDIATE / --save-intermediate is False).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
