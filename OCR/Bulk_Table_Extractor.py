#!/usr/bin/env python3
"""
Bulk-run table extraction on every PDF in a folder.

For each PDF:
1. Enhance table lines with enhance_pdf_table_lines_safe.py
2. Detect/extract tables with Microsoft_Table_Transformer_ORIGINAL.py
3. Write Excel next to the source PDF: report.pdf -> report.xlsx
4. Optionally delete intermediate enhanced PDFs and debug images
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import Microsoft_Table_Transformer_ORIGINAL as table_extractor
import enhance_pdf_table_lines_safe as line_enhancer


# =============================================================================
# CONFIGURATION
# =============================================================================

input_folder = "C:/Divyesh/Table Extraction Project/OCR/Current-Test"
INPUT_FOLDER = Path(input_folder)

# If True, delete enhanced PDFs and per-PDF debug images after extraction.
DELETE_INTERMEDIATE_FILES = True

# Where enhanced PDFs and debug crops are written temporarily.
INTERMEDIATE_ROOT = Path(__file__).resolve().parent / "table_debug_bulk"
ENHANCED_PDF_SUFFIX = "_safe_table_lines"


def list_source_pdfs(folder: Path) -> list[Path]:
    """List original PDFs only (skip previously generated enhanced copies)."""
    pdfs = []
    for path in folder.iterdir():
        if not path.is_file() or path.suffix.lower() != ".pdf":
            continue
        stem_lower = path.stem.lower()
        if stem_lower.endswith(ENHANCED_PDF_SUFFIX.lower()):
            continue
        pdfs.append(path.resolve())

    return sorted(pdfs, key=lambda path: path.name.lower())


def clear_directory(directory: Path) -> None:
    if not directory.exists():
        return
    shutil.rmtree(directory, ignore_errors=True)


def delete_file(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
            print(f"Deleted intermediate PDF: {path}")
    except OSError as exc:
        print(f"Warning: could not delete {path}: {exc}", file=sys.stderr)


def enhance_pdf(source_pdf: Path, enhanced_pdf: Path) -> Path:
    """Create a line-enhanced PDF for more reliable table detection."""
    enhanced_pdf.parent.mkdir(parents=True, exist_ok=True)
    if enhanced_pdf.exists():
        enhanced_pdf.unlink()

    print(f"Enhancing table lines -> {enhanced_pdf.name}")
    line_enhancer.process_pdf(
        input_pdf=source_pdf,
        output_pdf=enhanced_pdf,
        dpi=line_enhancer.DPI,
        output_mode=line_enhancer.MODE,
        line_thickness=line_enhancer.LINE_THICKNESS,
        horizontal_min_fraction=line_enhancer.MIN_HORIZONTAL_FRACTION,
        vertical_min_fraction=line_enhancer.MIN_VERTICAL_FRACTION,
        angle_tolerance=line_enhancer.ANGLE_TOLERANCE,
        hough_threshold=line_enhancer.HOUGH_THRESHOLD,
        repair_gap=line_enhancer.REPAIR_GAP,
        coordinate_tolerance=line_enhancer.COORDINATE_TOLERANCE,
        intersection_tolerance=line_enhancer.INTERSECTION_TOLERANCE,
        minimum_support=line_enhancer.MINIMUM_SUPPORT,
        debug_dir=None,
    )
    return enhanced_pdf


def extract_one_pdf(
    pdf_path: Path,
    delete_intermediate_files: bool,
    ) -> Path:
    """
    Enhance one PDF, extract tables into <pdf_stem>.xlsx next to the source,
    then optionally remove intermediate files.
    """
    output_xlsx = pdf_path.with_suffix(".xlsx")
    work_dir = INTERMEDIATE_ROOT / pdf_path.stem
    clear_directory(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    enhanced_pdf = work_dir / f"{pdf_path.stem}{ENHANCED_PDF_SUFFIX}.pdf"
    debug_dir = work_dir / "table_debug"

    print()
    print("=" * 72)
    print(f"PDF: {pdf_path}")
    print(f"Excel: {output_xlsx}")
    print(f"Work dir: {work_dir}")
    print("=" * 72)

    try:
        enhance_pdf(pdf_path, enhanced_pdf)

        debug_dir.mkdir(parents=True, exist_ok=True)
        table_extractor.DEBUG_DIR = debug_dir

        print(f"Extracting tables from enhanced PDF: {enhanced_pdf.name}")
        table_extractor.extract_tables_from_pdf(
            input_pdf=enhanced_pdf,
            output_xlsx=output_xlsx,
            ocr_function=table_extractor.default_ocr,
        )
    finally:
        if delete_intermediate_files:
            delete_file(enhanced_pdf)
            clear_directory(work_dir)
            print(f"Deleted intermediate files: {work_dir}")

    return output_xlsx


def process_folder(
    folder: Path,
    delete_intermediate_files: bool = DELETE_INTERMEDIATE_FILES,
    ) -> list[Path]:
    folder = folder.expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        raise NotADirectoryError(f"Input folder is invalid: {folder}")

    pdfs = list_source_pdfs(folder)
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in: {folder}")

    INTERMEDIATE_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(pdfs)} PDF(s) in {folder}")
    print(f"Delete intermediate files: {delete_intermediate_files}")

    outputs: list[Path] = []
    failures: list[tuple[Path, str]] = []

    for index, pdf_path in enumerate(pdfs, start=1):
        print(f"\n[{index}/{len(pdfs)}] Processing {pdf_path.name}")
        try:
            outputs.append(
                extract_one_pdf(
                    pdf_path,
                    delete_intermediate_files=delete_intermediate_files,
                )
            )
        except Exception as exc:
            failures.append((pdf_path, str(exc)))
            print(f"FAILED: {pdf_path.name}: {exc}", file=sys.stderr)

    if delete_intermediate_files and INTERMEDIATE_ROOT.exists():
        try:
            if not any(INTERMEDIATE_ROOT.iterdir()):
                clear_directory(INTERMEDIATE_ROOT)
        except OSError:
            pass

    print()
    print(f"Completed: {len(outputs)} Excel file(s)")
    for output in outputs:
        print(f"  {output}")

    if failures:
        print(f"Failed: {len(failures)} PDF(s)", file=sys.stderr)
        for pdf_path, message in failures:
            print(f"  {pdf_path.name}: {message}", file=sys.stderr)

    return outputs


def main() -> int:
    folder = Path(INPUT_FOLDER).expanduser().resolve()

    try:
        process_folder(
            folder,
            delete_intermediate_files=DELETE_INTERMEDIATE_FILES,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
