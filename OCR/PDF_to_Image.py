"""
Convert selected PDF pages to images and save them in a folder named after the PDF.
"""

from pathlib import Path

import fitz

PDF_RENDER_DPI = 300


def get_pdf_page_count(pdf_path: Path) -> int:
    with fitz.open(pdf_path) as doc:
        return len(doc)


def parse_page_selection(selection: str, total_pages: int) -> list[int]:
    """Parse page input like 'all', '1', '1-3', '1,3,5-7' into 0-based page indexes."""
    selection = selection.strip().lower()
    if selection in {"", "all"}:
        return list(range(total_pages))

    selected_pages: set[int] = set()
    for chunk in selection.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue

        if "-" in chunk:
            start_str, end_str = chunk.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if start > end:
                raise ValueError(f"Invalid page range: {chunk}")
            if start < 1 or end > total_pages:
                raise ValueError(f"Page range out of bounds: {chunk}")
            selected_pages.update(range(start - 1, end))
            continue

        page_num = int(chunk)
        if page_num < 1 or page_num > total_pages:
            raise ValueError(f"Page out of bounds: {page_num}")
        selected_pages.add(page_num - 1)

    if not selected_pages:
        raise ValueError("No valid pages selected.")

    return sorted(selected_pages)


def prompt_for_pdf_path() -> Path:
    while True:
        raw = input("Enter PDF path: ").strip().strip('"').strip("'")
        if not raw:
            print("Please enter a PDF path.")
            continue

        pdf_path = Path(raw)
        if not pdf_path.exists():
            print(f"File not found: {pdf_path.resolve()}")
            continue
        if pdf_path.suffix.lower() != ".pdf":
            print("Please provide a .pdf file.")
            continue
        return pdf_path


def prompt_for_pages(total_pages: int) -> list[int]:
    print(f"Total pages in PDF: {total_pages}")
    print("Enter pages to convert: all, 1, 1-3, 1,3,5-7")

    while True:
        selection = input("Pages to convert [all]: ")
        try:
            return parse_page_selection(selection, total_pages)
        except ValueError as exc:
            print(f"Invalid selection: {exc}")


def create_output_folder(pdf_path: Path) -> Path:
    output_dir = pdf_path.parent / pdf_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def render_pdf_pages(pdf_path: Path, page_numbers: list[int], output_dir: Path) -> list[Path]:
    image_paths: list[Path] = []
    scale = PDF_RENDER_DPI / 72.0
    matrix = fitz.Matrix(scale, scale)

    with fitz.open(pdf_path) as doc:
        for page_index in page_numbers:
            page = doc.load_page(page_index)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image_path = output_dir / f"{pdf_path.stem}_page_{page_index + 1}.png"
            pixmap.save(str(image_path))
            image_paths.append(image_path)
            print(f"Saved: {image_path.resolve()}")

    return image_paths


def main() -> None:
    pdf_path = prompt_for_pdf_path()
    total_pages = get_pdf_page_count(pdf_path)
    if total_pages == 0:
        raise SystemExit("PDF has no pages.")

    page_numbers = prompt_for_pages(total_pages)
    output_dir = create_output_folder(pdf_path)

    print(f"\nConverting {len(page_numbers)} page(s) to images...")
    print(f"Output folder: {output_dir.resolve()}")

    image_paths = render_pdf_pages(pdf_path, page_numbers, output_dir)

    print(f"\nDone. Saved {len(image_paths)} image(s) in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
