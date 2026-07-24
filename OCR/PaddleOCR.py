import shutil
import tempfile
from pathlib import Path

import cv2
import easyocr
import fitz
import numpy as np
import pandas as pd
from paddleocr import TableStructureRecognition

from table_structure_visualizer import build_cell_layout, parse_table_rows
from table_structure_visualizer import visualize_result

INPUT_SOURCE = "page_1_table_crop.png"
# INPUT_SOURCE = "CH-012026-0062-INVOICE.pdf"
OUTPUT_DIR = Path("./output")
PDF_RENDER_DPI = 300


def get_pdf_page_count(pdf_path: Path) -> int:
    with fitz.open(pdf_path) as doc:
        return len(doc)


def parse_page_selection(selection: str, total_pages: int) -> list[int]:
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


def prompt_for_pdf_pages(pdf_path: Path) -> list[int]:
    total_pages = get_pdf_page_count(pdf_path)
    print(f"Detected PDF input: {pdf_path}")
    print(f"Total pages: {total_pages}")
    print("Enter pages to process: all, 1, 1-3, 1,3,5-7")

    while True:
        selection = input("Pages to process [all]: ")
        try:
            return parse_page_selection(selection, total_pages)
        except ValueError as exc:
            print(f"Invalid selection: {exc}")


def render_pdf_pages(pdf_path: Path, page_numbers: list[int], temp_dir: Path) -> list[Path]:
    image_paths: list[Path] = []
    scale = PDF_RENDER_DPI / 72.0
    matrix = fitz.Matrix(scale, scale)

    with fitz.open(pdf_path) as doc:
        for page_index in page_numbers:
            page = doc.load_page(page_index)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image_path = temp_dir / f"{pdf_path.stem}_page_{page_index + 1}.png"
            pixmap.save(str(image_path))
            image_paths.append(image_path)

    return image_paths


def prepare_inputs(input_path: Path) -> tuple[list[Path], Path | None]:
    if input_path.suffix.lower() != ".pdf":
        return [input_path], None

    selected_pages = prompt_for_pdf_pages(input_path)
    temp_dir = Path(tempfile.mkdtemp(prefix="paddleocr_pdf_", dir="."))
    image_paths = render_pdf_pages(input_path, selected_pages, temp_dir)
    return image_paths, temp_dir


def polygon_to_crop(image: np.ndarray, bbox: list[float | int]) -> np.ndarray:
    points = np.array(bbox, dtype=np.float32).reshape(-1, 2)
    x, y, w, h = cv2.boundingRect(points.astype(np.int32))
    return image[y:y + h, x:x + w]


def read_cell_text(reader: easyocr.Reader, cell_image: np.ndarray) -> str:
    if cell_image.size == 0:
        return ""

    results = reader.readtext(cell_image, detail=0, paragraph=True)
    cleaned = [text.strip() for text in results if str(text).strip()]
    return " ".join(cleaned)


def extract_table_dataframe(result: dict, image_path: Path, reader: easyocr.Reader) -> pd.DataFrame:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    bboxes = result.get("bbox", [])
    structure_tokens = result.get("structure", [])
    rows = parse_table_rows(structure_tokens)
    cells = build_cell_layout(rows)

    if not cells or not bboxes:
        return pd.DataFrame()

    max_row = max(cell["row"] + cell["rowspan"] for cell in cells)
    max_col = max(cell["col"] + cell["colspan"] for cell in cells)
    grid = [["" for _ in range(max_col)] for _ in range(max_row)]

    for idx, cell in enumerate(cells):
        if idx >= len(bboxes):
            break

        cropped = polygon_to_crop(image, bboxes[idx])
        text = read_cell_text(reader, cropped)
        row = cell["row"]
        col = cell["col"]
        grid[row][col] = text

    normalized_rows = [row for row in grid if any(str(value).strip() for value in row)]
    if not normalized_rows:
        return pd.DataFrame()

    header = normalized_rows[0]
    data_rows = normalized_rows[1:] if len(normalized_rows) > 1 else []
    if any(str(value).strip() for value in header):
        return pd.DataFrame(data_rows, columns=header)
    return pd.DataFrame(normalized_rows)


def save_tables_to_excel(tables: list[tuple[str, pd.DataFrame]], output_path: Path) -> None:
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        if not tables:
            pd.DataFrame([{"message": "No tables extracted"}]).to_excel(
                writer, sheet_name="summary", index=False
            )
            return

        for sheet_name, dataframe in tables:
            safe_sheet_name = sheet_name[:31] or "table"
            dataframe.to_excel(writer, sheet_name=safe_sheet_name, index=False)


def process_image(
    model: TableStructureRecognition,
    image_path: Path,
    reader: easyocr.Reader,
    ) -> pd.DataFrame:
    print(f"\nProcessing: {image_path}")
    output = model.predict(input=str(image_path), batch_size=1)
    extracted_df = pd.DataFrame()

    for res in output:
        res.print(json_format=False)
        result_dict = dict(res)

        json_path = OUTPUT_DIR / f"{image_path.stem}_res.json"
        res.save_to_json(str(json_path))

        # Built-in PaddleX overlay (red polygons on image)
        res.save_to_img(str(OUTPUT_DIR))

        # Enhanced visualizations: indexed cells, structure grid, combined view, HTML
        vis_paths = visualize_result(
            result=result_dict,
            output_dir=OUTPUT_DIR,
            stem=image_path.stem,
            image_path=str(image_path),
        )

        extracted_df = extract_table_dataframe(result_dict, image_path, reader)
        excel_path = OUTPUT_DIR / f"{image_path.stem}_easyocr.xlsx"
        save_tables_to_excel([(f"{image_path.stem}_table", extracted_df)], excel_path)
        print(f"  excel: {excel_path}")

        print("\nVisualization files:")
        for name, path in vis_paths.items():
            print(f"  {name}: {path}")

    return extracted_df


def cleanup_temp_images(temp_dir: Path | None) -> None:
    if temp_dir is None:
        return

    shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    input_path = Path(INPUT_SOURCE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    model = TableStructureRecognition(model_name="SLANet")
    reader = easyocr.Reader(['en'], gpu=False)
    temp_dir: Path | None = None
    extracted_tables: list[tuple[str, pd.DataFrame]] = []

    try:
        image_paths, temp_dir = prepare_inputs(input_path)
        for image_path in image_paths:
            extracted_df = process_image(model, image_path, reader)
            if not extracted_df.empty:
                extracted_tables.append((image_path.stem[:31], extracted_df))

        combined_excel_path = OUTPUT_DIR / f"{input_path.stem}_easyocr_tables.xlsx"
        save_tables_to_excel(extracted_tables, combined_excel_path)
        print(f"\nCombined Excel output: {combined_excel_path}")
    finally:
        cleanup_temp_images(temp_dir)


if __name__ == "__main__":
    main()
