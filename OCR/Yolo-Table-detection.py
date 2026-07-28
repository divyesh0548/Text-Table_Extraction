import tempfile
from functools import partial
from pathlib import Path

import cv2
import fitz
import numpy as np
import pandas as pd
import pytesseract
import torch
import torch.serialization
import ultralytics
from PIL import Image, ImageDraw
from pytesseract import Output
from ultralyticsplus import YOLO, render_result

from env_config import get_input_pdf

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

# PDF path comes from OCR/.env (INPUT_PDF). YOLO converts PDF pages to images first.
INPUT_SOURCE = get_input_pdf()
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
PDF_RENDER_DPI = 300


def collapse_positions(positions, gap=10):
    if not positions:
        return []

    positions = sorted(int(pos) for pos in positions)
    collapsed = [[positions[0]]]

    for pos in positions[1:]:
        if pos - collapsed[-1][-1] <= gap:
            collapsed[-1].append(pos)
        else:
            collapsed.append([pos])

    return [int(sum(group) / len(group)) for group in collapsed]


def detect_table_grid(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, image_array.shape[0] // 25)))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, image_array.shape[1] // 25), 1))

    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    vertical_strength = vertical_lines.sum(axis=0) / 255
    horizontal_strength = horizontal_lines.sum(axis=1) / 255

    vertical_threshold = max(20, image_array.shape[0] * 0.25)
    horizontal_threshold = max(20, image_array.shape[1] * 0.25)

    vertical_positions = np.where(vertical_strength > vertical_threshold)[0].tolist()
    horizontal_positions = np.where(horizontal_strength > horizontal_threshold)[0].tolist()

    columns = collapse_positions(vertical_positions, gap=12)
    rows = collapse_positions(horizontal_positions, gap=12)

    if not columns or columns[0] > 5:
        columns = [0] + columns
    if columns[-1] < image_array.shape[1] - 5:
        columns.append(image_array.shape[1] - 1)

    if not rows or rows[0] > 5:
        rows = [0] + rows
    if rows[-1] < image_array.shape[0] - 5:
        rows.append(image_array.shape[0] - 1)

    return columns, rows


def clean_ocr_text(value):
    text = str(value).strip()
    text = text.strip("|_ ")
    return text


def build_structured_table(ocr_df, image_array):
    words = ocr_df.copy()
    words["text"] = words["text"].apply(clean_ocr_text)
    words = words[(words["text"] != "") & (words["conf"].astype(float) > -1)]

    if words.empty:
        return pd.DataFrame()

    columns, rows = detect_table_grid(image_array)
    if len(columns) < 2 or len(rows) < 2:
        return pd.DataFrame()

    cell_map = {}

    for _, word in words.iterrows():
        x_center = float(word["left"]) + float(word["width"]) / 2
        y_center = float(word["top"]) + float(word["height"]) / 2

        col_idx = next((idx for idx in range(len(columns) - 1) if columns[idx] <= x_center < columns[idx + 1]), None)
        row_idx = next((idx for idx in range(len(rows) - 1) if rows[idx] <= y_center < rows[idx + 1]), None)

        if row_idx is None or col_idx is None:
            continue

        key = (row_idx, col_idx)
        cell_map.setdefault(key, []).append((float(word["left"]), word["text"]))

    table_rows = []
    for row_idx in range(len(rows) - 1):
        row_values = []
        has_content = False
        for col_idx in range(len(columns) - 1):
            entries = cell_map.get((row_idx, col_idx), [])
            if entries:
                entries.sort(key=lambda item: item[0])
                cell_text = " ".join(text for _, text in entries).strip()
                has_content = has_content or bool(cell_text)
            else:
                cell_text = ""
            row_values.append(cell_text)

        if has_content:
            table_rows.append(row_values)

    if not table_rows:
        return pd.DataFrame()

    max_cols = max(len(row) for row in table_rows)
    normalized_rows = [row + [""] * (max_cols - len(row)) for row in table_rows]

    header = normalized_rows[0]
    data_rows = normalized_rows[1:] if len(normalized_rows) > 1 else []

    if any(header):
        structured_df = pd.DataFrame(data_rows, columns=header)
    else:
        structured_df = pd.DataFrame(normalized_rows)

    return structured_df


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


def prompt_for_pdf_conversion(pdf_path: Path) -> list[int]:
    total_pages = get_pdf_page_count(pdf_path)
    print(f"Detected PDF input: {pdf_path}")
    print(f"Total pages: {total_pages}")
    print("This PDF will be converted into page images (with page numbers) before table detection.")

    while True:
        confirm = input("Convert PDF pages to images and continue? [y/n]: ").strip().lower()
        if confirm in {"y", "yes"}:
            break
        if confirm in {"n", "no"}:
            raise SystemExit("PDF conversion cancelled.")
        print("Please enter y or n.")

    print("Enter pages to convert: all, 1, 1-3, 1,3,5-7")
    while True:
        selection = input("Pages to convert [all]: ")
        try:
            return parse_page_selection(selection, total_pages)
        except ValueError as exc:
            print(f"Invalid selection: {exc}")


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
            print(f"Saved page image: {image_path.resolve()}")

    return image_paths


def prepare_input_images(input_path: Path) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path.resolve()}")

    suffix = input_path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return [input_path]

    if suffix != ".pdf":
        raise ValueError(
            f"Unsupported input format: {suffix}. Use PDF or image "
            f"({', '.join(sorted(IMAGE_EXTENSIONS))})."
        )

    selected_pages = prompt_for_pdf_conversion(input_path)
    output_dir = Path(tempfile.mkdtemp(prefix="yolo_pdf_pages_", dir="."))
    print(f"Converting selected pages to images in: {output_dir.resolve()}")
    return render_pdf_pages(input_path, selected_pages, output_dir)


def load_table_model() -> YOLO:
    # PyTorch 2.6 changed torch.load(..., weights_only=True) by default.
    # This checkpoint requires full deserialization, so force the old behavior
    # for this trusted model load.
    torch.load = partial(torch.load, weights_only=False)
    with torch.serialization.safe_globals([ultralytics.nn.tasks.DetectionModel]):
        model = YOLO("keremberke/yolov8m-table-extraction")

    # Keep enough detections for multi-table pages; avoid over-aggressive NMS.
    model.overrides["conf"] = 0.20
    model.overrides["iou"] = 0.50
    model.overrides["agnostic_nms"] = False
    model.overrides["max_det"] = 100
    return model


def _box_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def collect_table_boxes(result, iou_dedupe: float = 0.70) -> list[tuple[int, int, int, int, float]]:
    """
    Return all table boxes for a page: (x1, y1, x2, y2, confidence),
    sorted top-to-bottom then left-to-right, with near-duplicates removed.
    """
    boxes_tensor = result.boxes.data
    if boxes_tensor is None or len(boxes_tensor) == 0:
        return []

    raw_boxes: list[tuple[int, int, int, int, float]] = []
    for row in boxes_tensor.cpu().numpy():
        x1, y1, x2, y2, conf = (float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]))
        if x2 <= x1 or y2 <= y1:
            continue
        raw_boxes.append((int(x1), int(y1), int(x2), int(y2), conf))

    # Keep highest-confidence box when two detections heavily overlap.
    raw_boxes.sort(key=lambda item: item[4], reverse=True)
    kept: list[tuple[int, int, int, int, float]] = []
    for box in raw_boxes:
        if any(_box_iou(box[:4], existing[:4]) >= iou_dedupe for existing in kept):
            continue
        kept.append(box)

    # Reading order for stable sheet naming / processing.
    kept.sort(key=lambda item: (item[1], item[0]))
    return kept


def process_image(image: Path, model: YOLO) -> None:
    excel_output = image.with_name(f"{image.stem}_ocr_output.xlsx")
    detection_output = image.with_name(f"{image.stem}_table_detection.png")

    img = Image.open(image).convert("RGB")
    results = model.predict(img)
    table_boxes = collect_table_boxes(results[0])
    if not table_boxes:
        print(f"No table detections found in: {image}")
        return

    print(f"Detected {len(table_boxes)} table(s) in: {image.name}")
    print("Boxes: ", results[0].boxes)
    render_result(model=model, image=img, result=results[0])

    annotated_image = img.copy()
    draw = ImageDraw.Draw(annotated_image)
    for table_index, (x1, y1, x2, y2, conf) in enumerate(table_boxes, start=1):
        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
        draw.text((x1 + 4, max(0, y1 - 14)), f"T{table_index} {conf:.2f}", fill="red")
    annotated_image.save(detection_output)

    image_array = np.array(img)
    extracted_count = 0

    with pd.ExcelWriter(excel_output, engine="openpyxl") as writer:
        for table_index, (x1, y1, x2, y2, conf) in enumerate(table_boxes, start=1):
            cropped_output = image.with_name(f"{image.stem}_table_{table_index}_crop.png")
            cropped_array = image_array[y1:y2, x1:x2]
            if cropped_array.size == 0:
                print(f"  Skipped table {table_index}: empty crop.")
                continue

            cropped_image = Image.fromarray(cropped_array)
            cropped_image.save(cropped_output)

            ext_df = pytesseract.image_to_data(
                cropped_image,
                output_type=Output.DATAFRAME,
                config="--psm 6 --oem 3",
            )
            ext_df = ext_df.fillna("")
            structured_df = build_structured_table(ext_df, np.array(cropped_image))

            sheet_structured = f"T{table_index}_structured"[:31]
            sheet_raw = f"T{table_index}_raw_ocr"[:31]
            if not structured_df.empty:
                structured_df.to_excel(writer, sheet_name=sheet_structured, index=False)
            ext_df.to_excel(writer, sheet_name=sheet_raw, index=False)

            extracted_count += 1
            print(
                f"  Table {table_index}/{len(table_boxes)} "
                f"(conf={conf:.2f}) -> {cropped_output.name}"
            )
            if structured_df.empty:
                print("    Structured reconstruction was not confident; raw OCR sheet saved.")
            else:
                print("    Structured table preview:")
                print(structured_df)

    print(f"Extracted {extracted_count} table(s) from {image.name}")
    print(f"Excel output saved to: {excel_output.resolve()}")
    print(f"Detection image saved to: {detection_output.resolve()}")


def main() -> None:
    image_paths = prepare_input_images(INPUT_SOURCE)
    model = load_table_model()

    for index, image_path in enumerate(image_paths, start=1):
        print(f"\n=== Processing image {index}/{len(image_paths)}: {image_path.name} ===")
        process_image(image_path, model)


if __name__ == "__main__":
    main()
