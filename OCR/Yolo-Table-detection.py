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

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

# INPUT_SOURCE = Path("./page_1.png")
INPUT_SOURCE = Path("Current-Test/MBP 1 AKB 2026.pdf")
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

    model.overrides["conf"] = 0.25
    model.overrides["iou"] = 0.45
    model.overrides["agnostic_nms"] = False
    model.overrides["max_det"] = 1000
    return model


def process_image(image: Path, model: YOLO) -> None:
    excel_output = image.with_name(f"{image.stem}_ocr_output.xlsx")
    detection_output = image.with_name(f"{image.stem}_table_detection.png")
    cropped_output = image.with_name(f"{image.stem}_table_crop.png")

    img = Image.open(image).convert("RGB")
    results = model.predict(img)
    if len(results[0].boxes.data) == 0:
        print(f"No table detections found in: {image}")
        return

    print("Boxes: ", results[0].boxes)
    render_result(model=model, image=img, result=results[0])

    x1, y1, x2, y2, _, _ = tuple(int(item) for item in results[0].boxes.data.numpy()[0])
    annotated_image = img.copy()
    draw = ImageDraw.Draw(annotated_image)
    draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
    annotated_image.save(detection_output)

    image_array = np.array(img)
    cropped_image = image_array[y1:y2, x1:x2]
    cropped_image = Image.fromarray(cropped_image)
    cropped_image.save(cropped_output)

    ext_df = pytesseract.image_to_data(cropped_image, output_type=Output.DATAFRAME, config="--psm 6 --oem 3")
    ext_df = ext_df.fillna("")
    structured_df = build_structured_table(ext_df, np.array(cropped_image))

    with pd.ExcelWriter(excel_output, engine="openpyxl") as writer:
        if not structured_df.empty:
            structured_df.to_excel(writer, sheet_name="structured_table", index=False)
        ext_df.to_excel(writer, sheet_name="raw_ocr", index=False)

    print(ext_df)
    if structured_df.empty:
        print("Structured table reconstruction was not confident; raw OCR sheet was still saved.")
    else:
        print("\nStructured table preview:")
        print(structured_df)
    print(f"Excel output saved to: {excel_output.resolve()}")
    print(f"Detection image saved to: {detection_output.resolve()}")
    print(f"Cropped table image saved to: {cropped_output.resolve()}")


def main() -> None:
    image_paths = prepare_input_images(INPUT_SOURCE)
    model = load_table_model()

    for index, image_path in enumerate(image_paths, start=1):
        print(f"\n=== Processing image {index}/{len(image_paths)}: {image_path.name} ===")
        process_image(image_path, model)


if __name__ == "__main__":
    main()
