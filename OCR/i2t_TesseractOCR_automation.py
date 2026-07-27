import os
import cv2
import fitz
import shutil
import numpy as np
from pathlib import Path
from typing import Optional

import polars as pl
from pdf2image import convert_from_path
from PIL import Image
import img2pdf
from img2table.ocr import TesseractOCR
from img2table.ocr.data import OCRDataframe
from img2table.document import PDF, Image as Img2TableImage
from img2table.tables.objects.cell import Cell
from img2table.tables.objects.table import Table
import pandas as pd

# OpenCV ximgproc compatibility patches
if not hasattr(cv2.ximgproc, "niBlackThreshold"):
    def niBlackThreshold(src, maxValue, *args, **kwargs):
        """
        Fallback to adaptiveThreshold when niBlackThreshold is unavailable.
        """
        return cv2.adaptiveThreshold(
            src,
            maxValue,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
    cv2.ximgproc.niBlackThreshold = niBlackThreshold

# Provide missing binarization constants
if not hasattr(cv2.ximgproc, "BINARIZATION_NIBLACK"):
    cv2.ximgproc.BINARIZATION_NIBLACK = 0

if not hasattr(cv2.ximgproc, "BINARIZATION_SAUVOLA"):
    cv2.ximgproc.BINARIZATION_SAUVOLA = 1


PDF_EXTENSIONS = {'.pdf'}
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def is_pdf_file(file_path):
    return Path(file_path).suffix.lower() in PDF_EXTENSIONS


def is_image_file(file_path):
    return Path(file_path).suffix.lower() in IMAGE_EXTENSIONS


def cleanup_temp_files():
    """Remove temporary files and directories"""
    temp_patterns = ['temp_pdf_images', 'extracted_tables']

    # Remove temp directories
    for pattern in temp_patterns:
        if os.path.exists(pattern):
            shutil.rmtree(pattern, ignore_errors=True)
            print(f"[CLEANUP] Removed {pattern} directory")

    # Remove temp files
    current_dir = os.getcwd()
    for filename in os.listdir(current_dir):
        if ('temp_page_' in filename and filename.endswith('.jpg')) or \
           ('temp_' in filename and filename.endswith('.png')) or \
           filename.endswith('_temp.pdf') or \
           (filename.startswith('page_') and filename.endswith('.csv')):
            try:
                os.remove(filename)
                print(f"[CLEANUP] Removed {filename}")
            except Exception as e:
                print(f"[WARN] Could not delete {filename}: {e}")


def get_pdf_page_count(pdf_path):
    """Get total number of pages in PDF"""
    try:
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        doc.close()
        return total_pages
    except Exception as e:
        print(f"Error getting page count: {e}")
        return 0


def parse_page_input(page_input, total_pages):
    """Parse page input like '1,3-5,7' into list of page numbers (0-indexed)"""
    pages = set()
    for part in page_input.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-', 1)
            try:
                s, e = int(start)-1, int(end)-1
                pages.update(range(max(s, 0), min(e, total_pages-1)+1))
            except ValueError:
                continue
        else:
            try:
                p = int(part)-1
                if 0 <= p < total_pages:
                    pages.add(p)
            except ValueError:
                continue
    return sorted(pages) if pages else None


def get_user_input():
    """Get all user inputs at the beginning"""
    print("=== Table Extraction Automation with TesseractOCR ===\n")

    # Get input path
    # input_path = input("Enter the path to your PDF or image file: ").strip()
    input_path = "Aarti - Mandays Apr 26.pdf"
    # input_path = "Current-Test/MBP 1 GG 2026.pdf"

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {Path(input_path).resolve()}")

    if not (is_pdf_file(input_path) or is_image_file(input_path)):
        raise ValueError("Unsupported input format. Use PDF, PNG, JPG, JPEG, BMP, TIF, or TIFF.")

    pages_to_process = None
    rotation_info = {}

    if is_pdf_file(input_path):
        total_pages = get_pdf_page_count(input_path)
        print(f"\nDetected {total_pages} pages in the PDF")

        page_input = input(f"Enter pages to process (1-{total_pages}, e.g., '1,3-5') or press Enter for all: ").strip()
        pages_to_process = parse_page_input(page_input, total_pages) if page_input else None

        rotate_input = "\nDo you want to rotate PDF pages? (y/n): "
    else:
        print("\nDetected image input")
        rotate_input = "\nDo you want to rotate the image? (y/n): "

    rotate_input_file = input(rotate_input).strip().lower() == 'y'

    if rotate_input_file:
        if is_pdf_file(input_path):
            rotation_pages_input = input("Enter pages to rotate (same format as above) or press Enter for all processed pages: ").strip()
            rotation_pages = parse_page_input(rotation_pages_input, total_pages) if rotation_pages_input else pages_to_process

            if rotation_pages:
                while True:
                    try:
                        rotation_angle = int(input("Enter rotation angle (90, 180, 270, -90): "))
                        if rotation_angle in [90, 180, 270, -90]:
                            break
                        else:
                            print("Please enter a valid rotation angle (90, 180, 270, -90)")
                    except ValueError:
                        print("Please enter a valid number")

                rotation_info = {'pages': rotation_pages, 'angle': rotation_angle}
        else:
            while True:
                try:
                    rotation_angle = int(input("Enter rotation angle (90, 180, 270, -90): "))
                    if rotation_angle in [90, 180, 270, -90]:
                        break
                    else:
                        print("Please enter a valid rotation angle (90, 180, 270, -90)")
                except ValueError:
                    print("Please enter a valid number")

            rotation_info = {'angle': rotation_angle}

    enhance_input = "\nDo you want to enhance the input before extraction? (y/n): "
    enhance_input_file = input(enhance_input).strip().lower() == 'y'

    return input_path, pages_to_process, rotation_info, enhance_input_file


def rotate_pdf_pages(pdf_path, rotation_info):
    """Rotate specific pages of PDF"""
    if not rotation_info:
        return pdf_path

    print(f"\n[ROTATE] Rotating pages {[p+1 for p in rotation_info['pages']]} by {rotation_info['angle']} degrees...")

    try:
        doc = fitz.open(pdf_path)

        for page_num in rotation_info['pages']:
            if 0 <= page_num < doc.page_count:
                page = doc[page_num]
                page.set_rotation(rotation_info['angle'])
                print(f"[ROTATE] Rotated page {page_num + 1}")

        base_name = os.path.splitext(pdf_path)[0]
        rotated_path = f"{base_name}_rotated.pdf"
        doc.save(rotated_path)
        doc.close()

        print(f"[ROTATE] Rotated PDF saved as: {rotated_path}")
        return rotated_path

    except Exception as e:
        print(f"[ERROR] Failed to rotate PDF: {e}")
        return pdf_path


def rotate_image_file(image_path, rotation_info):
    """Rotate image input"""
    if not rotation_info:
        return image_path

    print(f"\n[ROTATE] Rotating image by {rotation_info['angle']} degrees...")

    try:
        image = Image.open(image_path).convert("RGB")
        rotated_image = image.rotate(-rotation_info['angle'], expand=True)

        input_path = Path(image_path)
        rotated_path = input_path.with_name(f"{input_path.stem}_rotated{input_path.suffix}")
        rotated_image.save(rotated_path)

        print(f"[ROTATE] Rotated image saved as: {rotated_path}")
        return str(rotated_path)

    except Exception as e:
        print(f"[ERROR] Failed to rotate image: {e}")
        return image_path


def _enhance_table_page_bgr(img_bgr: np.ndarray, line_strength: int = 2) -> np.ndarray:
    """
    Strengthen table grid lines while preserving text for OCR.

    Previous logic used strong contrast/blur and masked line regions in a way
    that erased or damaged glyphs near borders, which made Tesseract struggle.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # OCR-friendly text base: light edge-preserving denoise + local contrast
    text_base = cv2.bilateralFilter(gray, d=5, sigmaColor=45, sigmaSpace=45)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    text_base = clahe.apply(text_base)

    # Detect lines from a separate binary image (do not reuse text_base as output mask)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        10,
    )

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

    table_mask = cv2.add(detect_horizontal, detect_vertical)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(1, line_strength), max(1, line_strength))
    )
    # One dilate pass: thicken lines for table detection without eating nearby text
    table_mask = cv2.dilate(table_mask, kernel, iterations=1)
    table_mask = cv2.morphologyEx(table_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Draw black grid lines on top of the preserved text image
    enhanced = text_base.copy()
    enhanced[table_mask > 0] = 0
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def enhance_pdf_tables(pdf_path, dpi=300, line_strength=2, show_progress=True):
    """Enhance PDF table grid lines without destroying text for OCR."""
    if show_progress:
        print(f"\n[ENHANCE] Starting PDF enhancement...")

    temp_dir = "temp_pdf_images"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        pages = convert_from_path(pdf_path, dpi=dpi)
        enhanced_images = []

        for i, page in enumerate(pages):
            if show_progress:
                print(f"[ENHANCE] Processing page {i+1}/{len(pages)}...")

            img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            enhanced_bgr = _enhance_table_page_bgr(img, line_strength=line_strength)

            enhanced_path = os.path.join(temp_dir, f"enhanced_page_{i+1}.png")
            cv2.imwrite(enhanced_path, enhanced_bgr)
            with Image.open(enhanced_path) as enhanced_img:
                enhanced_images.append(enhanced_img.convert("RGB"))

        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        pdf_output_path = os.path.join(os.getcwd(), f"{base_name}_enhanced.pdf")
        enhanced_images[0].save(pdf_output_path, save_all=True, append_images=enhanced_images[1:])

        if show_progress:
            print(f"[ENHANCE] Enhanced PDF saved as: {pdf_output_path}")

        return pdf_output_path

    except Exception as e:
        print(f"[ERROR] Enhancement failed: {e}")
        return pdf_path


def enhance_image_tables(image_path, line_strength=2, show_progress=True):
    """Enhance an image table grid while preserving text for OCR."""
    if show_progress:
        print(f"\n[ENHANCE] Starting image enhancement...")

    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        enhanced_bgr = _enhance_table_page_bgr(img, line_strength=line_strength)

        input_path = Path(image_path)
        output_path = input_path.with_name(f"{input_path.stem}_enhanced{input_path.suffix}")
        cv2.imwrite(str(output_path), enhanced_bgr)

        if show_progress:
            print(f"[ENHANCE] Enhanced image saved as: {output_path}")

        return str(output_path)

    except Exception as e:
        print(f"[ERROR] Enhancement failed: {e}")
        return image_path


def _cluster_words_into_lines(words: list[dict], y_overlap_ratio: float = 0.5) -> list[list[dict]]:
    """
    Group words into visual reading lines using vertical overlap, then sort
    each line left-to-right. Fixes wrong word order caused by large whitespace
    (PDF content-stream order is not always left-to-right).
    """
    if not words:
        return []

    words = sorted(words, key=lambda w: (((w["y1"] + w["y2"]) / 2), w["x1"]))
    lines: list[list[dict]] = []

    for word in words:
        y_center = (word["y1"] + word["y2"]) / 2
        word_height = max(1, word["y2"] - word["y1"])
        placed = False

        for line in lines:
            line_y_center = sum((w["y1"] + w["y2"]) / 2 for w in line) / len(line)
            line_height = max(1, sum(w["y2"] - w["y1"] for w in line) / len(line))
            threshold = y_overlap_ratio * min(word_height, line_height)
            if abs(y_center - line_y_center) <= threshold:
                line.append(word)
                placed = True
                break

        if not placed:
            lines.append([word])

    lines.sort(key=lambda line: min(w["y1"] for w in line))
    for line in lines:
        line.sort(key=lambda w: w["x1"])
    return lines


def _words_to_reading_order_text(words: list[dict]) -> Optional[str]:
    lines = _cluster_words_into_lines(words)
    if not lines:
        return None
    text = "\n".join(" ".join(str(w["value"]).strip() for w in line if str(w.get("value", "")).strip())
                     for line in lines).strip()
    return text or None


def _patch_img2table_cell_word_order() -> None:
    """
    Patch img2table OCR text assembly only.
    Table detection / grid structure is unchanged.
    """

    def get_text_cell(self, cell: Cell, margin: int = 0, page_number: Optional[int] = None,
                      min_confidence: int = 50) -> str:
        bbox = cell.bbox(margin=margin)
        df_words = self.df.filter(pl.col("class") == "ocrx_word")
        if page_number:
            df_words = df_words.filter(pl.col("page") == page_number)
        df_words = df_words.filter(pl.col("value").is_not_null() & (pl.col("confidence") >= min_confidence))

        df_words = (
            df_words.with_columns([
                pl.lit(bbox[0]).alias("x1_bbox"),
                pl.lit(bbox[1]).alias("y1_bbox"),
                pl.lit(bbox[2]).alias("x2_bbox"),
                pl.lit(bbox[3]).alias("y2_bbox"),
            ])
            .with_columns([
                pl.max_horizontal(["x1", "x1_bbox"]).alias("x_left"),
                pl.max_horizontal(["y1", "y1_bbox"]).alias("y_top"),
                pl.min_horizontal(["x2", "x2_bbox"]).alias("x_right"),
                pl.min_horizontal(["y2", "y2_bbox"]).alias("y_bottom"),
            ])
        )

        df_intersection = (
            df_words.filter(pl.col("x_right") > pl.col("x_left"))
            .filter(pl.col("y_bottom") > pl.col("y_top"))
        )
        df_areas = df_intersection.with_columns([
            ((pl.col("x2") - pl.col("x1")) * (pl.col("y2") - pl.col("y1"))).alias("w_area"),
            ((pl.col("x_right") - pl.col("x_left")) * (pl.col("y_bottom") - pl.col("y_top"))).alias("int_area"),
        ])
        df_words_contained = df_areas.filter(pl.col("int_area") / pl.col("w_area") > 0.5)

        words = df_words_contained.select(["value", "x1", "y1", "x2", "y2"]).to_dicts()
        return _words_to_reading_order_text(words)

    def get_text_table(self, table: Table, page_number: Optional[int] = None, min_confidence: int = 50) -> Table:
        df_words = self.df.filter(pl.col("class") == "ocrx_word")
        if page_number:
            df_words = df_words.filter(pl.col("page") == page_number)
        df_words = df_words.filter(pl.col("value").is_not_null() & (pl.col("confidence") >= min_confidence))

        list_cells = [
            {"row": id_row, "col": id_col, "x1_w": cell.x1, "x2_w": cell.x2, "y1_w": cell.y1, "y2_w": cell.y2}
            for id_row, row in enumerate(table.items)
            for id_col, cell in enumerate(row.items)
        ]
        df_cells = pl.DataFrame(data=list_cells)
        df_word_cells = df_words.join(other=df_cells, how="cross")

        df_word_cells = df_word_cells.with_columns([
            pl.max_horizontal(["x1", "x1_w"]).alias("x_left"),
            pl.max_horizontal(["y1", "y1_w"]).alias("y_top"),
            pl.min_horizontal(["x2", "x2_w"]).alias("x_right"),
            pl.min_horizontal(["y2", "y2_w"]).alias("y_bottom"),
        ])

        df_intersection = (
            df_word_cells.filter(pl.col("x_right") > pl.col("x_left"))
            .filter(pl.col("y_bottom") > pl.col("y_top"))
        )
        df_areas = df_intersection.with_columns([
            ((pl.col("x2") - pl.col("x1")) * (pl.col("y2") - pl.col("y1"))).alias("w_area"),
            ((pl.col("x_right") - pl.col("x_left")) * (pl.col("y_bottom") - pl.col("y_top"))).alias("int_area"),
        ])
        df_words_contained = df_areas.filter(pl.col("int_area") / pl.col("w_area") > 0.5)

        if df_words_contained.height == 0:
            return table

        for key, group in df_words_contained.group_by(["row", "col"]):
            row_idx, col_idx = key
            words = group.select(["value", "x1", "y1", "x2", "y2"]).to_dicts()
            table.items[int(row_idx)].items[int(col_idx)].content = _words_to_reading_order_text(words)

        return table

    OCRDataframe.get_text_cell = get_text_cell
    OCRDataframe.get_text_table = get_text_table


def reduce_pdf_resolution(input_pdf_path, target_dpi=30):
    """Reduce PDF resolution after enhancement (lower DPI = smaller pages)."""
    print(f"\n[REDUCE] Reducing PDF resolution to {target_dpi} DPI...")

    try:
        # Convert PDF to images with original resolution
        images = convert_from_path(input_pdf_path)
        reduced_images = []

        for i, img in enumerate(images):
            # Calculate new size based on target DPI and original DPI
            orig_dpi = img.info.get('dpi', (300, 300))[0]
            scale_factor = target_dpi / orig_dpi
            new_size = (max(1, int(img.width * scale_factor)), max(1, int(img.height * scale_factor)))

            # Resize image to new target resolution
            reduced_img = img.resize(new_size, Image.LANCZOS)

            # Save reduced image temporarily in JPEG format to reduce size
            temp_img_path = f"temp_page_{i}.jpg"
            reduced_img.save(temp_img_path, "JPEG", quality=95)
            reduced_images.append(temp_img_path)

        # Convert list of JPEG images back to PDF
        base_name = os.path.splitext(os.path.basename(input_pdf_path))[0]
        output_pdf_path = f"{base_name}_reduced.pdf"

        with open(output_pdf_path, "wb") as f:
            f.write(img2pdf.convert(reduced_images))

        # Clean up temporary images
        for img_path in reduced_images:
            os.remove(img_path)

        print(f"[REDUCE] Reduced resolution PDF saved as: {output_pdf_path}")
        return output_pdf_path

    except Exception as e:
        print(f"[ERROR] Resolution reduction failed: {e}")
        return input_pdf_path


def extract_tables_with_tesseract(input_path, pages=None):
    """Extract tables using TesseractOCR"""
    print(f"\n[EXTRACT] Starting table extraction with TesseractOCR...")

    try:
        # Fix cell word order only (left-to-right by bbox). Detection stays unchanged.
        _patch_img2table_cell_word_order()

        # Initialize OCR (Tesseract)
        ocr = TesseractOCR(n_threads=1, lang="eng")
        input_file = Path(input_path)

        extract_kwargs = dict(
            ocr=ocr,
            implicit_rows=False,
            implicit_columns=False,
            borderless_tables=True,
            min_confidence=10,
        )

        if is_pdf_file(input_path):
            document = PDF(src=input_path,
                           pages=pages,
                           detect_rotation=False,
                           pdf_text_extraction=True)
            page_label = lambda page_num: f"Page {page_num + 1}"
        else:
            document = Img2TableImage(src=input_path, detect_rotation=False)
            page_label = lambda page_num: "Image"

        extracted_tables = document.extract_tables(**extract_kwargs)

        if extracted_tables:
            excel_filename = f"{input_file.stem}_extracted_tables.xlsx"
            document.to_xlsx(dest=excel_filename, **extract_kwargs)
            print(f"[EXTRACT] Tables saved to: {excel_filename}")

            total_tables = sum(len(tables) for tables in extracted_tables.values())
            print(f"[EXTRACT] Total tables extracted: {total_tables}")

            for page_num, tables in extracted_tables.items():
                if tables:
                    print(f"  - {page_label(page_num)}: {len(tables)} table(s)")
                    for table_idx, table in enumerate(tables):
                        print(f"    Table {table_idx + 1}: {table.df.shape[0]} rows × {table.df.shape[1]} columns")

            return excel_filename, total_tables
        else:
            print("[EXTRACT] No tables found in the input file")
            return None, 0

    except Exception as e:
        print(f"[ERROR] Table extraction failed: {e}")
        return None, 0


def main():
    """Main automation function"""
    try:
        # Get all user inputs
        input_path, pages_to_process, rotation_info, enhance_input = get_user_input()

        print(f"\n=== Processing Summary ===")
        print(f"Input: {input_path}")
        if is_pdf_file(input_path):
            print(f"Pages to process: {'All' if pages_to_process is None else [p+1 for p in pages_to_process]}")
        else:
            print("Pages to process: N/A for image input")
        print(f"Rotation: {'Yes' if rotation_info else 'No'}")
        print(f"Enhancement: {'Yes' if enhance_input else 'No'}")

        current_input = input_path
        intermediate_files = []

        if rotation_info:
            if is_pdf_file(current_input):
                current_input = rotate_pdf_pages(current_input, rotation_info)
            else:
                current_input = rotate_image_file(current_input, rotation_info)

            if current_input != input_path:
                intermediate_files.append(current_input)

        if enhance_input:
            if is_pdf_file(current_input):
                current_input = enhance_pdf_tables(current_input)
                if not current_input.endswith('_enhanced.pdf'):
                    print("[WARN] Enhancement may have failed, proceeding with current input")
                else:
                    intermediate_files.append(current_input)

                reduced_pdf = reduce_pdf_resolution(current_input)
                if reduced_pdf != current_input:
                    current_input = reduced_pdf
                    intermediate_files.append(current_input)
            else:
                current_input = enhance_image_tables(current_input)
                if "_enhanced" not in Path(current_input).stem:
                    print("[WARN] Enhancement may have failed, proceeding with current input")
                else:
                    intermediate_files.append(current_input)

        excel_file, table_count = extract_tables_with_tesseract(current_input, pages_to_process)

        if excel_file and table_count > 0:
            print(f"\n=== SUCCESS ===")
            print(f"Extracted {table_count} tables to: {excel_file}")

            keep_final_output = input(f"\nKeep the final processed file ({current_input})? (y/n): ").strip().lower() == 'y'

            print(f"\n[CLEANUP] Cleaning up intermediate files...")
            for file_path in intermediate_files:
                if file_path != current_input or not keep_final_output:
                    try:
                        os.remove(file_path)
                        print(f"[CLEANUP] Removed {file_path}")
                    except Exception as e:
                        print(f"[WARN] Could not delete {file_path}: {e}")

            # Clean up temporary files
            cleanup_temp_files()

            if keep_final_output and current_input != input_path:
                print(f"[INFO] Final processed file kept as: {current_input}")

            print(f"\n=== FINAL OUTPUT ===")
            print(f"Excel file with extracted tables: {excel_file}")

        else:
            print(f"\n=== FAILED ===")
            print("No tables were extracted. Please check your input file and try again.")

            # Still cleanup intermediate files
            for file_path in intermediate_files:
                try:
                    os.remove(file_path)
                except:
                    pass
            cleanup_temp_files()

    except KeyboardInterrupt:
        print(f"\n\n[INTERRUPTED] Process interrupted by user")
        cleanup_temp_files()
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        cleanup_temp_files()


if __name__ == "__main__":
    main()
