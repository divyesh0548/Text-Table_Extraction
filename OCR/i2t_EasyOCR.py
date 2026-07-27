import os
import shutil
import tempfile
import cv2
import fitz         
from pathlib import Path
from img2table.ocr import EasyOCR
from img2table.document import PDF, Image as Img2TableImage
import pandas as pd
import re
from openpyxl import load_workbook

pd.set_option('display.max_rows', None)

# Monkey-patch OpenCV ximgproc thresholds if missing
if not hasattr(cv2.ximgproc, "niBlackThreshold"):
    def niBlackThreshold(src, maxValue, *args, **kwargs):
        return cv2.adaptiveThreshold(
            src,
            maxValue,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
    cv2.ximgproc.niBlackThreshold = niBlackThreshold

if not hasattr(cv2.ximgproc, "BINARIZATION_SAUVOLA"):
    cv2.ximgproc.BINARIZATION_SAUVOLA = 1  # or another default int value

if not hasattr(cv2.ximgproc, "BINARIZATION_NIBLACK"):
    cv2.ximgproc.BINARIZATION_NIBLACK = 0  # or a default int value

PDF_EXTENSIONS = {'.pdf'}
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def is_pdf_file(file_path):
    return Path(file_path).suffix.lower() in PDF_EXTENSIONS


def is_image_file(file_path):
    return Path(file_path).suffix.lower() in IMAGE_EXTENSIONS

def parse_page_input(page_input, total_pages):
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
    return sorted(pages) or None

def get_user_page_selection(pdf_path):
    if not is_pdf_file(pdf_path):
        return None

    try:
        pdf_temp = PDF(src=pdf_path)
        if hasattr(pdf_temp, "pages") and pdf_temp.pages:
            total = len(pdf_temp.pages)
        else:
            # Fallback: use fitz to count pages
            doc = fitz.open(pdf_path)
            total = doc.page_count
            doc.close()
            print(f"Info: Detected {total} pages via fitz.")

        inp = input(f"Enter pages/ranges (1–{total}, e.g. '1,3-5') or press Enter for all: ").strip()
        return parse_page_input(inp, total) if inp else None

    except Exception as e:
        print(f"Page selection error: {e}")
        return None

def get_pdf_page_count(pdf_path):
    doc = fitz.open(pdf_path)
    try:
        return doc.page_count
    finally:
        doc.close()

def get_rotation_selection(pdf_path):
    if is_image_file(pdf_path):
        rotation_input = input(
            "Enter rotation angle for the image (90, 180, 270) or press Enter for none: "
        ).strip()

        if not rotation_input:
            return None, None, None

        try:
            rotation = int(rotation_input)
        except ValueError:
            print("Invalid rotation angle. Skipping rotation.")
            return None, None, None

        if rotation not in (90, 180, 270):
            print("Rotation angle must be one of 90, 180, 270. Skipping rotation.")
            return None, None, None

        direction_input = input(
            "Rotate clockwise or counterclockwise? Enter 'cw' or 'ccw' (default: cw): "
        ).strip().lower()

        if direction_input in ("", "cw", "clockwise"):
            return rotation, None, "clockwise"
        if direction_input in ("ccw", "counterclockwise", "counter-clockwise"):
            return (360 - rotation) % 360, None, "counterclockwise"

        print("Invalid rotation direction. Skipping rotation.")
        return None, None, None

    total_pages = get_pdf_page_count(pdf_path)
    rotation_input = input(
        "Enter rotation angle for PDF pages (90, 180, 270) or press Enter for none: "
    ).strip()

    if not rotation_input:
        return None, None, None

    try:
        rotation = int(rotation_input)
    except ValueError:
        print("Invalid rotation angle. Skipping rotation.")
        return None, None, None

    if rotation not in (90, 180, 270):
        print("Rotation angle must be one of 90, 180, 270. Skipping rotation.")
        return None, None, None

    direction_input = input(
        "Rotate clockwise or counterclockwise? Enter 'cw' or 'ccw' (default: cw): "
    ).strip().lower()

    if direction_input in ("", "cw", "clockwise"):
        direction_label = "clockwise"
        effective_rotation = rotation
    elif direction_input in ("ccw", "counterclockwise", "counter-clockwise"):
        direction_label = "counterclockwise"
        effective_rotation = (360 - rotation) % 360
    else:
        print("Invalid rotation direction. Skipping rotation.")
        return None, None, None

    pages_input = input(
        f"Enter pages/ranges to rotate (1-{total_pages}, e.g. '1,3-5') or press Enter for all: "
    ).strip()
    rotate_pages = parse_page_input(pages_input, total_pages) if pages_input else None

    return effective_rotation, rotate_pages, direction_label

def create_rotated_pdf(pdf_path, rotation, rotate_pages=None):
    doc = fitz.open(pdf_path)
    try:
        total_pages = doc.page_count
        target_pages = set(rotate_pages if rotate_pages is not None else range(total_pages))

        for page_index in target_pages:
            if 0 <= page_index < total_pages:
                page = doc.load_page(page_index)
                page.set_rotation(rotation % 360)

        temp_dir = tempfile.mkdtemp(prefix="easyocr_rotated_")
        rotated_pdf_path = os.path.join(temp_dir, os.path.basename(pdf_path))
        doc.save(rotated_pdf_path)
        return rotated_pdf_path, temp_dir
    finally:
        doc.close()


def create_rotated_image(image_path, rotation):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    if rotation == 90:
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        rotated = cv2.rotate(image, cv2.ROTATE_180)
    elif rotation == 270:
        rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError(f"Unsupported rotation angle: {rotation}")

    temp_dir = tempfile.mkdtemp(prefix="easyocr_rotated_")
    image_path_obj = Path(image_path)
    rotated_image_path = os.path.join(temp_dir, image_path_obj.name)
    cv2.imwrite(rotated_image_path, rotated)
    return rotated_image_path, temp_dir

def extract_and_save_tables_with_dataframes(input_path, pages=None):
    # Create OCR instance
    ocr = EasyOCR(lang=["en"], kw={"download_enabled": False, "gpu": False})

    input_path_obj = Path(input_path)
    excel_output = f"{input_path_obj.stem}_extracted_tables.xlsx"

    if is_pdf_file(input_path):
        document = PDF(src=input_path, pages=pages)
    else:
        document = Img2TableImage(src=input_path)

    extracted_tables = document.extract_tables(ocr=ocr)

    document.to_xlsx(dest=excel_output, ocr=ocr)
    print(f"Formatted Excel saved as '{excel_output}'")
    
    # Build list of dataframes
    dataframes_list = []
    for page_num, tables_on_page in extracted_tables.items():
        for table_idx, table_obj in enumerate(tables_on_page, start=1):
            dataframes_list.append({
                'dataframe': table_obj.df,
                'page': page_num + 1 if is_pdf_file(input_path) else "Image",
                'table_num': table_idx
            })
    
    return dataframes_list  # Return the list!

def main():
    # pdf_path = input("Enter the path to your PDF file: ").strip()
    # pdf_path = "Current-Test/MBP 1 AKB 2026.pdf"
    # pdf_path = "page_1_table_crop.png"
    pdf_path = "Aarti - Mandays Apr 26.pdf"

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Input file not found: {Path(pdf_path).resolve()}")

    if not (is_pdf_file(pdf_path) or is_image_file(pdf_path)):
        raise ValueError("Unsupported input format. Use PDF, PNG, JPG, JPEG, BMP, TIF, or TIFF.")

    rotation, rotate_pages, rotation_direction = get_rotation_selection(pdf_path)
    working_pdf_path = pdf_path
    temp_dir = None

    if rotation:
        if is_pdf_file(pdf_path):
            working_pdf_path, temp_dir = create_rotated_pdf(pdf_path, rotation, rotate_pages)
            rotated_display = ','.join(str(p + 1) for p in rotate_pages) if rotate_pages else 'all'
            requested_angle = rotation if rotation_direction == "clockwise" else (360 - rotation) % 360
            print(f"\nApplied {requested_angle}° {rotation_direction} rotation to pages: {rotated_display}")
        else:
            working_pdf_path, temp_dir = create_rotated_image(pdf_path, rotation)
            requested_angle = rotation if rotation_direction == "clockwise" else (360 - rotation) % 360
            print(f"\nApplied {requested_angle}° {rotation_direction} rotation to image")

    try:
        pages = get_user_page_selection(working_pdf_path)
        if is_pdf_file(working_pdf_path):
            display = ','.join(str(p+1) for p in pages) if pages else 'all'
            print(f"\nProcessing pages: {display}")
        else:
            print("\nProcessing image input")

        print("\nExtracting tables... This may take a moment.")
        
        dataframes_list = extract_and_save_tables_with_dataframes(working_pdf_path, pages)
            
        for item in dataframes_list:
            df = item['dataframe']
            page = item['page']
            table_num = item['table_num']
            
            if page == "Image":
                print(f"\n=== Processing Image, Table {table_num} ===")
            else:
                print(f"\n=== Processing Page {page}, Table {table_num} ===")
            
            # Example processing:
            if not df.empty:
                print(f"Table shape: {df.shape}")
                print("Sample data:")
                print(df.head(3))
            
            print("-" * 50)
        
        print(f"\nTotal tables extracted: {len(dataframes_list)}")
        print("Check the generated '*_extracted_tables.xlsx' file for properly formatted output")
    finally:
        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
    


if __name__ == "__main__":
    main()
