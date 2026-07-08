import os
import cv2
import fitz
import shutil
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import img2pdf
from img2table.ocr import TesseractOCR
from img2table.document import PDF
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
    print("=== PDF Table Extraction Automation with TesseractOCR ===\n")

    # Get PDF path
    # pdf_path = input("Enter the path to your PDF file: ").strip()
    pdf_path = "current_test/CH-012026-0062-INVOICE.pdf"

    total_pages = get_pdf_page_count(pdf_path)
    print(f"\nDetected {total_pages} pages in the PDF")

    # Get pages to process
    page_input = input(f"Enter pages to process (1-{total_pages}, e.g., '1,3-5') or press Enter for all: ").strip()
    pages_to_process = parse_page_input(page_input, total_pages) if page_input else None

    # Get rotation preferences
    rotate_pdf = input("\nDo you want to rotate PDF pages? (y/n): ").strip().lower() == 'y'
    rotation_info = {}

    if rotate_pdf:
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

    # Get enhancement preference
    enhance_pdf = input("\nDo you want to enhance the PDF? (y/n): ").strip().lower() == 'y'

    return pdf_path, pages_to_process, rotation_info, enhance_pdf


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


def enhance_pdf_tables(pdf_path, dpi=300, line_strength=2, show_progress=True):
    """Enhanced PDF table enhancement function"""
    if show_progress:
        print(f"\n[ENHANCE] Starting PDF enhancement...")

    # Temporary folder for page images
    temp_dir = "temp_pdf_images"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Convert PDF pages to images
        pages = convert_from_path(pdf_path, dpi=dpi)
        enhanced_images = []

        for i, page in enumerate(pages):
            if show_progress:
                print(f"[ENHANCE] Processing page {i+1}/{len(pages)}...")

            # Convert PIL image to OpenCV format
            img = np.array(page)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # --- Step 1: Grayscale ---
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # --- Step 2: Improve contrast ---
            gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)

            # --- Step 3: Reduce noise ---
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

            # --- Step 4: Adaptive threshold ---
            thresh = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV,
                15, 10
            )

            # --- Step 5: Detect horizontal & vertical lines ---
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

            detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

            # --- Step 6: Combine and strengthen lines ---
            table_mask = cv2.add(detect_horizontal, detect_vertical)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_strength, line_strength))
            table_mask = cv2.dilate(table_mask, kernel, iterations=2)

            # --- Step 7: Smooth edges and repair small gaps ---
            table_mask = cv2.morphologyEx(table_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

            # --- Step 8: Invert and merge with original ---
            inverted_mask = cv2.bitwise_not(table_mask)
            enhanced_gray = cv2.bitwise_and(gray, gray, mask=inverted_mask)
            enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

            # Save temporary enhanced image
            enhanced_path = os.path.join(temp_dir, f"enhanced_page_{i+1}.png")
            cv2.imwrite(enhanced_path, enhanced_bgr)
            enhanced_images.append(Image.open(enhanced_path).convert("RGB"))

        # --- Step 9: Save final enhanced PDF ---
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        pdf_output_path = os.path.join(os.getcwd(), f"{base_name}_enhanced.pdf")
        enhanced_images[0].save(pdf_output_path, save_all=True, append_images=enhanced_images[1:])

        if show_progress:
            print(f"[ENHANCE] Enhanced PDF saved as: {pdf_output_path}")

        return pdf_output_path

    except Exception as e:
        print(f"[ERROR] Enhancement failed: {e}")
        return pdf_path


def reduce_pdf_resolution(input_pdf_path, target_dpi=72):
    """Reduce PDF resolution"""
    print(f"\n[REDUCE] Reducing PDF resolution to {target_dpi} DPI...")

    try:
        # Convert PDF to images with original resolution
        images = convert_from_path(input_pdf_path)
        reduced_images = []

        for i, img in enumerate(images):
            # Calculate new size based on target DPI and original DPI
            orig_dpi = img.info.get('dpi', (300,300))[0]
            scale_factor = target_dpi / orig_dpi
            new_size = (int(img.width * scale_factor), int(img.height * scale_factor))

            # Resize image to new target resolution
            reduced_img = img.resize(new_size, Image.LANCZOS)

            # Save reduced image temporarily in JPEG format to reduce size
            temp_img_path = f"temp_page_{i}.jpg"
            reduced_img.save(temp_img_path, "JPEG", quality=85)
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


def extract_tables_with_tesseract(pdf_path, pages=None):
    """Extract tables using TesseractOCR"""
    print(f"\n[EXTRACT] Starting table extraction with TesseractOCR...")

    try:
        # Initialize OCR (Tesseract)
        ocr = TesseractOCR(n_threads=1, lang="eng")

        # Initialize PDF document with specific pages
        pdf = PDF(src=pdf_path,
                  pages=pages,
                  detect_rotation=False,
                  pdf_text_extraction=True)

        # Extract tables from the selected pages
        extracted_tables = pdf.extract_tables(
            ocr=ocr,
            implicit_rows=False,
            implicit_columns=False,
            borderless_tables=True,
            min_confidence=10
        )

        if extracted_tables:
            # Create a single Excel file with all tables
            excel_filename = "extracted_tables.xlsx"
            pdf.to_xlsx(dest=excel_filename, ocr=ocr)
            print(f"[EXTRACT] Tables saved to: {excel_filename}")

            # Count and display summary
            total_tables = sum(len(tables) for tables in extracted_tables.values())
            print(f"[EXTRACT] Total tables extracted: {total_tables}")

            for page_num, tables in extracted_tables.items():
                if tables:
                    print(f"  - Page {page_num + 1}: {len(tables)} table(s)")
                    for table_idx, table in enumerate(tables):
                        print(f"    Table {table_idx + 1}: {table.df.shape[0]} rows × {table.df.shape[1]} columns")

            return excel_filename, total_tables
        else:
            print("[EXTRACT] No tables found in the PDF")
            return None, 0

    except Exception as e:
        print(f"[ERROR] Table extraction failed: {e}")
        return None, 0


def main():
    """Main automation function"""
    try:
        # Get all user inputs
        pdf_path, pages_to_process, rotation_info, enhance_pdf = get_user_input()

        print(f"\n=== Processing Summary ===")
        print(f"PDF: {pdf_path}")
        print(f"Pages to process: {'All' if pages_to_process is None else [p+1 for p in pages_to_process]}")
        print(f"Rotation: {'Yes' if rotation_info else 'No'}")
        print(f"Enhancement: {'Yes' if enhance_pdf else 'No'}")

        current_pdf = pdf_path
        intermediate_files = []

        # Step 1: Rotate PDF if requested
        if rotation_info:
            current_pdf = rotate_pdf_pages(current_pdf, rotation_info)
            if current_pdf != pdf_path:
                intermediate_files.append(current_pdf)

        # Step 2: Enhance PDF if requested
        if enhance_pdf:
            current_pdf = enhance_pdf_tables(current_pdf)
            if not current_pdf.endswith('_enhanced.pdf'):
                print("[WARN] Enhancement may have failed, proceeding with current PDF")
            else:
                intermediate_files.append(current_pdf)

            # Step 3: Reduce resolution (only if enhanced)
            reduced_pdf = reduce_pdf_resolution(current_pdf)
            if reduced_pdf != current_pdf:
                current_pdf = reduced_pdf
                intermediate_files.append(current_pdf)

        # Step 4: Extract tables using TesseractOCR
        excel_file, table_count = extract_tables_with_tesseract(current_pdf, pages_to_process)

        if excel_file and table_count > 0:
            print(f"\n=== SUCCESS ===")
            print(f"Extracted {table_count} tables to: {excel_file}")

            # Ask user about keeping the final processed PDF
            keep_final_pdf = input(f"\nKeep the final processed PDF ({current_pdf})? (y/n): ").strip().lower() == 'y'

            # Cleanup intermediate files
            print(f"\n[CLEANUP] Cleaning up intermediate files...")
            for file_path in intermediate_files:
                if file_path != current_pdf or not keep_final_pdf:
                    try:
                        os.remove(file_path)
                        print(f"[CLEANUP] Removed {file_path}")
                    except Exception as e:
                        print(f"[WARN] Could not delete {file_path}: {e}")

            # Clean up temporary files
            cleanup_temp_files()

            if keep_final_pdf and current_pdf != pdf_path:
                print(f"[INFO] Final processed PDF kept as: {current_pdf}")

            print(f"\n=== FINAL OUTPUT ===")
            print(f"Excel file with extracted tables: {excel_file}")

        else:
            print(f"\n=== FAILED ===")
            print("No tables were extracted. Please check your PDF and try again.")

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
