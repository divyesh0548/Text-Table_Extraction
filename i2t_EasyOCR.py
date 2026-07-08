import os
import cv2
from img2table.ocr import EasyOCR
from img2table.document import PDF

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
    cv2.ximgproc.BINARIZATION_SAUVOLA = 0
if not hasattr(cv2.ximgproc, "BINARIZATION_NIBLACK"):
    cv2.ximgproc.BINARIZATION_NIBLACK = 1

def extract_tables_from_pdf(pdf_path, selected_pages=None):
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file '{pdf_path}' not found.")
        return None

    try:
        # Use img2table's built-in EasyOCR wrapper
        ocr = EasyOCR(
            lang=["en"],
            kw={  # pass any Tesseract or EasyOCR kwargs here
                "download_enabled": False,
                "gpu": False
            }
        )

        pdf = PDF(
            src=pdf_path,
            pages=selected_pages,
            detect_rotation=False,
            pdf_text_extraction=True
        )

        tables = pdf.extract_tables(
            ocr=ocr,
            implicit_rows=True,
            implicit_columns=True,
            borderless_tables=True,
            min_confidence=50
        )
        return tables

    except Exception as e:
        print(f"Error extracting tables: {e}")
        return None

def get_user_page_selection(pdf_path):
    try:
        pdf_temp = PDF(src=pdf_path)
        if not hasattr(pdf_temp, "pages") or not pdf_temp.pages:
            print("Warning: No pages detected in PDF. Will process all pages.")
            return None
        total_pages = len(pdf_temp.pages)

        print(f"\nPDF Analysis:\nFile: {pdf_path}\nTotal pages: {total_pages}")
        # rest remains same
        # ...
    except Exception as e:
        print(f"Page selection error: {e}")
        return None

def save_tables_to_files(tables, output_dir="extracted_tables"):
    if not tables:
        print("No tables to save.")
        return

    os.makedirs(output_dir, exist_ok=True)
    for page_idx, tbl_list in tables.items():
        if not tbl_list:
            print(f"\nPage {page_idx+1}: No tables found")
            continue
        print(f"\nPage {page_idx+1}: Found {len(tbl_list)} table(s)")
        for idx, table in enumerate(tbl_list, start=1):
            fname = f"page_{page_idx+1}_table_{idx}.csv"
            path = os.path.join(output_dir, fname)
            table.df.to_csv(path, index=False)
            print(f" Table {idx} saved to {path}")
            print(table.df.head().to_string(index=False))
            print("-" * 50)

def main():
    pdf_path = input("Enter the path to your PDF file: ").strip().strip('"\'')
    pages = get_user_page_selection(pdf_path)
    print(f"\nProcessing pages: {([p+1 for p in pages] if pages else 'all')}")

    print("\nExtracting tables... This may take a moment.")
    tables = extract_tables_from_pdf(pdf_path, pages)
    if tables:
        save_tables_to_files(tables)
        if input("\nExport tables to Excel? (y/n): ").strip().lower() == "y":
            try:
                ocr = EasyOCR(lang=["en"], kw={"download_enabled": False, "gpu": False})
                PDF(src=pdf_path, pages=pages).to_xlsx(dest="extracted_tables.xlsx", ocr=ocr)
                print("Excel saved as extracted_tables.xlsx")
            except Exception as e:
                print(f"Error exporting Excel: {e}")
    else:
        print("No tables were extracted.")

if __name__ == "__main__":
    main()
