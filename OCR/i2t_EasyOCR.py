import os
import cv2
import fitz         
from img2table.ocr import EasyOCR
from img2table.document import PDF
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

def extract_and_save_tables_with_dataframes(pdf_path, pages=None):
    # Create OCR instance
    ocr = EasyOCR(lang=["en"], kw={"download_enabled": False, "gpu": False})
    
    # Extract tables
    pdf_doc = PDF(src=pdf_path, pages=pages)
    extracted_tables = pdf_doc.extract_tables(ocr=ocr)
    
    # Save to Excel
    pdf_doc.to_xlsx(dest="extracted_tables.xlsx", ocr=ocr)
    print("Formatted Excel saved as 'extracted_tables.xlsx'")
    
    # Build list of dataframes
    dataframes_list = []
    for page_num, tables_on_page in extracted_tables.items():
        for table_idx, table_obj in enumerate(tables_on_page, start=1):
            dataframes_list.append({
                'dataframe': table_obj.df,
                'page': page_num + 1, 
                'table_num': table_idx
            })
    
    return dataframes_list  # Return the list!

def main():
    # pdf_path = input("Enter the path to your PDF file: ").strip()
    pdf_path = "current_test/CH-012026-0062-INVOICE.pdf"

    pages = get_user_page_selection(pdf_path)
    display = ','.join(str(p+1) for p in pages) if pages else 'all'
    print(f"\nProcessing pages: {display}")

    print("\nExtracting tables... This may take a moment.")
    
    dataframes_list = extract_and_save_tables_with_dataframes(pdf_path, pages)
        
    for item in dataframes_list:
        df = item['dataframe']
        page = item['page']
        table_num = item['table_num']
        
        print(f"\n=== Processing Page {page}, Table {table_num} ===")
        
        # Example processing:
        if not df.empty:
            print(f"Table shape: {df.shape}")
            print("Sample data:")
            print(df.head(3))
        
        print("-" * 50)
    
    print(f"\nTotal tables extracted: {len(dataframes_list)}")
    print("Check 'extracted_tables.xlsx' for properly formatted output")
    


if __name__ == "__main__":
    main()
