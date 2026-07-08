import os
from img2table.ocr import TesseractOCR
from img2table.document import PDF
import cv2


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

# -- Provide missing binarization constants --
if not hasattr(cv2.ximgproc, "BINARIZATION_NIBLACK"):
    cv2.ximgproc.BINARIZATION_NIBLACK = 0
if not hasattr(cv2.ximgproc, "BINARIZATION_SAUVOLA"):
    cv2.ximgproc.BINARIZATION_SAUVOLA = 1

def extract_tables_from_pdf(pdf_path, selected_pages=None):
    
    # Check if PDF file exists
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file '{pdf_path}' not found.")
        return None
    
    try:
        # Initialize OCR (Tesseract)
        ocr = TesseractOCR(n_threads=1, lang="eng")
        
        # Initialize PDF document with specific pages
        pdf = PDF(src=pdf_path, 
                 pages=selected_pages,  # This is the key parameter for page selection
                 detect_rotation=False,
                 pdf_text_extraction=True)
        
        # Extract tables from the selected pages
        extracted_tables = pdf.extract_tables(
            ocr=ocr,
            implicit_rows=True,
            implicit_columns=True,
            borderless_tables=True,
            min_confidence=50
        )
        
        return extracted_tables
    
    except Exception as e:
        print(f"Error extracting tables: {str(e)}")
        return None

def get_user_page_selection(pdf_path):
    try:
        pdf_temp = PDF(src=pdf_path)
        # If pages attribute is missing or empty, default to all pages
        if not hasattr(pdf_temp, "pages") or not pdf_temp.pages:
            print("Warning: No pages detected. Processing all pages.")
            return None
        total_pages = len(pdf_temp.pages)

        print(f"\nPDF Analysis:")
        print(f"File: {pdf_path}")
        print(f"Total pages: {total_pages}")
        print("\nPage Selection Options:")
        print("1. Process all pages")
        print("2. Process specific pages")
        print("3. Process a range of pages")

        choice = input("\nEnter your choice (1-3): ").strip()
        if choice == "1":
            return None
        elif choice == "2":
            page_input = input(f"Enter page numbers (1-{total_pages}) separated by commas: ").strip()
            try:
                selected = [int(p.strip())-1 for p in page_input.split(",")]
                valid = [p for p in selected if 0 <= p < total_pages]
                return valid or None
            except ValueError:
                print("Invalid input. Processing all pages.")
                return None
        elif choice == "3":
            start = input(f"Enter start page (1-{total_pages}): ").strip()
            end   = input(f"Enter end page (1-{total_pages}): ").strip()
            try:
                s, e = int(start)-1, int(end)-1
                if 0 <= s <= e < total_pages:
                    return list(range(s, e+1))
            except ValueError:
                pass
            print("Invalid range. Processing all pages.")
            return None
        else:
            print("Invalid choice. Processing all pages.")
            return None

    except Exception as e:
        print(f"Error in page selection: {e}")
        return None

def save_tables_to_files(extracted_tables, output_dir="extracted_tables"):
    
    if not extracted_tables:
        print("No tables to save.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for page_num, tables in extracted_tables.items():
        if tables:  # If there are tables on this page
            print(f"\nPage {page_num + 1}: Found {len(tables)} table(s)")
            
            for table_idx, table in enumerate(tables):
                # Save as CSV
                csv_filename = f"page_{page_num + 1}_table_{table_idx + 1}.csv"
                csv_path = os.path.join(output_dir, csv_filename)
                table.df.to_csv(csv_path, index=False)
                
                # Display table info
                print(f"  Table {table_idx + 1}:")
                print(f"    Shape: {table.df.shape}")
                print(f"    Saved to: {csv_path}")
                print(f"    Preview:")
                print(table.df.head().to_string(index=False))
                print("-" * 50)
        else:
            print(f"\nPage {page_num + 1}: No tables found")

def main():
    # Get PDF file path from user
    pdf_path = input("Enter the path to your PDF file: ").strip()
    
    # Remove quotes if user added them
    pdf_path = pdf_path.strip('"\'')
    
    # Get user's page selection
    selected_pages = get_user_page_selection(pdf_path)
    
    if selected_pages is not None:
        print(f"\nProcessing pages: {[p + 1 for p in selected_pages]}")
    else:
        print("\nProcessing all pages...")
    
    # Extract tables
    print("\nExtracting tables... This may take a moment.")
    extracted_tables = extract_tables_from_pdf(pdf_path, selected_pages)
    
    if extracted_tables:
        # Save tables to files
        save_tables_to_files(extracted_tables)
        
        # Option to export to Excel
        export_excel = input("\nWould you like to export all tables to Excel? (y/n): ").strip().lower()
        if export_excel == 'y':
            try:
                pdf = PDF(src=pdf_path, pages=selected_pages)
                ocr = TesseractOCR(n_threads=1, lang="eng")
                excel_path = "i2t_output.xlsx"
                pdf.to_xlsx(dest=excel_path, ocr=ocr)
                print(f"Excel file saved as: {excel_path}")
            except Exception as e:
                print(f"Error creating Excel file: {str(e)}")
    else:
        print("No tables were extracted.")

if __name__ == "__main__":
    main()
