import pdfplumber
from openpyxl import Workbook

PDF_PATH = 'B&W-INV=53-BLR.pdf'
OUTPUT_XLSX = 'invoice_seq_extraction.xlsx'

def extract_and_export(pdf_path, output_path):
    # Create a new Excel workbook and sheet
    wb = Workbook()
    ws = wb.active
    ws.title = 'Extraction'

    table_counter = 0

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # 1. Write page text block
            text = page.extract_text() or ''
            ws.append([f'PAGE_{page_num}_TEXT'])
            for line in text.splitlines():
                ws.append([line])
            ws.append([])  # blank row separator

            # 2. Extract tables with both methods
            raw_tables = []
            raw_tables += page.extract_tables({
                'vertical_strategy': 'lines',
                'horizontal_strategy': 'lines'
            }) or []
            raw_tables += page.extract_tables() or []

            # 3. Write each table sequentially
            for raw in raw_tables:
                table_counter += 1
                ws.append([f'TABLE_{table_counter}_START'])
                # Header row
                ws.append(raw[0])
                # Data rows
                for row in raw[1:]:
                    ws.append(row)
                ws.append([f'TABLE_{table_counter}_END'])
                ws.append([])  # blank row separator

    # Save workbook
    wb.save(output_path)
    print(f'Extraction complete: {output_path}')

if __name__ == '__main__':
    extract_and_export(PDF_PATH, OUTPUT_XLSX)
