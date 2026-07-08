import pdfplumber
import pandas as pd

PDF_PATH = 'B&W-INV=53-BLR.pdf'
OUTPUT_XLSX = 'invoice_data.xlsx'

def extract_text_and_tables(pdf_path):
    texts = []
    tables = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # 1. Extract raw text
            raw_text = page.extract_text() or ''
            texts.append({'page': page_num, 'text': raw_text})

            # 2. Extract tables using visible lines
            table = page.extract_table({
                'vertical_strategy': 'lines',
                'horizontal_strategy': 'lines'
            })
            if table:
                # First row as header, rest as data
                headers, *rows = table
                df = pd.DataFrame(rows, columns=headers)
                tables.append((page_num, df))

    return texts, tables

def save_to_excel(texts, tables, output_path):
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Write raw text
        df_text = pd.DataFrame(texts)
        df_text.to_excel(writer, sheet_name='Full_Text', index=False)

        # Write each table on its own sheet
        for idx, (page_num, df_table) in enumerate(tables, start=1):
            sheet_name = f'Table_Page_{page_num}'
            # Excel sheet names max-length = 31 chars
            df_table.to_excel(writer, sheet_name=sheet_name[:31], index=False)

if __name__ == '__main__':
    texts, tables = extract_text_and_tables(PDF_PATH)
    save_to_excel(texts, tables, OUTPUT_XLSX)
    print(f'Extraction complete: {OUTPUT_XLSX}')
