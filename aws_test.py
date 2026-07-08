import boto3
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

textract = boto3.client(
    'textract',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)

def extract_tables_local(file_path):
    """Extracts tables from a local image or PDF (<5 MB) via Textract."""
    with open(file_path, 'rb') as f:
        bytes_data = f.read()
    response = textract.analyze_document(
        Document={'Bytes': bytes_data},
        FeatureTypes=['TABLES']
    )
    # Build ID lookup
    id_map = {b['Id']: b for b in response['Blocks']}
    tables = []
    for block in response['Blocks']:
        if block['BlockType'] == 'TABLE':
            # Collect child CELL blocks
            cells = []
            for rel in block.get('Relationships', []):
                if rel['Type'] == 'CHILD':
                    for cid in rel['Ids']:
                        cell = id_map[cid]
                        if cell['BlockType'] == 'CELL':
                            # Concatenate WORD texts
                            text = ''
                            for crel in cell.get('Relationships', []):
                                if crel['Type'] == 'CHILD':
                                    for wid in crel['Ids']:
                                        if id_map[wid]['BlockType'] == 'WORD':
                                            text += id_map[wid]['Text'] + ' '
                            cells.append({
                                'row': cell['RowIndex'],
                                'col': cell['ColumnIndex'],
                                'text': text.strip()
                            })
            # Build 2D list
            if cells:
                max_row = max(c['row'] for c in cells)
                max_col = max(c['col'] for c in cells)
                table = [['' for _ in range(max_col)] for _ in range(max_row)]
                for c in cells:
                    table[c['row']-1][c['col']-1] = c['text']
                tables.append(table)
    return tables

def export_tables_to_excel(tables, excel_path):
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for idx, table in enumerate(tables, start=1):
            # Convert each table (2D list) to DataFrame
            df = pd.DataFrame(table)
            sheet_name = f"Table{idx}"
            df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
    print(f"✅ Exported {len(tables)} tables to '{excel_path}'")

if __name__ == "__main__":
    # 1. Specify your document
    document = 'page2.png'  # or .pdf (<5 MB)
    # 2. Extract tables
    extracted_tables = extract_tables_local(document)
    # 3. Export to Excel
    export_tables_to_excel(extracted_tables, 'extracted_tables.xlsx')

