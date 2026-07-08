import pandas as pd
import camelot
import os
# import pytesseract

# pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Divyesh Parmar\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

def remove_duplicate_headers_largest_content(df):
        df['row_signature'] = df.apply(lambda row: tuple(str(cell).strip().lower() for cell in row if str(cell).strip()), axis=1)
        # Count non-empty cells per row
        df['non_empty_cells'] = df.apply(lambda row: sum(1 for cell in row if str(cell).strip()), axis=1)
        # Or use total character length of all cells combined as "size"
        df['content_length'] = df.apply(lambda row: sum(len(str(cell)) for cell in row), axis=1)
    
        # Find duplicated signatures
        duplicated_sigs = df['row_signature'][df['row_signature'].duplicated(keep=False)]
        duplicated_sigs_unique = duplicated_sigs.unique()
        
        rows_to_drop = []
        for sig in duplicated_sigs_unique:
            rows = df[df['row_signature'] == sig]
            # Select the row with maximum content length (size)
            idx_to_keep = rows['content_length'].idxmax()
            # Drop other duplicates except the largest (likely header)
            to_drop = rows.index.difference([idx_to_keep])
            rows_to_drop.extend(to_drop)
        
        df_cleaned = df.drop(rows_to_drop).drop(['row_signature', 'non_empty_cells', 'content_length'], axis=1)
        return df_cleaned.reset_index(drop=True)

table_reports = []
pdf_file = "AnnexureASP.pdf"
    
tables = camelot.read_pdf(pdf_file, pages="all", flavor="stream") 
# tables = camelot.read_pdf(pdf_file, pages="all", flavor="lattice") 
# tables = camelot.read_pdf(pdf_file, pages="all", process_background=True) 
print(f"{len(tables)} tables found")
table_num = len(tables)
for index, table in enumerate(tables):
    report = table.parsing_report
    table_reports.append({
        'index' : index + 1,
        'accuracy' : report.get('accuracy', 'N/A')
    })

print(f"Table Reports: {table_reports}")

final_df = pd.concat([table.df for table in tables], ignore_index=True)
# final_df = remove_duplicate_headers_largest_content(final_df)
pd.set_option('display.max_rows', None)

def consolidate_dataframe_rows(df):
    """
    Consolidates DataFrame rows based on numeric ID patterns.
    Ensures proper spacing between merged text values.
    PRESERVES original column headers.
    """
    # Reset index to ensure consistent numbering
    df_work = df.reset_index(drop=True).copy()
    
    # Store original column names to preserve headers
    original_columns = df_work.columns.tolist()
    
    # Convert all columns to string and replace NaN with empty strings
    for col in df_work.columns:
        df_work[col] = df_work[col].astype(str).replace('nan', '').replace('NaN', '')
    
    # Find where new records start (rows containing long numeric IDs)
    def contains_long_number(row):
        for val in row:
            val_clean = str(val).strip().replace('.', '').replace('e+', '').replace('E+', '')
            if val_clean.isdigit() and len(val_clean) >= 10:
                return True
        return False
    
    # Identify group boundaries
    group_starts = []
    for i, row in df_work.iterrows():
        if contains_long_number(row.values):
            group_starts.append(i)
    
    if not group_starts:
        return df_work
    
    # Create groups and consolidate
    consolidated_rows = []
    
    for i, start_idx in enumerate(group_starts):
        end_idx = group_starts[i + 1] if i + 1 < len(group_starts) else len(df_work)
        group_data = df_work.iloc[start_idx:end_idx]
        consolidated_row = {}
        
        for col in original_columns:  # Use original column names
            values = []
            for val in group_data[col]:
                val_str = str(val).strip()
                if val_str and val_str != '' and val_str != 'nan':
                    values.append(val_str)
            
            if values:
                numeric_values = []
                text_values = []
                
                for v in values:
                    v_clean = v.replace('.', '').replace('e+', '').replace('E+', '')
                    if v_clean.isdigit() and len(v_clean) >= 10:
                        numeric_values.append(v)
                    else:
                        text_values.append(v)
                
                if numeric_values:
                    consolidated_row[col] = numeric_values[0]
                elif text_values:
                    # Join with proper spacing and clean up multiple spaces
                    combined_text = ' '.join(text_values)
                    consolidated_row[col] = ' '.join(combined_text.split())
                else:
                    consolidated_row[col] = values[0]
            else:
                consolidated_row[col] = ''
        
        consolidated_rows.append(consolidated_row)
    
    # Create the result DataFrame with original column names preserved
    result_df = pd.DataFrame(consolidated_rows, columns=original_columns)
    
    return result_df

# Usage:
# final_df = consolidate_dataframe_rows(final_df)

# Apply it to your DataFrame

final_df.to_excel('extracted_tables.xlsx', index=False, header=False)
print("✅ Final DataFrame saved to 'extracted_tables.xlsx'")

