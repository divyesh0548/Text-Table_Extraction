import pandas as pd
import camelot
import os
pd.set_option('display.max_rows', None)
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

def remove_duplicate_headers_in_body(df, metadata_cols=2):
    # Split off metadata vs. body
    meta = df.iloc[:, :metadata_cols]
    body = df.iloc[:, metadata_cols:].copy()
    
    # Compute a signature per row based on non-empty, lowercased cell values
    body['row_sig'] = body.apply(
        lambda row: tuple(str(cell).strip().lower() for cell in row if str(cell).strip()),
        axis=1
    )
    # Compute “size” of each row by total character count
    body['size'] = body.apply(lambda row: sum(len(str(cell)) for cell in row), axis=1)
    
    # Find duplicated signatures
    dup_sigs = body['row_sig'][body['row_sig'].duplicated(keep=False)].unique()
    to_drop = []
    for sig in dup_sigs:
        rows = body[body['row_sig'] == sig]
        # Keep the one with largest size
        keep_idx = rows['size'].idxmax()
        drop_idx = rows.index.difference([keep_idx])
        to_drop.extend(drop_idx)
    
    # Drop duplicates, then drop helper cols and reset index
    body_clean = (
        body
        .drop(index=to_drop)
        .drop(columns=['row_sig','size'])
        .reset_index(drop=True)
    )
    
    # Re-attach metadata columns (repeat metadata for remaining body rows)
    # We need to repeat each meta row the same number of times as it survived in body_clean
    # Here we assume metadata rows align one-to-one with body rows before cleaning
    meta_clean = meta.reset_index(drop=True).loc[body_clean.index]
    
    return pd.concat([meta_clean, body_clean], axis=1)

def consolidate_dataframeX_rows(df):
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

def consolidate_dataframe_rows(df, metadata_cols=2):
    """
    Consolidates DataFrame rows based on numeric ID patterns,
    ignoring the first `metadata_cols` columns (e.g., PDF_File, Page_Number).
    PRESERVES original column headers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame with metadata + rows to consolidate
    metadata_cols (int): Number of leading metadata columns to preserve
    
    Returns:
    pd.DataFrame: Consolidated DataFrame
    """
    # Split off metadata vs. body
    meta = df.iloc[:, :metadata_cols].reset_index(drop=True)
    body = df.iloc[:, metadata_cols:].reset_index(drop=True).copy()

    # Preserve original body column names
    original_body_cols = body.columns.tolist()
    
    # Convert all to string and replace NaN
    body = body.astype(str).replace({'nan': '', 'NaN': ''})
    
    # Helper: detect rows that start a new record (long numeric ID)
    def is_start(row):
        for val in row:
            v = str(val).strip().replace('.', '').replace('e+', '').replace('E+', '')
            if v.isdigit() and len(v) >= 10:
                return True
        return False
    
    # Find group boundaries
    starts = [i for i, r in body.iterrows() if is_start(r.values)]
    if not starts:
        return df.copy()
    
    # Build consolidated rows
    rows = []
    for i, start in enumerate(starts):
        end = starts[i+1] if i+1 < len(starts) else len(body)
        chunk = body.iloc[start:end]
        
        consolidated = {}
        for col in original_body_cols:
            vals = [v.strip() for v in chunk[col] if v.strip()]
            if not vals:
                consolidated[col] = ''
                continue
            
            # Separate numeric IDs vs. text
            nums = [v for v in vals if v.replace('.', '').replace('e+', '').replace('E+', '').isdigit() and len(v.replace('.', ''))>=10]
            text = [v for v in vals if v not in nums]
            
            if nums:
                consolidated[col] = nums[0]
            else:
                # join text preserving single spaces
                consolidated[col] = ' '.join(' '.join(text).split())
        
        rows.append(consolidated)
    
    # Create body result
    body_res = pd.DataFrame(rows, columns=original_body_cols)
    
    # Repeat metadata rows for each consolidated row
    # If metadata is constant per group, take first row’s meta
    meta_rows = []
    for i, start in enumerate(starts):
        meta_rows.append(meta.iloc[start].to_dict())
    meta_res = pd.DataFrame(meta_rows)
    
    # Concatenate metadata + body
    result = pd.concat([meta_res.reset_index(drop=True), body_res.reset_index(drop=True)], axis=1)
    return result

def esic_data_cleaning(df):
    """
    Clean the Camelot extracted dataframe by:
    1. Removing the summary/total rows (first 2 rows)  
    2. Saving header information separately for later use
    3. Returning cleaned main dataframe and header dataframe
    """
    
    # Extract and remove summary rows (first 2 rows containing totals)
    summary_rows = df.iloc[0:2].copy()
    
    # Extract header rows (rows 2-3) for later use
    header_dataframe = df.iloc[2:4].copy()
    
    # Create main dataframe without summary rows (from row 2 onwards)
    main_dataframe = df.iloc[2:].copy().reset_index(drop=True)
    
    return main_dataframe, header_dataframe, summary_rows

def add_header_to_dataframe(final_df, header_df):
    """
    Add header dataframe on top of the final dataframe
    
    Parameters:
    final_df (pd.DataFrame): Your final processed dataframe
    header_df (pd.DataFrame): The header dataframe from the cleaning function
    
    Returns:
    pd.DataFrame: Combined dataframe with headers on top
    """
    
    # Ensure both dataframes have the same number of columns
    if len(final_df.columns) != len(header_df.columns):
        print(f"Warning: Column count mismatch - Final DF: {len(final_df.columns)}, Header DF: {len(header_df.columns)}")
        # Adjust columns to match the smaller one
        min_cols = min(len(final_df.columns), len(header_df.columns))
        final_df = final_df.iloc[:, :min_cols]
        header_df = header_df.iloc[:, :min_cols]
    
    # Reset index for both dataframes to ensure clean concatenation
    header_df_reset = header_df.reset_index(drop=True)
    final_df_reset = final_df.reset_index(drop=True)
    
    # Concatenate header on top of final dataframe
    combined_df = pd.concat([header_df_reset, final_df_reset], ignore_index=True)
    
    print("Header added to the final dataframe.")
    
    return combined_df

def add_metadata_columns(df, pdf_filename, page_number):
    # Make a copy to avoid modifying the original dataframe
    df_with_metadata = df.copy()
    
    # Add filename column at the beginning
    df_with_metadata.insert(0, 'PDF_File', pdf_filename)
    
    # Add page number column after filename
    df_with_metadata.insert(1, 'Page_Number', page_number)
    
    return df_with_metadata


table_reports = []
# pdf_file = "NARENDRA H PATEL ESI JULY.2024.pdf"
pdf_file = "xyz.pdf"
# pdf_file = "sample3.pdf"

#This is just filename not path
pdf_filename = os.path.basename(pdf_file)
    
tables = camelot.read_pdf(pdf_file, pages="all", flavor="stream") 
# tables = camelot.read_pdf(pdf_file, pages="all", flavor="lattice") 
# tables = camelot.read_pdf(pdf_file, pages="all", process_background=True) 
print(f"{len(tables)} tables found")

metadata_tables = []

for index, table in enumerate(tables):
    page_number = table.page 
    report = table.parsing_report
    table_reports.append({
        'index' : index + 1,
        'accuracy' : report.get('accuracy', 'N/A')
    })
    df_with_metadata = add_metadata_columns(table.df, pdf_filename, page_number)
    metadata_tables.append(df_with_metadata)


final_df = pd.concat([table for table in metadata_tables], ignore_index=True)


# final_df = remove_duplicate_headers_largest_content(final_df)
final_df = remove_duplicate_headers_in_body(final_df, metadata_cols=2)

final_df, header_df, removed_summary = esic_data_cleaning(final_df)

print(f"Header Dataframe: \n {header_df}")
print(f"removed Dataframe: \n {removed_summary}")

final_df = consolidate_dataframe_rows(final_df, metadata_cols=2)

# Assume header_df is your header DataFrame of shape (2, N)
# and its first two columns are 'PDF_File' and 'Page_Number'

# Replace the first cell of row 0 with 'PDF_File' and clear row 1
header_df.iat[0, 0] = 'PDF_File'
header_df.iat[1, 0] = ''

# Replace the second cell of row 0 with 'Page_Number' and clear row 1
header_df.iat[0, 1] = 'Page_Number'
header_df.iat[1, 1] = ''
final_df = add_header_to_dataframe(final_df, header_df)

print(f"Final Dataframe: \n {final_df}")

final_df.to_excel('extracted_tables.xlsx', index=False, header=False)

# xyz = tables[4].df

# xyz.to_excel('extracted_table_4.xlsx', index=False, header=False)
