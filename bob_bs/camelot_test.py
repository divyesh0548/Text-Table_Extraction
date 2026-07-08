import pandas as pd
import camelot
from header_fixed import insert_digital_header_fixed_font
import re
import os
pd.set_option('display.max_rows', None)
# import pytesseract

# pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Divyesh Parmar\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

def remove_duplicate_headers(df):
    """
    Remove duplicate header rows from the dataframe.
    Keeps only the first occurrence of the header row.
    """
    
    if df.empty:
        return df
    
    print("🔄 Removing duplicate headers...")
    
    # Define header pattern - these are the expected header column names
    header_keywords = ['DATE', 'NARRATION', 'CHQ.NO.', 'CHQ NO', 'WITHDRAWAL(DR)', 'DEPOSIT(CR)', 'BALANCE(INR)']
    
    # Find all header rows
    header_indices = []
    first_header_idx = None
    
    for idx, row in df.iterrows():
        row_text = ' '.join(str(val) for val in row.values).upper()
        
        # Check if this row contains header keywords
        keyword_count = sum(1 for keyword in header_keywords if keyword in row_text)
        
        # If row contains 4+ header keywords, it's likely a header
        if keyword_count >= 4:
            header_indices.append(idx)
            if first_header_idx is None:
                first_header_idx = idx
    
    print(f"📊 Found {len(header_indices)} header rows at indices: {header_indices}")
    
    if len(header_indices) <= 1:
        print("✅ No duplicate headers found")
        return df
    
    # Remove duplicate headers (keep only the first one)
    rows_to_remove = header_indices[1:]  # Remove all except first
    
    print(f"🗑️ Removing duplicate headers at indices: {rows_to_remove}")
    
    # Create new dataframe without duplicate headers
    cleaned_df = df.drop(rows_to_remove).reset_index(drop=True)
    
    print(f"✅ Removed {len(rows_to_remove)} duplicate headers")
    print(f"   Original rows: {len(df)} → Final rows: {len(cleaned_df)}")
    
    return cleaned_df

def remove_duplicate_headers_strict(df):
    """
    Alternative version with stricter header detection.
    Only considers exact header pattern matches.
    """
    
    if df.empty:
        return df
    
    print("🔄 Removing duplicate headers (strict mode)...")
    
    # More strict header detection
    header_indices = []
    first_header_idx = None
    
    for idx, row in df.iterrows():
        row_text = ' '.join(str(val) for val in row.values)
        
        # Check for exact header pattern
        if (('DATE' in row_text and 'NARRATION' in row_text and 
             'WITHDRAWAL(DR)' in row_text and 'DEPOSIT(CR)' in row_text and 
             'BALANCE(INR)' in row_text) and
            ('CHQ.NO.' in row_text or 'CHQ NO' in row_text)):
            
            header_indices.append(idx)
            if first_header_idx is None:
                first_header_idx = idx
    
    print(f"📊 Found {len(header_indices)} header rows at indices: {header_indices}")
    
    if len(header_indices) <= 1:
        print("✅ No duplicate headers found")
        return df
    
    # Remove duplicate headers
    rows_to_remove = header_indices[1:]
    cleaned_df = df.drop(rows_to_remove).reset_index(drop=True)
    
    print(f"✅ Removed {len(rows_to_remove)} duplicate headers")
    print(f"   Original rows: {len(df)} → Final rows: {len(cleaned_df)}")
    
    return cleaned_df

def clean_bob_statement_with_spacing(df):
    """
    Clean BOB statement by:
    1. Preserving account information
    2. Adding empty line for clean look
    3. Removing content BEFORE header
    4. Keeping header and ALL transaction data intact
    """
    
    if df.empty:
        return df
    
    # Convert all columns to strings
    df = df.astype(str)
    
    # Step 1: Extract and preserve important dates
    preserved_info = []
    
    for idx, row in df.iterrows():
        row_text = ' '.join(row.values)
        
        # Preserve statement date and period
        if 'Your Account Statement as on' in row_text and 'Statement Period from' in row_text:
            preserved_info.append(row)
            break  # Only need this one row
    
    # Step 2: Find the header row (with DATE, NARRATION, etc.)
    header_row_idx = None
    
    for idx, row in df.iterrows():
        row_text = ' '.join(row.values)
        # Look for the transaction header
        if ('DATE' in row_text and 'NARRATION' in row_text and 
            'WITHDRAWAL' in row_text and 'DEPOSIT' in row_text and 'BALANCE' in row_text):
            header_row_idx = idx
            break
    
    if header_row_idx is None:
        print("❌ Transaction header not found")
        return df
    
    print(f"✅ Header found at row {header_row_idx}")
    
    # Step 3: Create cleaned dataframe with spacing
    cleaned_rows = []
    
    # Add preserved account information
    for info_row in preserved_info:
        cleaned_rows.append(info_row)
    
    # Add empty line for clean look
    empty_row = pd.Series([''] * len(df.columns))
    cleaned_rows.append(empty_row)
    
    # Add header and all subsequent data
    for idx in range(header_row_idx, len(df)):
        row = df.iloc[idx]
        row_text = ' '.join(row.values)
        
        # Skip only completely empty rows or page footers
        if (row_text.strip() and row_text.strip() != 'nan' and 
            'Page' not in row_text and 'Contact-Us' not in row_text and 
            'computer-generated' not in row_text):
            cleaned_rows.append(row)
    
    # Create new dataframe with cleaned rows
    if cleaned_rows:
        cleaned_df = pd.DataFrame(cleaned_rows)
        cleaned_df.reset_index(drop=True, inplace=True)
        
        print(f"✅ Cleaned with spacing! Original: {len(df)} rows → Clean: {len(cleaned_df)} rows")
        
        return cleaned_df
    
    else:
        print("❌ No valid data found")
        return pd.DataFrame()

def merge_multiline_transactions(df):
    """
    Merge multi-line transaction rows based on DATE column.
    Each new date indicates start of new transaction.
    """
    
    if df.empty:
        return df
    
    print("🔄 Merging multi-line transactions...")
    
    # Find header row
    header_row_idx = None
    for idx, row in df.iterrows():
        row_text = ' '.join(str(val) for val in row.values)
        if ('DATE' in row_text and 'NARRATION' in row_text and 
            'WITHDRAWAL' in row_text and 'DEPOSIT' in row_text):
            header_row_idx = idx
            break
    
    if header_row_idx is None:
        print("❌ Header not found for merging")
        return df
    
    # Preserve rows before and including header
    preserved_rows = []
    for idx in range(header_row_idx + 1):
        preserved_rows.append(df.iloc[idx])
    
    # Process transaction rows (after header)
    merged_transactions = []
    current_transaction = None
    
    for idx in range(header_row_idx + 1, len(df)):
        row = df.iloc[idx]
        first_cell = str(row.iloc[0]).strip()
        
        # Check if this row starts with a date (new transaction)
        if re.match(r'^\d{2}/\d{2}/\d{4}$', first_cell):
            # Save previous transaction if exists
            if current_transaction is not None:
                merged_transactions.append(current_transaction)
            
            # Start new transaction
            current_transaction = row.copy()
            
        else:
            # This is a continuation row - merge with current transaction
            if current_transaction is not None:
                for col_idx, cell_value in enumerate(row.values):
                    cell_str = str(cell_value).strip()
                    
                    if cell_str and cell_str != 'nan':
                        current_col_value = str(current_transaction.iloc[col_idx]).strip()
                        
                        if current_col_value and current_col_value != 'nan':
                            # Append to existing content with space
                            current_transaction.iloc[col_idx] = current_col_value + ' ' + cell_str
                        else:
                            # Set new content
                            current_transaction.iloc[col_idx] = cell_str
    
    # Don't forget the last transaction
    if current_transaction is not None:
        merged_transactions.append(current_transaction)
    
    # Combine all rows
    final_rows = preserved_rows + merged_transactions
    
    if final_rows:
        merged_df = pd.DataFrame(final_rows)
        merged_df.reset_index(drop=True, inplace=True)
        
        print(f"✅ Merged successfully!")
        print(f"   Original rows after header: {len(df) - header_row_idx - 1}")
        print(f"   Merged transactions: {len(merged_transactions)}")
        print(f"   Total final rows: {len(merged_df)}")
        
        return merged_df
    
    else:
        print("❌ No data to merge")
        return df


table_reports = []
pdf_file = "bs2.pdf"

insert_digital_header_fixed_font(pdf_file, f"{pdf_file}_header_fixed.pdf")

pdf_file = f"{pdf_file}_header_fixed.pdf"

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
final_df = remove_duplicate_headers(final_df)
pd.set_option('display.max_rows', None)

final_df = clean_bob_statement_with_spacing(final_df)

final_df = merge_multiline_transactions(final_df)

print(f"Final Dataframe : \n{final_df}")

# Apply it to your DataFrame

final_df.to_excel('extracted_tables.xlsx', index=False, header=False)

