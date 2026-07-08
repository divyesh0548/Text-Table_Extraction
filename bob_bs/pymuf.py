import fitz  # PyMuPDF
import pandas as pd
import re

def extract_bob_with_pymupdf(pdf_path, output_excel_path="bob_statement_pymupdf.xlsx"):
    """
    COMPLETELY NEW APPROACH using PyMuPDF for precise column positioning
    This will extract text with exact X,Y coordinates and properly align columns
    """
    
    print(f"🎯 NEW APPROACH: PyMuPDF with precise text positioning")
    print(f"Processing: {pdf_path}")
    
    # Open PDF with PyMuPDF
    doc = fitz.open(pdf_path)
    print(f"Total pages: {len(doc)}")
    
    # Step 1: Analyze first page to get column boundaries
    first_page = doc[0]
    column_boundaries = analyze_header_with_pymupdf(first_page)
    
    print(f"\n📏 Column boundaries from header:")
    for col_name, bounds in column_boundaries.items():
        print(f"  {col_name:15} | x_start={bounds['x_start']:6.1f} x_end={bounds['x_end']:6.1f} width={bounds['x_end']-bounds['x_start']:5.1f}")
    
    # Step 2: Extract data from all pages using these precise boundaries
    all_transactions = []
    
    for page_num, page in enumerate(doc):
        print(f"\n📄 Processing page {page_num + 1} with precise coordinates...")
        
        # Get all text with coordinates
        text_dict = page.get_text("dict")
        
        # Group text by lines (same y-coordinate)
        lines_data = group_text_by_lines(text_dict)
        
        transactions_on_page = []
        for line_data in lines_data:
            # Check if this line contains a date
            if has_date_pattern(line_data):
                # Extract data using column boundaries
                transaction = extract_transaction_by_coordinates(line_data, column_boundaries)
                if transaction:
                    transactions_on_page.append(transaction)
        
        print(f"  ✓ Extracted {len(transactions_on_page)} transactions")
        all_transactions.extend(transactions_on_page)
    
    doc.close()
    
    print(f"\n📈 Total transactions extracted: {len(all_transactions)}")
    
    if all_transactions:
        # Create DataFrame
        df = pd.DataFrame(all_transactions)
        
        # Ensure column order
        column_order = ['DATE', 'NARRATION', 'CHQ NO', 'WITHDRAWAL(DR)', 'DEPOSIT(CR)', 'BALANCE(INR)']
        for col in column_order:
            if col not in df.columns:
                df[col] = ''
        df = df[column_order]
        
        # Clean data
        df = clean_pymupdf_data(df)
        
        # Save to Excel
        df.to_excel(output_excel_path, index=False)
        print(f"✅ PyMuPDF extraction saved to: {output_excel_path}")
        
        return df
    else:
        print("❌ No transactions found with PyMuPDF approach")
        return pd.DataFrame()

def analyze_header_with_pymupdf(first_page):
    """Analyze header to get precise column boundaries using PyMuPDF"""
    
    text_dict = first_page.get_text("dict")
    
    # Look for header text
    header_keywords = ['DATE', 'NARRATION', 'CHQ.NO.', 'WITHDRAWAL(DR)', 'DEPOSIT(CR)', 'BALANCE(INR)']
    header_positions = {}
    
    # Extract header positions
    for block in text_dict["blocks"]:
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if text in header_keywords:
                        header_positions[text] = {
                            'x_start': span['bbox'][0],
                            'x_end': span['bbox'][2],
                            'y': span['bbox'][1]
                        }
    
    # Create column boundaries
    column_boundaries = {}
    
    if len(header_positions) >= 4:  # At least some headers found
        # Sort by x position
        sorted_headers = sorted(header_positions.items(), key=lambda x: x[1]['x_start'])
        
        for i, (header_name, pos) in enumerate(sorted_headers):
            # Calculate column boundaries
            x_start = pos['x_start'] - 10  # Small buffer
            
            if i < len(sorted_headers) - 1:
                # Not the last column - end where next column starts
                next_pos = sorted_headers[i + 1][1]
                x_end = next_pos['x_start'] - 5
            else:
                # Last column - extend to page width
                x_end = 600  # Approximate page width
            
            # Map header names to our column names
            if header_name == 'CHQ.NO.':
                column_boundaries['CHQ NO'] = {'x_start': x_start, 'x_end': x_end}
            elif header_name == 'WITHDRAWAL(DR)':
                column_boundaries['WITHDRAWAL(DR)'] = {'x_start': x_start, 'x_end': x_end}
            elif header_name == 'DEPOSIT(CR)':
                column_boundaries['DEPOSIT(CR)'] = {'x_start': x_start, 'x_end': x_end}
            elif header_name == 'BALANCE(INR)':
                column_boundaries['BALANCE(INR)'] = {'x_start': x_start, 'x_end': x_end}
            else:
                column_boundaries[header_name] = {'x_start': x_start, 'x_end': x_end}
    else:
        # Fallback boundaries if header detection fails
        column_boundaries = {
            'DATE': {'x_start': 10, 'x_end': 90},
            'NARRATION': {'x_start': 90, 'x_end': 360},
            'CHQ NO': {'x_start': 360, 'x_end': 450},
            'WITHDRAWAL(DR)': {'x_start': 450, 'x_end': 530},
            'DEPOSIT(CR)': {'x_start': 530, 'x_end': 610},
            'BALANCE(INR)': {'x_start': 610, 'x_end': 700}
        }
    
    return column_boundaries

def group_text_by_lines(text_dict):
    """Group text elements by line (y-coordinate)"""
    
    lines_data = {}
    
    for block in text_dict["blocks"]:
        if "lines" in block:
            for line in block["lines"]:
                y_coord = round(line["bbox"][1], 1)  # Round y-coordinate
                
                if y_coord not in lines_data:
                    lines_data[y_coord] = []
                
                # Add all spans in this line
                for span in line["spans"]:
                    lines_data[y_coord].append({
                        'text': span['text'],
                        'x_start': span['bbox'][0],
                        'x_end': span['bbox'][2],
                        'y': span['bbox'][1]
                    })
    
    # Sort by y-coordinate (top to bottom)
    return [lines_data[y] for y in sorted(lines_data.keys(), reverse=True)]

def has_date_pattern(line_data):
    """Check if line contains a date pattern"""
    if not line_data:
        return False
    
    # Concatenate all text in the line
    line_text = ''.join([item['text'] for item in line_data])
    return bool(re.match(r'^\d{2}/\d{2}/\d{4}', line_text.strip()))

def extract_transaction_by_coordinates(line_data, column_boundaries):
    """Extract transaction data using precise coordinates"""
    
    # Sort elements by x-position
    sorted_elements = sorted(line_data, key=lambda x: x['x_start'])
    
    # Initialize transaction data
    transaction = {
        'DATE': '',
        'NARRATION': '',
        'CHQ NO': '',
        'WITHDRAWAL(DR)': '',
        'DEPOSIT(CR)': '',
        'BALANCE(INR)': ''
    }
    
    # Assign text to columns based on x-position
    for element in sorted_elements:
        text = element['text'].strip()
        x_pos = element['x_start']
        
        # Find which column this text belongs to
        for col_name, bounds in column_boundaries.items():
            if bounds['x_start'] <= x_pos <= bounds['x_end']:
                if transaction[col_name]:  # Column already has text
                    transaction[col_name] += ' ' + text
                else:
                    transaction[col_name] = text
                break
    
    # Validate - must have a date to be a valid transaction
    if transaction['DATE'] and re.match(r'^\d{2}/\d{2}/\d{4}', transaction['DATE']):
        return transaction
    else:
        return None

def clean_pymupdf_data(df):
    """Clean the extracted data"""
    
    print(f"\n🧹 Cleaning PyMuPDF extracted data...")
    
    # Clean all string columns
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip().replace('nan', '').replace('None', '')
    
    # Remove empty rows
    df = df[df['DATE'] != '']
    
    # Validation
    total = len(df)
    with_wd = len(df[df['WITHDRAWAL(DR)'] != ''])
    with_dp = len(df[df['DEPOSIT(CR)'] != ''])
    with_both = len(df[(df['WITHDRAWAL(DR)'] != '') & (df['DEPOSIT(CR)'] != '')])
    
    print(f"  📊 Cleaned data stats:")
    print(f"     Total: {total}")
    print(f"     With withdrawal: {with_wd}")
    print(f"     With deposit: {with_dp}") 
    print(f"     With both: {with_both}")
    
    return df

# Run PyMuPDF extraction
print("="*80)
print("🆕 TRYING COMPLETELY DIFFERENT APPROACH: PyMuPDF")
print("   (This should handle precise text positioning much better)")
print("="*80)

pymupdf_df = extract_bob_with_pymupdf("bs1.pdf", "bob_statement_PyMuPDF.xlsx")

if not pymupdf_df.empty:
    print(f"\n{'='*80}")
    print("🔍 PyMuPDF RESULTS - CHECKING COLUMN ALIGNMENT")
    print('='*80)
    
    # Show first 10 rows to compare with your output
    sample = pymupdf_df.head(10)
    print("DATE       | NARRATION                            | CHQ NO | WITHDRAWAL(DR) | DEPOSIT(CR)   | BALANCE(INR)")
    print("-" * 115)
    
    for idx, row in sample.iterrows():
        date = row['DATE'][:10]  # First 10 chars
        narration = (row['NARRATION'][:35] + '...') if len(row['NARRATION']) > 35 else row['NARRATION']
        chq_no = row['CHQ NO'][:6] if row['CHQ NO'] else '---'
        withdrawal = row['WITHDRAWAL(DR)'][:12] if row['WITHDRAWAL(DR)'] else '---'
        deposit = row['DEPOSIT(CR)'][:12] if row['DEPOSIT(CR)'] else '---'
        balance = row['BALANCE(INR)'][:15] if row['BALANCE(INR)'] else '---'
        
        print(f"{date} | {narration:<35} | {chq_no:>6} | {withdrawal:>12} | {deposit:>13} | {balance}")
    
    print(f"\n📁 PyMuPDF file saved: bob_statement_PyMuPDF.xlsx")
    print(f"🤔 Does this look better than the previous results?")
else:
    print("❌ PyMuPDF approach also failed. Let me try one more approach...")