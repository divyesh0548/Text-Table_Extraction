import tabula
import pandas as pd
import re

def extract_bob_with_tabula(pdf_path, output_excel_path="bob_statement_tabula.xlsx"):
    """
    FINAL ATTEMPT using tabula-py - the gold standard for PDF table extraction
    """
    
    print(f"🎯 FINAL APPROACH: tabula-py (gold standard for PDF tables)")
    print(f"Processing: {pdf_path}")
    
    try:
        # Method 1: Extract all tables from all pages
        print(f"\n📊 Extracting tables with tabula-py...")
        
        # Extract tables with different options
        all_dfs = tabula.read_pdf(
            pdf_path, 
            pages='all',
            multiple_tables=True,
            pandas_options={'header': None}
        )
        
        print(f"Found {len(all_dfs)} table(s) across all pages")
        
        # Combine and process all tables
        all_transactions = []
        
        for i, df in enumerate(all_dfs):
            print(f"\n📋 Processing table {i+1}: {df.shape[0]} rows x {df.shape[1]} columns")
            
            if df.empty:
                continue
            
            # Show sample of raw table
            print("Sample raw table:")
            print(df.head(3).to_string())
            
            # Process this table
            processed_transactions = process_tabula_table(df, i+1)
            all_transactions.extend(processed_transactions)
        
        print(f"\n📈 Total transactions processed: {len(all_transactions)}")
        
        if all_transactions:
            # Create final DataFrame
            final_df = pd.DataFrame(all_transactions)
            
            # Ensure column structure
            column_order = ['DATE', 'NARRATION', 'CHQ NO', 'WITHDRAWAL(DR)', 'DEPOSIT(CR)', 'BALANCE(INR)']
            for col in column_order:
                if col not in final_df.columns:
                    final_df[col] = ''
            
            final_df = final_df[column_order]
            
            # Final cleanup
            final_df = cleanup_tabula_data(final_df)
            
            # Save to Excel
            final_df.to_excel(output_excel_path, index=False)
            print(f"✅ tabula-py extraction saved to: {output_excel_path}")
            
            return final_df
        else:
            print("❌ No valid transactions found")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ tabula-py extraction failed: {str(e)}")
        print("This might be because Java is not installed on the system")
        return pd.DataFrame()

def process_tabula_table(df, table_num):
    """Process a single table from tabula"""
    
    transactions = []
    
    for idx, row in df.iterrows():
        # Convert row to list, handling NaN values
        row_data = [str(cell) if pd.notna(cell) else '' for cell in row.values]
        
        # Look for date in first column or concatenated text
        row_text = ' '.join(row_data)
        
        if re.search(r'\d{2}/\d{2}/\d{4}', row_text):
            # This row contains a transaction
            parsed = parse_tabula_row(row_data, row_text)
            if parsed:
                transactions.append(parsed)
    
    print(f"  ✓ Extracted {len(transactions)} transactions from table {table_num}")
    return transactions

def parse_tabula_row(row_data, row_text):
    """Parse a row from tabula extraction"""
    
    # Extract date
    date_match = re.search(r'(\d{2}/\d{2}/\d{4})', row_text)
    if not date_match:
        return None
    
    date = date_match.group(1)
    
    # Extract balance (ends with Cr)
    balance_pattern = r'([\d,]+\.\d{2}Cr)'
    balance_matches = re.findall(balance_pattern, row_text)
    balance = balance_matches[-1] if balance_matches else ""
    
    # Remove balance and date from text
    clean_text = row_text.replace(date, '').replace(balance, '') if balance else row_text.replace(date, '')
    
    # Extract amounts
    amount_pattern = r'\b([\d,]+\.\d{2})\b'
    amounts = re.findall(amount_pattern, clean_text)
    
    # Clean narration
    narration = clean_text
    for amount in amounts:
        narration = re.sub(r'\b' + re.escape(amount) + r'\b', '', narration, count=1)
    narration = re.sub(r'\s+', ' ', narration).strip()
    
    # Determine withdrawal vs deposit
    withdrawal = ""
    deposit = ""
    
    if amounts:
        main_amount = amounts[0]
        
        if 'Charges for PORD Customer Payment' in row_text:
            withdrawal = main_amount
        elif any(keyword in row_text.upper() for keyword in ['UPI/', 'NEFT-', 'RTGS-', 'SALARY', 'EBANK:']):
            deposit = main_amount
        else:
            deposit = main_amount
    
    return {
        'DATE': date,
        'NARRATION': narration,
        'CHQ NO': '',
        'WITHDRAWAL(DR)': withdrawal,
        'DEPOSIT(CR)': deposit,
        'BALANCE(INR)': balance
    }

def cleanup_tabula_data(df):
    """Clean up tabula-extracted data"""
    
    print(f"\n🧹 Cleaning tabula data...")
    
    # Clean string columns
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip().replace('nan', '').replace('None', '')
    
    # Remove empty rows
    df = df[df['DATE'] != '']
    
    # Stats
    total = len(df)
    with_wd = len(df[df['WITHDRAWAL(DR)'] != ''])
    with_dp = len(df[df['DEPOSIT(CR)'] != ''])
    with_both = len(df[(df['WITHDRAWAL(DR)'] != '') & (df['DEPOSIT(CR)'] != '')])
    
    print(f"  📊 Tabula results:")
    print(f"     Total: {total}")
    print(f"     Withdrawals: {with_wd}")
    print(f"     Deposits: {with_dp}")
    print(f"     Both (should be 0): {with_both}")
    
    return df

# Try tabula-py approach
print("="*80)
print("🆕 FINAL ATTEMPT: tabula-py (Professional PDF Table Extractor)")
print("   (This is specifically designed for extracting tables from PDFs)")
print("="*80)

tabula_df = extract_bob_with_tabula("bs1.pdf", "bob_statement_TABULA.xlsx")

if not tabula_df.empty:
    print(f"\n{'='*80}")
    print("🔍 TABULA-PY RESULTS")
    print('='*80)
    
    # Show comparison with your original output
    sample = tabula_df.head(15)
    
    for idx, row in sample.iterrows():
        wd_status = "WD" if row['WITHDRAWAL(DR)'] else "DP" if row['DEPOSIT(CR)'] else "--"
        amount = row['WITHDRAWAL(DR)'] if row['WITHDRAWAL(DR)'] else row['DEPOSIT(CR)'] if row['DEPOSIT(CR)'] else "---"
        print(f"{row['DATE']} | {wd_status} | {amount:>10} | {row['NARRATION'][:45]}...")
    
    print(f"\n📁 Tabula file: bob_statement_TABULA.xlsx")
    print(f"🤔 Does this finally match the actual PDF column layout?")
    
else:
    print("❌ All approaches failed.")
    print("\n💡 HONEST ASSESSMENT:")
    print("The issue seems to be that this particular Bank of Baroda PDF has:")
    print("  1. Non-standard table structure")
    print("  2. Multi-line transactions with amounts on different lines")
    print("  3. Column alignment issues that are very difficult to parse programmatically")
    print("\n🛠️ POSSIBLE SOLUTIONS:")
    print("  1. Ask the bank for a CSV export instead of PDF")
    print("  2. Use a specialized PDF-to-Excel service like Adobe Acrobat")
    print("  3. Manual data entry (unfortunately)")
    print("  4. Try a different PDF processing tool like Camelot or pdfquery")
    
    # Let me try one final approach
    print("\n🔧 Let me try one final approach with raw text and manual parsing...")
    final_attempt_result = extract_bob_with_tabula("bs1.pdf")