import pdfplumber
import pandas as pd
import re
from datetime import datetime

def extract_bob_bank_statement_complete(pdf_path, output_excel_path="bob_statement_complete.xlsx"):
    """
    Complete Bank of Baroda statement extractor with exact PDF column structure.

    Column Structure: DATE | NARRATION | CHQ NO | WITHDRAWAL(DR) | DEPOSIT(CR) | BALANCE(INR)

    Features:
    - Uses date as transaction separator (handles multi-line narrations)
    - Only ONE of WITHDRAWAL(DR) or DEPOSIT(CR) has value per transaction
    - CHQ NO column always empty (position 2 as per PDF)
    - Always extracts BALANCE(INR)
    - Removes headers/metadata that appear only on first page
    - Handles empty columns and column shifting issues

    Args:
        pdf_path (str): Path to BOB PDF statement
        output_excel_path (str): Path for output Excel file

    Returns:
        pandas.DataFrame: Clean extracted transaction data with proper column structure
    """

    print(f"Processing BOB Statement: {pdf_path}")

    with pdfplumber.open(pdf_path) as pdf:
        print(f"Total pages: {len(pdf.pages)}")

        # Extract text from all pages
        all_text = ""
        for page_num, page in enumerate(pdf.pages, 1):
            page_text = page.extract_text()
            if page_text:
                all_text += page_text + "\n"
                print(f"✓ Page {page_num} processed")

        # Filter out metadata, headers, and irrelevant lines
        lines = all_text.split('\n')
        filtered_lines = []

        # Patterns to skip (metadata, headers, footers)
        skip_patterns = [
            r'PRIYANSHU MANPOWER SERVICES',
            r'Account.*554XXXXXXXX070', 
            r'DATE.*NARRATION.*CHQ\.NO\..*WITHDRAWAL.*DEPOSIT.*BALANCE',
            r'Customer Id:', r'Branch Name:', r'IFSC Code:', r'MICR Code:',
            r'Statement Period', r'Your Account Statement', r'Statement of transactions',
            r'Date and Time:', r'Contact-Us@', r'\*This is computer-generated',
            r'Page \d+ of \d+', r'A\*B\*K\* N\*G\*R', r'O\*P S\*M\*A\* D\*N\*E\*H\*A\*',
            r'V\*D\*D', r'G\*J\*R\*T-3\*0\*0\*', r'I\*D\*A', r'^\s*$'
        ]

        for line in lines:
            line = line.strip()
            if line and not any(re.search(pattern, line, re.IGNORECASE) for pattern in skip_patterns):
                filtered_lines.append(line)

        print(f"✓ Filtered to {len(filtered_lines)} relevant lines")

        # Group lines into transaction blocks using date pattern
        date_pattern = r'^\d{2}/\d{2}/\d{4}'
        transaction_blocks = []
        current_block = []

        for line in filtered_lines:
            if re.match(date_pattern, line):
                # New transaction starts - save previous block
                if current_block:
                    transaction_blocks.append(current_block)
                current_block = [line]
            elif current_block:  # Add to current transaction
                current_block.append(line)

        # Don't forget the last block
        if current_block:
            transaction_blocks.append(current_block)

        print(f"✓ Identified {len(transaction_blocks)} transaction blocks")

        # Parse each transaction block
        transactions = []
        for i, block in enumerate(transaction_blocks, 1):
            parsed = parse_single_transaction_complete(block)
            if parsed:
                transactions.append(parsed)
            else:
                print(f"⚠️  Warning: Could not parse transaction {i}")

        print(f"✓ Successfully parsed {len(transactions)} transactions")

        # Create DataFrame with proper column order
        df = pd.DataFrame(transactions)

        # Ensure exact column order as per PDF: DATE | NARRATION | CHQ NO | WITHDRAWAL(DR) | DEPOSIT(CR) | BALANCE(INR)
        column_order = ['DATE', 'NARRATION', 'CHQ NO', 'WITHDRAWAL(DR)', 'DEPOSIT(CR)', 'BALANCE(INR)']
        df = df[column_order]

        # Validate data quality
        validate_extracted_data_complete(df)

        # Save to Excel
        df.to_excel(output_excel_path, index=False)
        print(f"✅ Data saved to: {output_excel_path}")

        return df

def parse_single_transaction_complete(block):
    """
    Parse a single transaction block into structured data with CHQ NO column.

    Args:
        block (list): Lines belonging to one transaction

    Returns:
        dict: Parsed transaction data with proper column structure
    """
    if not block:
        return None

    # Extract date from first line
    main_line = block[0]
    date_match = re.match(r'^(\d{2}/\d{2}/\d{4})', main_line)
    if not date_match:
        return None

    date = date_match.group(1)
    remaining_text = main_line[len(date):].strip()

    # Combine all lines for complete transaction text
    if len(block) > 1:
        full_text = remaining_text + ' ' + ' '.join(block[1:])
    else:
        full_text = remaining_text

    # Extract balance (always present, format: X,XXX.XXCr)
    balance_pattern = r'([\d,]+\.\d{2}Cr)'
    balance_matches = re.findall(balance_pattern, full_text)
    balance = balance_matches[0] if balance_matches else ""

    # Remove balance from text for further processing
    text_without_balance = full_text
    if balance:
        text_without_balance = re.sub(re.escape(balance), '', text_without_balance, count=1)

    # Extract all monetary amounts
    amount_pattern = r'\b([\d,]+\.\d{2})\b'
    amounts = re.findall(amount_pattern, text_without_balance)

    # Clean narration by removing amounts and bank codes
    narration = text_without_balance
    for amount in amounts:
        narration = re.sub(r'\b' + re.escape(amount) + r'\b', '', narration)

    # Remove common bank codes and clean up
    narration = re.sub(r'\b(000|ARB|CBA|BKI|5/EM|SBIN|CBIN0)\b', '', narration)
    narration = re.sub(r'\s+', ' ', narration).strip()

    # Determine transaction type (CRITICAL: Only ONE of withdrawal or deposit)
    withdrawal = ""
    deposit = ""

    if amounts:
        main_amount = amounts[0]  # Primary amount

        # Classification logic
        if 'Charges for PORD Customer Payment' in full_text:
            # Bank service charges = withdrawal
            withdrawal = main_amount

        elif any(keyword in full_text.upper() for keyword in ['UPI/', 'NEFT-', 'RTGS-', 'SALARY', 'EBANK:']):
            # Electronic receipts = deposit
            deposit = main_amount

        else:
            # Default to deposit for other transactions
            deposit = main_amount

    return {
        'DATE': date,
        'NARRATION': narration,
        'CHQ NO': '',  # Always empty for electronic transactions
        'WITHDRAWAL(DR)': withdrawal,
        'DEPOSIT(CR)': deposit, 
        'BALANCE(INR)': balance
    }

def validate_extracted_data_complete(df):
    """Validate the extracted data quality"""
    print("\n" + "="*60)
    print("DATA QUALITY VALIDATION")
    print("="*60)

    # Check column structure
    expected_columns = ['DATE', 'NARRATION', 'CHQ NO', 'WITHDRAWAL(DR)', 'DEPOSIT(CR)', 'BALANCE(INR)']
    actual_columns = list(df.columns)
    print(f"✅ Column structure matches PDF: {actual_columns == expected_columns}")
    print(f"✅ CHQ NO at position 2: {actual_columns[2] == 'CHQ NO'}")

    # Check mutual exclusivity of withdrawal/deposit
    both_filled = df[(df['WITHDRAWAL(DR)'] != '') & (df['DEPOSIT(CR)'] != '')]
    print(f"✅ Rows with BOTH withdrawal and deposit: {len(both_filled)} (should be 0)")

    # Check coverage
    neither_filled = df[(df['WITHDRAWAL(DR)'] == '') & (df['DEPOSIT(CR)'] == '')]
    print(f"✅ Rows with NEITHER withdrawal nor deposit: {len(neither_filled)} (should be 0)")

    # Balance coverage
    with_balance = df[df['BALANCE(INR)'] != '']
    print(f"✅ Rows with balance: {len(with_balance)}/{len(df)} ({len(with_balance)/len(df)*100:.1f}%)")

    # CHQ NO column validation
    chq_filled = df[df['CHQ NO'] != '']
    print(f"✅ CHQ NO entries (should be 0): {len(chq_filled)}")

    # Transaction type breakdown
    withdrawals = df[df['WITHDRAWAL(DR)'] != '']
    deposits = df[df['DEPOSIT(CR)'] != '']
    print(f"✅ Withdrawals: {len(withdrawals)}")
    print(f"✅ Deposits: {len(deposits)}")
    print(f"✅ Total: {len(withdrawals) + len(deposits)}")

def main():
    """Demonstration of usage"""

    # Process the statement
    pdf_file = "bs1.pdf"
    excel_file = "BOB_Statement_Complete_Final.xlsx"

    try:
        df = extract_bob_bank_statement_complete(pdf_file, excel_file)

        print("\n" + "="*70)
        print("🎉 EXTRACTION COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"📊 Total transactions: {len(df)}")
        print(f"📅 Date range: {df['DATE'].min()} to {df['DATE'].max()}")

        # Show column structure
        print(f"📋 Column structure: {' | '.join(df.columns)}")

        # Show sample data
        print("\n📋 SAMPLE EXTRACTED DATA:")
        print("-" * 80)
        sample_data = df[['DATE', 'NARRATION', 'CHQ NO', 'WITHDRAWAL(DR)', 'DEPOSIT(CR)', 'BALANCE(INR)']].head(5)
        for idx, row in sample_data.iterrows():
            wd = row['WITHDRAWAL(DR)'] if row['WITHDRAWAL(DR)'] else '---'
            dp = row['DEPOSIT(CR)'] if row['DEPOSIT(CR)'] else '---'
            chq = row['CHQ NO'] if row['CHQ NO'] else '---'
            narr = row['NARRATION'][:30] + '...' if len(row['NARRATION']) > 30 else row['NARRATION']
            print(f"{row['DATE']} | {narr:33} | {chq:>3} | {wd:>10} | {dp:>12} | {row['BALANCE(INR)']}")

        return df

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

if __name__ == "__main__":
    extracted_data = main()
