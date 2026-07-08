import pdfplumber
import pandas as pd
import re
import os
from statement_period_extractor import extract_and_display_period

def extract_bank_statement_table(pdf_path, excel_output_path):
    """Universal HDFC extraction with fixed amount logic and generic metadata filtering"""
    print(f"📄 Opening PDF: {pdf_path}")

    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        return None

    transactions = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            print(f"✅ PDF opened. Pages: {len(pdf.pages)}")

            # Extract clean text with universal metadata filtering
            full_text = extract_universal_clean_text(pdf)

            # Parse transactions with fixed amount logic
            transactions = parse_transactions_with_fixed_amounts(full_text)

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

    if not transactions:
        print("❌ No transactions found!")
        return None

    print(f"✅ Extracted {len(transactions)} transactions with fixed amounts")

    df = create_clean_dataframe(transactions)
    save_to_excel_clean(df, excel_output_path)

    return df

def extract_universal_clean_text(pdf):
    """Extract text with universal metadata filtering (works for any HDFC PDF)"""
    full_text = ""

    for page_num, page in enumerate(pdf.pages):
        page_text = page.extract_text()
        if page_text:
            # Universal page cleaning (not PDF-specific)
            clean_page_text = universal_page_cleaning(page_text)
            full_text += f"\n--- PAGE {page_num + 1} ---\n" + clean_page_text + "\n"

    return full_text

def universal_page_cleaning(page_text):
    """Universal cleaning that works for any HDFC PDF"""
    lines = page_text.split('\n')
    clean_lines = []

    in_header_metadata = False
    in_footer_metadata = False

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Universal header detection: anything between "Page No" and "Statement of account" is metadata
        if 'Page No' in line:
            in_header_metadata = True
            continue

        if 'Statement of account' in line:
            in_header_metadata = False
            continue

        # Universal footer detection: common footer patterns
        if is_universal_footer_metadata(line):
            in_footer_metadata = True
            continue

        # If we're in header or footer metadata, skip
        if in_header_metadata or in_footer_metadata:
            continue

        # Reset footer flag if we hit a transaction-like line
        if re.match(r'^\d{2}/\d{2}/\d{2,4}', line):
            in_footer_metadata = False

        # Additional filtering for obvious metadata
        if not is_obvious_metadata_line(line):
            clean_lines.append(line)

    return '\n'.join(clean_lines)

def is_universal_footer_metadata(line):
    """Detect universal footer metadata patterns (common across HDFC PDFs)"""
    footer_patterns = [
        r'^HDFC\s*BANK\s*LIMITED',
        r'^\*Closing\s*balance\s*includes',
        r'^Contents\s*of\s*this\s*statement',
        r'^State\s*account\s*branch\s*GSTN',
        r'^HDFC\s*Bank\s*GSTIN\s*number',
        r'^Registered\s*Office\s*Address',
        r'^This\s*is\s*a\s*computer\s*generated',
        r'^does\s*not\s*require\s*signature',
        r'^STATEMENT\s*SUMMARY',
        r'^Opening\s*Balance',
        r'^Dr\s*Count',
        r'^Cr\s*Count',
        r'^Debits',
        r'^Credits',
        r'^Closing\s*Bal',
        r'^Generated\s*On:',
        r'^Generated\s*By:',
        r'^Requesting\s*Branch\s*Code:'
    ]

    return any(re.search(pattern, line, re.IGNORECASE) for pattern in footer_patterns)

def is_obvious_metadata_line(line):
    """Detect obviously metadata lines (universal patterns)"""
    # Very long lines are usually metadata
    if len(line) > 150:
        return True

    # Lines with URLs
    if 'http' in line.lower() or 'www.' in line.lower():
        return True

    # Lines that are just numbers (page numbers, etc.)
    if re.match(r'^\d{1,3}$', line):
        return True

    return False

def parse_transactions_with_fixed_amounts(text):
    """Parse transactions with FIXED amount assignment logic"""
    lines = text.split('\n')
    transactions = []
    current_transaction = None

    print("\n🔄 Processing with FIXED amount assignment logic...")

    for line in lines:
        line = line.strip()

        # Skip empty lines and page markers
        if not line or line.startswith('--- PAGE'):
            continue

        # Check for date at start of line (transaction boundary)
        date_match = re.match(r'^(\d{2}/\d{2}/\d{2,4})\s+(.*)$', line)

        if date_match:
            # Process previous transaction before starting new one
            if current_transaction:
                processed = process_transaction_with_fixed_amounts(current_transaction)
                if processed and validate_transaction_amounts(processed):
                    transactions.append(processed)
                    print(f"   ✓ {processed['Date']}: {processed['Narration'][:50]}... | W:{processed['Withdrawal_Amount']} | D:{processed['Deposit_Amount']} | B:{processed['Closing_Balance']}")

            # Start new transaction
            date = date_match.group(1)
            remainder = date_match.group(2).strip()

            current_transaction = {
                'date': date,
                'content_lines': []
            }

            # Add remainder if meaningful
            if remainder and not is_obvious_metadata_line(remainder):
                current_transaction['content_lines'].append(remainder)

        elif current_transaction:
            # Continuation line - filter carefully
            if not is_obvious_metadata_line(line) and not is_continuation_metadata(line):
                current_transaction['content_lines'].append(line)

    # Process the last transaction
    if current_transaction:
        processed = process_transaction_with_fixed_amounts(current_transaction)
        if processed and validate_transaction_amounts(processed):
            transactions.append(processed)
            print(f"   ✓ {processed['Date']}: {processed['Narration'][:50]}... | W:{processed['Withdrawal_Amount']} | D:{processed['Deposit_Amount']} | B:{processed['Closing_Balance']}")

    return transactions

def is_continuation_metadata(line):
    """Detect metadata in continuation lines (universal patterns)"""
    # Common metadata indicators that appear in continuation lines
    metadata_indicators = [
        'HDFC', 'Bank', 'Limited', 'statement', 'generated', 'signature',
        'Mumbai', 'Marg', 'Lower', 'Parel', 'GSTIN', 'IFSC', 'MICR'
    ]

    # Count indicators
    indicator_count = sum(1 for indicator in metadata_indicators 
                         if indicator.lower() in line.lower())

    # If too many indicators, it's likely metadata
    if indicator_count >= 2:
        return True

    # Specific problematic patterns
    problematic_patterns = [
        r'Page\s*No',
        r'not\s*require\s*signature',
        r'computer\s*generated',
        r'\d{6}\s*GUJARAT',
        r'GSTIN\s*number'
    ]

    return any(re.search(pattern, line, re.IGNORECASE) for pattern in problematic_patterns)

def process_transaction_with_fixed_amounts(trans_data):
    """Process transaction with FIXED amount assignment logic"""
    date = trans_data['date']
    lines = trans_data['content_lines']

    if not lines:
        return None

    # Clean content lines
    clean_lines = []
    for line in lines:
        cleaned_line = clean_transaction_line(line)
        if cleaned_line:
            clean_lines.append(cleaned_line)

    if not clean_lines:
        return None

    # Combine clean content
    full_content = ' '.join(clean_lines)

    transaction = {
        'Date': date,
        'Narration': '',
        'Chq_Ref_No': '',
        'Value_Date': date,
        'Withdrawal_Amount': '',
        'Deposit_Amount': '',
        'Closing_Balance': ''
    }

    # FIXED parsing logic
    parse_content_with_fixed_amounts(transaction, full_content)

    return transaction

def parse_content_with_fixed_amounts(transaction, content):
    """FIXED amount parsing that prevents closing balance from becoming withdrawal amount"""
    if not content.strip():
        return

    tokens = content.split()

    amounts = []
    refs = []
    narration_parts = []

    for token in tokens:
        if is_valid_amount_token(token):
            amounts.append(token)
        elif is_reference_token(token):
            if is_scientific_notation(token):
                converted = convert_scientific_to_proper_format(token)
                refs.append(converted)
            else:
                refs.append(token)
        else:
            # Only add to narration if not metadata
            if not is_token_metadata(token):
                narration_parts.append(token)

    # Build narration
    narration = ' '.join(narration_parts).strip()
    narration = clean_final_narration(narration)
    transaction['Narration'] = narration

    # Assign reference
    if refs:
        transaction['Chq_Ref_No'] = refs[0]

    # FIXED AMOUNT ASSIGNMENT LOGIC
    if amounts:
        # Filter out unreasonably small amounts that are likely page numbers
        valid_amounts = [amt for amt in amounts if is_reasonable_amount(amt)]

        if valid_amounts:
            # Last amount is always closing balance
            transaction['Closing_Balance'] = valid_amounts[-1]

            # If we have at least 2 amounts, second-to-last is transaction amount
            if len(valid_amounts) >= 2:
                transaction_amount = valid_amounts[-2]

                # Determine if it's deposit or withdrawal based on transaction type
                if is_deposit_transaction(narration):
                    transaction['Deposit_Amount'] = transaction_amount
                    transaction['Withdrawal_Amount'] = ''
                else:
                    transaction['Withdrawal_Amount'] = transaction_amount
                    transaction['Deposit_Amount'] = ''

            # If only one amount, it might be the closing balance from a cross-page transaction
            elif len(valid_amounts) == 1:
                # Check if this looks like a cross-page continuation
                if is_cross_page_continuation(narration):
                    # Don't assign any transaction amount, just keep the closing balance
                    pass

def is_valid_amount_token(token):
    """Enhanced amount token validation"""
    # Standard monetary format
    if not re.match(r'^\d{1,3}(,\d{3})*(\.\d{2})?$', token):
        return False

    try:
        amount = float(token.replace(',', ''))
        # Must be reasonable amount (not page numbers or small metadata numbers)
        return 0.50 <= amount <= 100000000  # 50 paisa to 10 crore
    except ValueError:
        return False

def is_reasonable_amount(amount_str):
    """Check if amount is reasonable for bank transactions"""
    try:
        amount = float(amount_str.replace(',', ''))
        # Avoid tiny amounts that are likely page numbers
        return amount >= 1.0  # At least 1 rupee
    except ValueError:
        return False

def is_deposit_transaction(narration):
    """Determine if transaction is a deposit"""
    deposit_indicators = [
        'RTGS CR', 'NEFT CR', 'CR-', 'REV-UPI', 'CREDIT', 'DEPOSIT'
    ]
    return any(indicator.upper() in narration.upper() for indicator in deposit_indicators)

def is_cross_page_continuation(narration):
    """Check if this looks like a cross-page continuation"""
    # Very short narrations with just fragments are likely cross-page issues
    return len(narration.strip()) < 20 or '.:' in narration

def clean_transaction_line(line):
    """Clean individual transaction line"""
    # Remove extra spaces
    line = re.sub(r'\s+', ' ', line).strip()

    # Remove metadata fragments
    metadata_fragments = [
        r'HDFC\s*BANK\s*LIMITED',
        r'\*Closing\s*balance[^.]*\.',
        r'Contents\s*of\s*this[^.]*\.',
        r'https?://[^\s]+',
        r'not\s*require\s*signature',
        r'computer\s*generated'
    ]

    for pattern in metadata_fragments:
        line = re.sub(pattern, ' ', line, flags=re.IGNORECASE)

    # Clean up spaces
    line = re.sub(r'\s+', ' ', line).strip()

    return line if len(line) > 3 else ''

def is_token_metadata(token):
    """Check if individual token is metadata"""
    metadata_tokens = {
        'HDFC', 'BANK', 'LIMITED', 'Page', 'No', 'statement', 'signature',
        'generated', 'computer', 'Mumbai', 'Parel', 'Marg'
    }
    return token.upper() in metadata_tokens

def clean_final_narration(narration):
    """Final narration cleaning"""
    # Remove remaining metadata words
    cleanup_words = [
        'HDFC', 'Bank', 'Limited', 'statement', 'signature', 'generated',
        'computer', 'Mumbai', 'Parel', 'Marg', 'Page'
    ]

    words = narration.split()
    clean_words = [word for word in words if word not in cleanup_words]

    result = ' '.join(clean_words).strip()

    # Remove common fragments
    result = re.sub(r'\s*\.:.*$', '', result)  # Remove .: and everything after
    result = re.sub(r'\s+', ' ', result).strip()

    return result

def validate_transaction_amounts(transaction):
    """Validate transaction with amount checks"""
    # Must have date and some content
    if not transaction.get('Date') or not transaction.get('Narration', '').strip():
        return False

    # Must have some amount
    if not any([
        transaction.get('Withdrawal_Amount'),
        transaction.get('Deposit_Amount'),
        transaction.get('Closing_Balance')
    ]):
        return False

    # Narration shouldn't be too short or too fragmented
    narration = transaction.get('Narration', '').strip()
    if len(narration) < 8:
        return False

    # Check that amounts make sense
    for field in ['Withdrawal_Amount', 'Deposit_Amount', 'Closing_Balance']:
        amount = transaction.get(field, '')
        if amount and not is_reasonable_amount(str(amount)):
            return False

    return True

# Keep existing helper functions
def is_reference_token(token):
    """Check if token is reference"""
    patterns = [
        r'^[A-Z]\d{8,}$',
        r'^[A-Z]+\d{8,}$',
        r'^\d{12,}$',
        r'^\d+\.\d+[eE][+-]?\d+$'
    ]
    return any(re.match(pattern, token) for pattern in patterns)

def is_scientific_notation(token):
    """Check for scientific notation"""
    return bool(re.match(r'^\d+\.\d+[eE][+-]?\d+$', token))

def convert_scientific_to_proper_format(sci_token):
    """Convert scientific notation to proper format"""
    try:
        num = int(float(sci_token))
        return str(num).zfill(16)
    except (ValueError, OverflowError):
        return sci_token

def create_clean_dataframe(transactions):
    """Create clean DataFrame"""
    df = pd.DataFrame(transactions)

    columns = ['Date', 'Narration', 'Chq_Ref_No', 'Value_Date',
              'Withdrawal_Amount', 'Deposit_Amount', 'Closing_Balance']

    for col in columns:
        if col not in df.columns:
            df[col] = ''

    df = df[columns]

    # Clean string columns
    for col in ['Date', 'Narration', 'Chq_Ref_No', 'Value_Date']:
        df[col] = df[col].astype(str).str.strip()

    return df

def save_to_excel_clean(df, excel_path):
    """Save to Excel with formatting"""
    print(f"💾 Saving universal clean data to Excel: {excel_path}")

    excel_df = df.copy()
    amount_cols = ['Withdrawal_Amount', 'Deposit_Amount', 'Closing_Balance']

    for col in amount_cols:
        excel_df[col] = excel_df[col].apply(
            lambda x: float(x.replace(',', '')) if x and str(x).strip() else None
        )

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        excel_df.to_excel(writer, sheet_name='Bank_Statement', index=False)

        worksheet = writer.sheets['Bank_Statement']

        # Set column widths
        widths = {'A': 12, 'B': 120, 'C': 20, 'D': 12, 'E': 18, 'F': 18, 'G': 18}
        for col, width in widths.items():
            worksheet.column_dimensions[col].width = width

        # Format headers
        for cell in worksheet[1]:
            cell.font = cell.font.copy(bold=True)

        # Format currency columns
        from openpyxl.styles import NamedStyle
        currency_style = NamedStyle(name='currency')
        currency_style.number_format = '#,##0.00'

        for col in ['E', 'F', 'G']:
            for row in range(2, len(excel_df) + 2):
                cell = worksheet[f'{col}{row}']
                if cell.value is not None:
                    cell.style = currency_style

        # Format reference column as text
        for row in range(2, len(excel_df) + 2):
            ref_cell = worksheet[f'C{row}']
            ref_cell.value = str(df.iloc[row-2]['Chq_Ref_No'])
            ref_cell.number_format = '@'

    print(f"✅ Successfully saved {len(df)} transactions with fixed amounts")

def main():
    """Main function"""
    print("🏦 UNIVERSAL HDFC BANK STATEMENT EXTRACTOR 🏦")
    print("=" * 85)
    print("✅ FIXED amount assignment (no more closing balance → withdrawal)")
    print("✅ Universal metadata detection (works with any HDFC PDF)")
    print("✅ Generic header detection: Page No → Statement of account")
    print("✅ Universal footer patterns for all HDFC statements")
    print("✅ Better cross-page transaction handling")
    print("=" * 85)

    pdf_file = "bs2.pdf"  # Change this to your PDF filename
    excel_file = "PP3_output.xlsx"

    print(f"📥 Input: {pdf_file}")
    print(f"📤 Output: {excel_file}\n")
    
    
    result = extract_bank_statement_table(pdf_file, excel_file)

    if result is not None:
        print("\n🎉 UNIVERSAL EXTRACTION SUCCESS! 🎉")
        print("=" * 75)

        withdrawal_count = sum(1 for x in result['Withdrawal_Amount'] if x and str(x).strip())
        deposit_count = sum(1 for x in result['Deposit_Amount'] if x and str(x).strip())

        print(f"📊 Universal Results:")
        print(f"   Total transactions: {len(result)}")
        print(f"   Withdrawal entries: {withdrawal_count}")
        print(f"   Deposit entries: {deposit_count}")
        print(f"   Amount assignment: FIXED ✅")
        print(f"   Universal metadata filtering: ✅")

        # Check for problematic small balances
        small_balances = sum(1 for x in result['Closing_Balance'] 
                           if x and float(str(x).replace(',', '')) < 10)
        print(f"   Small closing balances (<₹10): {small_balances}")

        print(f"\n📋 Sample transactions with FIXED amounts:")
        print("-" * 110)

        for i, row in result.head(3).iterrows():
            # withdrawal = f"₹{float(row['Withdrawal_Amount']):,.2f}" if row['Withdrawal_Amount'] else '[blank]'
            # deposit = f"₹{float(row['Deposit_Amount']):,.2f}" if row['Deposit_Amount'] else '[blank]'
            # balance = f"₹{float(row['Closing_Balance']):,.2f}" if row['Closing_Balance'] else '[blank]'
            
            withdrawal = f"₹{float(str(row['Withdrawal_Amount']).replace(',', '')):,.2f}" if row['Withdrawal_Amount'] else '[blank]'
            deposit = f"₹{float(str(row['Deposit_Amount']).replace(',', '')):,.2f}" if row['Deposit_Amount'] else '[blank]'
            balance = f"₹{float(str(row['Closing_Balance']).replace(',', '')):,.2f}" if row['Closing_Balance'] else '[blank]'


            print(f"{i+1}. {row['Date']}")
            print(f"   Narration: {row['Narration'][:80]}...")
            print(f"   Reference: {row['Chq_Ref_No']}")
            print(f"   W: {withdrawal} | D: {deposit} | B: {balance}")
            print()

        return True
    else:
        print("❌ Failed!")
        return False

if __name__ == "__main__":
    main()
