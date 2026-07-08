
import pdfplumber
import pandas as pd
import re
import os

def extract_bank_statement_table(pdf_path, excel_output_path):
    """Extract with cross-page transaction handling"""
    print(f"📄 Opening PDF: {pdf_path}")

    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        return None

    transactions = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            print(f"✅ PDF opened. Pages: {len(pdf.pages)}")

            # Extract clean text from all pages
            full_text = extract_clean_text_from_pdf(pdf)

            # Parse transactions with cross-page handling
            transactions = parse_cross_page_transactions(full_text)

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

    if not transactions:
        print("❌ No transactions found!")
        return None

    print(f"✅ Extracted {len(transactions)} clean transactions")

    df = create_clean_dataframe(transactions)
    save_to_excel_clean(df, excel_output_path)

    return df

def extract_clean_text_from_pdf(pdf):
    """Extract text from PDF with page markers for better processing"""
    full_text = ""

    for page_num, page in enumerate(pdf.pages):
        page_text = page.extract_text()
        if page_text:
            # Add page marker but clean the text
            clean_page_text = clean_page_text_content(page_text)
            full_text += f"\n--- PAGE {page_num + 1} ---\n" + clean_page_text + "\n"

    return full_text

def clean_page_text_content(page_text):
    """Clean page text to remove headers, footers, and metadata"""
    lines = page_text.split('\n')
    clean_lines = []

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Skip page metadata and headers/footers
        if should_skip_line(line):
            continue

        clean_lines.append(line)

    return '\n'.join(clean_lines)

def should_skip_line(line):
    """Determine if a line should be skipped as metadata"""
    skip_patterns = [
        # Page numbers and headers
        r'^Page\s*No\.?\s*:',
        r'^M/S\.',
        r'^TF-\d+',
        r'^ATLADARA',
        r'^VADODARA',
        r'^GUJARAT\s*INDIA',

        # Account information blocks
        r'^JOINT\s*HOLDERS\s*:',
        r'^Nomination\s*:',
        r'^Statement\s*of\s*account',
        r'^From\s*:',
        r'^To\s*:',
        r'^Account\s*Branch\s*:',
        r'^Address\s*:',
        r'^City\s*:',
        r'^State\s*:',
        r'^Phone\s*no\.\s*:',
        r'^Email\s*:',
        r'^Cust\s*ID\s*:',
        r'^Account\s*No\s*:',
        r'^A/C\s*Open\s*Date\s*:',
        r'^Account\s*Status\s*:',
        r'^RTGS/NEFT\s*IFSC:',
        r'^MICR\s*:',
        r'^Branch\s*Code\s*:',
        r'^Product\s*Code\s*:',
        r'^OD\s*Limit\s*:',
        r'^Currency\s*:',
        r'^Preferred\s*Customer',

        # Legal text and disclaimers
        r'^HDFC\s*BANK\s*LIMITED',
        r'^\*Closing\s*balance\s*includes',
        r'^Contents\s*of\s*this\s*statement',
        r'^State\s*account\s*branch\s*GSTN',
        r'^HDFC\s*Bank\s*GSTIN\s*number',
        r'^Registered\s*Office\s*Address',
        r'^This\s*is\s*a\s*computer\s*generated',
        r'^does\s*not\s*require\s*signature',

        # Summary section
        r'^STATEMENT\s*SUMMARY\s*:-?',
        r'^Opening\s*Balance',
        r'^Dr\s*Count',
        r'^Cr\s*Count',
        r'^Debits',
        r'^Credits',
        r'^Closing\s*Bal',
        r'^Generated\s*On:',
        r'^Generated\s*By:',
        r'^Requesting\s*Branch\s*Code:',

        # URLs and technical info
        r'^https?://',
        r'^www\.',
        r'@',

        # Very long repetitive text (likely metadata)
        r'.{200,}'  # Lines longer than 200 chars are likely metadata
    ]

    return any(re.search(pattern, line, re.IGNORECASE) for pattern in skip_patterns)

def parse_cross_page_transactions(text):
    """Parse transactions with strict date boundaries and cross-page handling"""
    lines = text.split('\n')
    transactions = []
    current_transaction = None

    print("\n🔄 Processing transactions with cross-page handling...")

    for line in lines:
        line = line.strip()

        # Skip empty lines and page markers
        if not line or line.startswith('--- PAGE'):
            continue

        # Check for date at start of line (strict boundary)
        date_match = re.match(r'^(\d{2}/\d{2}/\d{2,4})\s+(.*)$', line)

        if date_match:
            # Process previous transaction before starting new one
            if current_transaction:
                processed = process_clean_transaction(current_transaction)
                if processed and is_valid_transaction(processed):
                    transactions.append(processed)
                    print(f"   ✓ {processed['Date']}: {processed['Narration'][:50]}... | Ref: {processed['Chq_Ref_No']}")

            # Start new transaction
            date = date_match.group(1)
            remainder = date_match.group(2).strip()

            current_transaction = {
                'date': date,
                'content_lines': []
            }

            # Add remainder if it exists and is not metadata
            if remainder and not should_skip_line(remainder):
                current_transaction['content_lines'].append(remainder)

        elif current_transaction:
            # This is a continuation line - but filter out metadata
            if not should_skip_line(line):
                # Additional filter for obvious metadata in continuation lines
                if not is_metadata_continuation(line):
                    current_transaction['content_lines'].append(line)

    # Process the last transaction
    if current_transaction:
        processed = process_clean_transaction(current_transaction)
        if processed and is_valid_transaction(processed):
            transactions.append(processed)
            print(f"   ✓ {processed['Date']}: {processed['Narration'][:50]}... | Ref: {processed['Chq_Ref_No']}")

    return transactions

def is_metadata_continuation(line):
    """Additional check for metadata in continuation lines"""
    metadata_indicators = [
        'HDFC', 'Bank', 'Limited', 'statement', 'account', 'branch',
        'address', 'phone', 'email', 'GSTIN', 'IFSC', 'MICR',
        'generated', 'signature', 'Mumbai', 'Marg', 'Lower', 'Parel',
        'TF-44', 'SAMANVAY', 'STATUS', 'ATLADARA', 'BHAYLI',
        'MIDWAY', 'HEIGHTS', 'TILAK', 'ROAD', 'SSG', 'HOSPITAL',
        'PANCHMUKHI', 'VADODARA', 'GUJARAT', 'registration'
    ]

    # If line contains multiple metadata indicators, it's likely metadata
    indicator_count = sum(1 for indicator in metadata_indicators 
                         if indicator.lower() in line.lower())

    return indicator_count >= 3

def process_clean_transaction(trans_data):
    """Process transaction with clean content filtering"""
    date = trans_data['date']
    lines = trans_data['content_lines']

    if not lines:
        return None

    # Filter and clean the content lines
    clean_lines = []
    for line in lines:
        # Additional cleaning for transaction content
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

    # Parse the clean content
    parse_clean_content(transaction, full_content)

    return transaction

def clean_transaction_line(line):
    """Clean individual transaction lines"""
    # Remove extra spaces
    line = re.sub(r'\s+', ' ', line).strip()

    # Remove obvious metadata fragments
    metadata_fragments = [
        r'HDFC\s*BANK\s*LIMITED',
        r'\*Closing\s*balance\s*includes[^.]*\.',
        r'Contents\s*of\s*this\s*statement[^.]*\.',
        r'State\s*account\s*branch\s*GSTN:[A-Z0-9]+',
        r'HDFC\s*Bank\s*GSTIN[^.]*\.',
        r'Registered\s*Office\s*Address:[^.]*Mumbai\s*\d+',
        r'https?://[^\s]+',
        r'Page\s*No\.?\s*:\s*\d+',
        r'Account\s*Branch\s*:\s*[A-Z]+',
        r'Phone\s*no\.?\s*:\s*[\d.e+-]+',
        r'Email\s*:\s*[^\s]+',
        r'Cust\s*ID\s*:\s*\d+',
        r'Account\s*No\s*:\s*[\d.e+-]+',
    ]

    for pattern in metadata_fragments:
        line = re.sub(pattern, '', line, flags=re.IGNORECASE)

    # Clean up multiple spaces again
    line = re.sub(r'\s+', ' ', line).strip()

    return line if len(line) > 5 else ''  # Ignore very short fragments

def parse_clean_content(transaction, content):
    """Parse clean content for transaction components"""
    if not content.strip():
        return

    tokens = content.split()

    amounts = []
    refs = []
    narration_parts = []

    for token in tokens:
        if is_amount_token(token):
            amounts.append(token)
        elif is_reference_token(token):
            # Handle scientific notation
            if is_scientific_notation(token):
                converted = convert_scientific_to_proper_format(token)
                refs.append(converted)
            else:
                refs.append(token)
        else:
            narration_parts.append(token)

    # Build clean narration
    narration = ' '.join(narration_parts).strip()

    # Additional narration cleaning
    narration = clean_narration_text(narration)

    transaction['Narration'] = narration

    if refs:
        transaction['Chq_Ref_No'] = refs[0]

    if amounts:
        transaction['Closing_Balance'] = amounts[-1]

        if len(amounts) >= 2:
            trans_amount = amounts[-2]

            if 'CR-' in narration.upper() or 'RTGS CR' in narration.upper():
                transaction['Deposit_Amount'] = trans_amount
                transaction['Withdrawal_Amount'] = ''
            else:
                transaction['Withdrawal_Amount'] = trans_amount
                transaction['Deposit_Amount'] = ''

def clean_narration_text(narration):
    """Final cleaning of narration text"""
    # Remove common metadata phrases that might slip through
    cleanup_patterns = [
        r'\b(?:HDFC|Bank|Limited|Statement|Account|Branch)\b',
        r'\b(?:Address|Phone|Email|GSTIN|IFSC|MICR)\b',
        r'\b(?:Generated|Signature|Mumbai|Parel|Marg)\b',
        r'\b(?:TF-44|SAMANVAY|STATUS|ATLADARA|BHAYLI)\b',
        r'\b(?:MIDWAY|HEIGHTS|TILAK|ROAD|HOSPITAL)\b',
        r'\b(?:VADODARA|GUJARAT|INDIA)\b'
    ]

    for pattern in cleanup_patterns:
        narration = re.sub(pattern, '', narration, flags=re.IGNORECASE)

    # Clean up multiple spaces and return
    narration = re.sub(r'\s+', ' ', narration).strip()

    return narration

def is_valid_transaction(transaction):
    """Validate that this is a real transaction"""
    # Must have either narration or reference number
    if not transaction.get('Narration', '').strip() and not transaction.get('Chq_Ref_No', '').strip():
        return False

    # Must have some amount
    if not any([
        transaction.get('Withdrawal_Amount', ''),
        transaction.get('Deposit_Amount', ''),
        transaction.get('Closing_Balance', '')
    ]):
        return False

    # Narration shouldn't be too short (likely metadata fragment)
    narration = transaction.get('Narration', '').strip()
    if narration and len(narration) < 10:
        return False

    return True

def is_amount_token(token):
    """Check if token is amount"""
    return bool(re.match(r'^\d{1,3}(,\d{3})*(\.\d{2})?$', token))

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
    """Save clean DataFrame to Excel"""
    print(f"💾 Saving clean data to Excel: {excel_path}")

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
        widths = {'A': 12, 'B': 100, 'C': 20, 'D': 12, 'E': 18, 'F': 18, 'G': 18}
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

    print(f"✅ Successfully saved {len(df)} clean transactions")

def main():
    """Main function"""
    print("🏦 CROSS-PAGE TRANSACTION FIXED EXTRACTOR 🏦")
    print("=" * 75)
    print("✅ Handles transactions spanning multiple pages")
    print("✅ Uses dates as strict transaction boundaries")  
    print("✅ Filters out page headers, footers, and metadata")
    print("✅ Clean narrations without unnecessary data")
    print("=" * 75)

    pdf_file = "bs1.pdf"
    excel_file = "PP2_output.xlsx"

    print(f"📥 Input: {pdf_file}")
    print(f"📤 Output: {excel_file}\n")

    result = extract_bank_statement_table(pdf_file, excel_file)

    if result is not None:
        print("\n🎉 CROSS-PAGE HANDLING SUCCESS! 🎉")
        print("=" * 60)

        withdrawal_count = sum(1 for x in result['Withdrawal_Amount'] if x and str(x).strip())
        deposit_count = sum(1 for x in result['Deposit_Amount'] if x and str(x).strip())

        print(f"📊 Clean Results:")
        print(f"   Total transactions: {len(result)}")
        print(f"   Withdrawal entries: {withdrawal_count}")
        print(f"   Deposit entries: {deposit_count}")
        print(f"   Cross-page handling: ✅")
        print(f"   Metadata filtering: ✅")

        print(f"\n📋 Sample clean transactions:")
        print("-" * 90)

        for i, row in result.head(5).iterrows():
            withdrawal = row['Withdrawal_Amount'] if row['Withdrawal_Amount'] else '[blank]'
            deposit = row['Deposit_Amount'] if row['Deposit_Amount'] else '[blank]'

            print(f"{i+1}. {row['Date']}")
            print(f"   Narration: {row['Narration'][:80]}...")
            print(f"   Reference: {row['Chq_Ref_No']}")
            print(f"   W: {withdrawal} | D: {deposit}")
            print()

        return True
    else:
        print("❌ Failed!")
        return False

if __name__ == "__main__":
    main()
