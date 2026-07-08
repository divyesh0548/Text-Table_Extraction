import pdfplumber
import pandas as pd
import re
import os

def extract_bank_statement_table(pdf_path, excel_output_path):
    """Extract with enhanced cross-page transaction handling and metadata filtering"""
    print(f"📄 Opening PDF: {pdf_path}")

    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        return None

    transactions = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            print(f"✅ PDF opened. Pages: {len(pdf.pages)}")

            # Extract clean text from all pages with enhanced filtering
            full_text = extract_clean_text_from_pdf(pdf)

            # Parse transactions with enhanced cross-page handling
            transactions = parse_cross_page_transactions_enhanced(full_text)

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
    """Extract text from PDF with enhanced page markers and cleaning"""
    full_text = ""

    for page_num, page in enumerate(pdf.pages):
        page_text = page.extract_text()
        if page_text:
            # Enhanced page text cleaning
            clean_page_text = clean_page_text_content_enhanced(page_text)
            full_text += f"\n--- PAGE {page_num + 1} ---\n" + clean_page_text + "\n"

    return full_text

def clean_page_text_content_enhanced(page_text):
    """Enhanced cleaning of page text to remove headers, footers, and metadata"""
    lines = page_text.split('\n')
    clean_lines = []

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Enhanced metadata filtering
        if should_skip_line_enhanced(line):
            continue

        clean_lines.append(line)

    return '\n'.join(clean_lines)

def should_skip_line_enhanced(line):
    """Enhanced metadata detection with more comprehensive patterns"""
    skip_patterns = [
        # Page numbers and headers (enhanced)
        r'^Page\s*No\.?\s*:',
        r'^Page\s*No\.?\s*:\s*\d+',
        r'^M/S\.',

        # Company and address info (enhanced)
        r'^KIEARRA\s*MANPOWER',
        r'^JAY\s*MATAJI\s*MANPOWER',
        r'^TF-\d+',
        r'^AT\s*RAJ\s*FALIYU',
        r'^ATLADARA',
        r'^VADODARA',
        r'^BHARUCH',
        r'^JAMBUSAR',
        r'^GUJARAT\s*INDIA',
        r'^GROUND\s*FLOOR',
        r'^SURAJ\s*BUILDING',
        r'^OPP\.SWAMINARAYAN',
        r'^TANKARI\s*BHAGOLE',
        r'^KAVI\s*ROAD',

        # Account information blocks (enhanced)
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

        # Legal text and disclaimers (enhanced)
        r'^HDFC\s*BANK\s*LIMITED',
        r'^\*Closing\s*balance\s*includes',
        r'^Contents\s*of\s*this\s*statement',
        r'^State\s*account\s*branch\s*GSTN',
        r'^HDFC\s*Bank\s*GSTIN\s*number',
        r'^Registered\s*Office\s*Address',
        r'^This\s*is\s*a\s*computer\s*generated',
        r'^does\s*not\s*require\s*signature',

        # Summary section (enhanced)
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

        # Standalone numbers (likely page numbers or metadata)
        r'^\d{1,3}$',
        r'^\d{6,}$',  # Long standalone numbers

        # Address fragments that slip through
        r'^MIDWAY\s*HEIGHTS',
        r'^TIKKA\s*24/1',
        r'^LOKMANYA\s*TILAK',
        r'^OPP\s*SSG\s*HOSPITAL',
        r'^NEAR\s*PANCHMUKHI',
        r'^392150$',
        r'^390001$',
        r'^390012$',

        # Very long repetitive text (likely metadata)
        r'.{200,}'  # Lines longer than 200 chars are likely metadata
    ]

    return any(re.search(pattern, line, re.IGNORECASE) for pattern in skip_patterns)

def parse_cross_page_transactions_enhanced(text):
    """Enhanced parsing with better cross-page handling and metadata filtering"""
    lines = text.split('\n')
    transactions = []
    current_transaction = None

    print("\n🔄 Enhanced processing with better metadata filtering...")

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
                processed = process_clean_transaction_enhanced(current_transaction)
                if processed and is_valid_transaction_enhanced(processed):
                    transactions.append(processed)
                    print(f"   ✓ {processed['Date']}: {processed['Narration'][:50]}... | Ref: {processed['Chq_Ref_No']} | W:{processed['Withdrawal_Amount']} | D:{processed['Deposit_Amount']}")

            # Start new transaction
            date = date_match.group(1)
            remainder = date_match.group(2).strip()

            current_transaction = {
                'date': date,
                'content_lines': []
            }

            # Add remainder if it exists and is not metadata
            if remainder and not should_skip_line_enhanced(remainder):
                current_transaction['content_lines'].append(remainder)

        elif current_transaction:
            # This is a continuation line - apply enhanced filtering
            if not should_skip_line_enhanced(line):
                # Additional enhanced filter for metadata in continuation lines
                if not is_metadata_continuation_enhanced(line):
                    current_transaction['content_lines'].append(line)

    # Process the last transaction
    if current_transaction:
        processed = process_clean_transaction_enhanced(current_transaction)
        if processed and is_valid_transaction_enhanced(processed):
            transactions.append(processed)
            print(f"   ✓ {processed['Date']}: {processed['Narration'][:50]}... | Ref: {processed['Chq_Ref_No']} | W:{processed['Withdrawal_Amount']} | D:{processed['Deposit_Amount']}")

    return transactions

def is_metadata_continuation_enhanced(line):
    """Enhanced check for metadata in continuation lines"""
    # Check for specific metadata indicators
    metadata_indicators = [
        'HDFC', 'Bank', 'Limited', 'statement', 'account', 'branch',
        'address', 'phone', 'email', 'GSTIN', 'IFSC', 'MICR',
        'generated', 'signature', 'Mumbai', 'Marg', 'Lower', 'Parel',
        'TF-44', 'SAMANVAY', 'STATUS', 'ATLADARA', 'BHAYLI',
        'MIDWAY', 'HEIGHTS', 'TILAK', 'ROAD', 'SSG', 'HOSPITAL',
        'PANCHMUKHI', 'VADODARA', 'GUJARAT', 'JAMBUSAR', 'BHARUCH',
        'GROUND', 'FLOOR', 'SURAJ', 'BUILDING', 'SWAMINARAYAN',
        'TEMPLE', 'TANKARI', 'BHAGOLE', 'KAVI'
    ]

    # Count metadata indicators
    indicator_count = sum(1 for indicator in metadata_indicators 
                         if indicator.lower() in line.lower())

    # If line has too many metadata indicators, skip it
    if indicator_count >= 3:
        return True

    # Check for specific problematic patterns
    problematic_patterns = [
        r'Page\s*No\.',
        r'OPP\.SWAMINARAYAN',
        r'TANKARI\s*BHAGOLE',
        r'KAVI\s*ROAD',
        r'JAMBUSAR\s*392150',
        r'BHARUCH',
        r'OD\s*Limit\s*:',
        r'Phone\s*no\.',
        r'\d{6}\s*BHARUCH',
        r'this\s*statement\.',
        r'not\s*require\s*signature',
        r'GSTIN\s*number'
    ]

    return any(re.search(pattern, line, re.IGNORECASE) for pattern in problematic_patterns)

def process_clean_transaction_enhanced(trans_data):
    """Enhanced transaction processing with better content filtering"""
    date = trans_data['date']
    lines = trans_data['content_lines']

    if not lines:
        return None

    # Enhanced filtering and cleaning of content lines
    clean_lines = []
    for line in lines:
        # Enhanced cleaning for transaction content
        cleaned_line = clean_transaction_line_enhanced(line)
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

    # Enhanced parsing of the clean content
    parse_clean_content_enhanced(transaction, full_content)

    return transaction

def clean_transaction_line_enhanced(line):
    """Enhanced cleaning of individual transaction lines"""
    # Remove extra spaces
    line = re.sub(r'\s+', ' ', line).strip()

    # Enhanced removal of obvious metadata fragments
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
        r'this\s*statement\.',
        r'not\s*require\s*signature\.',
        r'OPP\.SWAMINARAYAN\s*TEMPLE',
        r'TANKARI\s*BHAGOLE,\s*KAVI\s*ROAD',
        r'JAMBUSAR\s*392150',
        r'BHARUCH',
        r'OD\s*Limit\s*:\s*[\d.]+',
        r'\d{6}\s*/\d{6,8}'
    ]

    for pattern in metadata_fragments:
        line = re.sub(pattern, ' ', line, flags=re.IGNORECASE)

    # Clean up multiple spaces again
    line = re.sub(r'\s+', ' ', line).strip()

    return line if len(line) > 5 else ''  # Ignore very short fragments

def parse_clean_content_enhanced(transaction, content):
    """Enhanced parsing of clean content for transaction components"""
    if not content.strip():
        return

    tokens = content.split()

    amounts = []
    refs = []
    narration_parts = []

    for token in tokens:
        if is_amount_token_enhanced(token):
            amounts.append(token)
        elif is_reference_token(token):
            # Handle scientific notation
            if is_scientific_notation(token):
                converted = convert_scientific_to_proper_format(token)
                refs.append(converted)
            else:
                refs.append(token)
        else:
            # Only add to narration if it's not a metadata fragment
            if not is_token_metadata_fragment(token):
                narration_parts.append(token)

    # Build clean narration with enhanced cleaning
    narration = ' '.join(narration_parts).strip()
    narration = clean_narration_text_enhanced(narration)

    transaction['Narration'] = narration

    if refs:
        transaction['Chq_Ref_No'] = refs[0]

    # Enhanced amount assignment with validation
    if amounts:
        # Validate amounts to avoid picking up metadata numbers
        valid_amounts = [amt for amt in amounts if is_valid_amount(amt)]

        if valid_amounts:
            transaction['Closing_Balance'] = valid_amounts[-1]

            if len(valid_amounts) >= 2:
                trans_amount = valid_amounts[-2]

                if 'CR-' in narration.upper() or 'RTGS CR' in narration.upper() or 'REV-' in narration.upper():
                    transaction['Deposit_Amount'] = trans_amount
                    transaction['Withdrawal_Amount'] = ''
                else:
                    transaction['Withdrawal_Amount'] = trans_amount
                    transaction['Deposit_Amount'] = ''

def is_amount_token_enhanced(token):
    """Enhanced amount token detection"""
    # Standard monetary format but with better validation
    if not re.match(r'^\d{1,3}(,\d{3})*(\.\d{2})?$', token):
        return False

    # Convert to float for validation
    try:
        amount = float(token.replace(',', ''))
        # Reasonable amount range - avoid very small numbers that are likely metadata
        return 0.01 <= amount <= 50000000  # Minimum 1 paisa, maximum 5 crore
    except ValueError:
        return False

def is_valid_amount(amount_str):
    """Validate that an amount is reasonable for a bank transaction"""
    try:
        amount = float(amount_str.replace(',', ''))
        # Avoid very small amounts that are likely page numbers or other metadata
        return amount >= 0.01  # At least 1 paisa
    except ValueError:
        return False

def is_token_metadata_fragment(token):
    """Check if individual token is likely metadata"""
    metadata_tokens = {
        'Page', 'No', 'Account', 'Branch', 'Address', 'City', 'State', 'Phone',
        'OD', 'Limit', 'Currency', 'Email', 'Cust', 'ID', 'HDFC', 'BANK',
        'LIMITED', 'JAMBUSAR', 'BHARUCH', 'GUJARAT', 'INDIA', 'GROUND', 'FLOOR',
        'SWAMINARAYAN', 'TEMPLE', 'TANKARI', 'BHAGOLE', 'KAVI', 'ROAD',
        'SURAJ', 'BUILDING', 'JAY', 'MATAJI', 'MANPOWER', 'SERVICE',
        'KIEARRA', 'MIDWAY', 'HEIGHTS', 'TILAK', 'SSG', 'HOSPITAL',
        'this', 'statement', 'signature', 'require'
    }

    return token.upper() in metadata_tokens

def clean_narration_text_enhanced(narration):
    """Enhanced final cleaning of narration text"""
    # Enhanced removal of common metadata phrases
    cleanup_patterns = [
        r'\b(?:HDFC|Bank|Limited|Statement|Account|Branch)\b',
        r'\b(?:Address|Phone|Email|GSTIN|IFSC|MICR)\b',
        r'\b(?:Generated|Signature|Mumbai|Parel|Marg)\b',
        r'\b(?:TF-44|SAMANVAY|STATUS|ATLADARA|BHAYLI)\b',
        r'\b(?:MIDWAY|HEIGHTS|TILAK|ROAD|HOSPITAL)\b',
        r'\b(?:VADODARA|GUJARAT|INDIA|JAMBUSAR|BHARUCH)\b',
        r'\b(?:GROUND|FLOOR|SURAJ|BUILDING)\b',
        r'\b(?:SWAMINARAYAN|TEMPLE|TANKARI|BHAGOLE|KAVI)\b',
        r'\b(?:this|statement|signature|require)\b',
        r'\b(?:Page|No|Limit|Currency)\b'
    ]

    for pattern in cleanup_patterns:
        narration = re.sub(pattern, '', narration, flags=re.IGNORECASE)

    # Clean up multiple spaces and return
    narration = re.sub(r'\s+', ' ', narration).strip()

    return narration

def is_valid_transaction_enhanced(transaction):
    """Enhanced validation for transactions"""
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

    # Check that amounts are reasonable
    for amt_field in ['Withdrawal_Amount', 'Deposit_Amount', 'Closing_Balance']:
        amt = transaction.get(amt_field, '')
        if amt and not is_valid_amount(str(amt)):
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
    """Save clean DataFrame to Excel with proper formatting"""
    print(f"💾 Saving enhanced clean data to Excel: {excel_path}")

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

    print(f"✅ Successfully saved {len(df)} enhanced clean transactions")

def main():
    """Main function"""
    print("🏦 ENHANCED HDFC BANK STATEMENT EXTRACTOR 🏦")
    print("=" * 80)
    print("✅ Enhanced metadata filtering to remove page noise")
    print("✅ Better cross-page transaction handling")
    print("✅ Improved amount validation to avoid metadata numbers")
    print("✅ Clean narrations without address/page fragments")
    print("=" * 80)

    pdf_file = "bs1.pdf"  # Change this to your PDF filename
    excel_file = "PP3_output.xlsx"

    print(f"📥 Input: {pdf_file}")
    print(f"📤 Output: {excel_file}\n")

    result = extract_bank_statement_table(pdf_file, excel_file)

    if result is not None:
        print("\n🎉 ENHANCED EXTRACTION SUCCESS! 🎉")
        print("=" * 70)

        withdrawal_count = sum(1 for x in result['Withdrawal_Amount'] if x and str(x).strip())
        deposit_count = sum(1 for x in result['Deposit_Amount'] if x and str(x).strip())

        print(f"📊 Enhanced Results:")
        print(f"   Total transactions: {len(result)}")
        print(f"   Withdrawal entries: {withdrawal_count}")
        print(f"   Deposit entries: {deposit_count}")
        print(f"   Enhanced metadata filtering: ✅")
        print(f"   Cross-page noise removal: ✅")

        print(f"\n📋 Sample enhanced clean transactions:")
        print("-" * 100)

        for i, row in result.head(5).iterrows():
            withdrawal = f"₹{row['Withdrawal_Amount']:,.2f}" if row['Withdrawal_Amount'] else '[blank]'
            deposit = f"₹{row['Deposit_Amount']:,.2f}" if row['Deposit_Amount'] else '[blank]'

            print(f"{i+1}. {row['Date']}")
            print(f"   Narration: {row['Narration'][:90]}...")
            print(f"   Reference: {row['Chq_Ref_No']}")
            print(f"   W: {withdrawal} | D: {deposit}")
            print()

        return True
    else:
        print("❌ Failed!")
        return False

if __name__ == "__main__":
    main()
