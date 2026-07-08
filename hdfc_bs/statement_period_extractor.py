import pdfplumber
import pandas as pd
import re
from datetime import datetime

def extract_statement_period(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Usually statement period is on the first page
            first_page = pdf.pages[0]
            page_text = first_page.extract_text()

            if not page_text:
                return create_empty_period_df("Could not extract text from PDF")

            # Extract From and To dates using regex patterns
            from_date, to_date = extract_dates_from_text(page_text)

            if from_date and to_date:
                return create_period_dataframe(from_date, to_date)
            else:
                return create_empty_period_df("Statement period not found")

    except Exception as e:
        return create_empty_period_df(f"Error reading PDF: {str(e)}")

def extract_dates_from_text(text):
    """Extract From and To dates from PDF text"""

    from_date = None
    to_date = None

    # Pattern 1: "From : DD/MM/YYYY To : DD/MM/YYYY" (most common)
    pattern1 = r'From\s*:\s*(\d{2}/\d{2}/\d{4})\s+To\s*:\s*(\d{2}/\d{2}/\d{4})'
    match1 = re.search(pattern1, text, re.IGNORECASE)

    if match1:
        from_date = match1.group(1)
        to_date = match1.group(2)
        return from_date, to_date

    # Pattern 2: "From: DD/MM/YYYY To: DD/MM/YYYY" (without spaces)
    pattern2 = r'From:\s*(\d{2}/\d{2}/\d{4})\s+To:\s*(\d{2}/\d{2}/\d{4})'
    match2 = re.search(pattern2, text, re.IGNORECASE)

    if match2:
        from_date = match2.group(1)
        to_date = match2.group(2)
        return from_date, to_date

    # Pattern 3: Try to find dates separately
    from_match = re.search(r'From\s*:\s*(\d{2}/\d{2}/\d{4})', text, re.IGNORECASE)
    to_match = re.search(r'To\s*:\s*(\d{2}/\d{2}/\d{4})', text, re.IGNORECASE)

    if from_match and to_match:
        from_date = from_match.group(1)
        to_date = to_match.group(1)
        return from_date, to_date

    return None, None

def create_period_dataframe(from_date, to_date):
    """Create a formatted DataFrame with statement period"""

    try:
        # Parse dates to validate and format them
        from_dt = datetime.strptime(from_date, '%d/%m/%Y')
        to_dt = datetime.strptime(to_date, '%d/%m/%Y')

        # Format dates nicely
        from_formatted = from_dt.strftime('%d-%b-%Y')  # e.g., 07-Sep-2024
        to_formatted = to_dt.strftime('%d-%b-%Y')      # e.g., 09-Sep-2024

        # Calculate period duration
        duration = (to_dt - from_dt).days + 1  # +1 to include both start and end dates

        # Create DataFrame
        period_data = {
            'Field': ['Statement Period', 'From Date', 'To Date', 'Duration (Days)'],
            'Value': [
                f"{from_formatted} to {to_formatted}",
                from_formatted,
                to_formatted,
                f"{duration} days"
            ]
        }

        return pd.DataFrame(period_data)

    except ValueError as e:
        return create_empty_period_df(f"Date parsing error: {str(e)}")

def create_empty_period_df(error_message):
    """Create an empty DataFrame when extraction fails"""
    return pd.DataFrame({
        'Field': ['Statement Period', 'Error'],
        'Value': ['Not Available', error_message]
    })

def save_period_to_excel(period_df, excel_path, sheet_name='Statement_Info'):
    """
    Save statement period to Excel file

    Args:
        period_df (DataFrame): Period information DataFrame
        excel_path (str): Path to Excel file
        sheet_name (str): Name of the sheet to create
    """

    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            period_df.to_excel(writer, sheet_name=sheet_name, index=False)

            # Format the sheet
            worksheet = writer.sheets[sheet_name]

            # Set column widths
            worksheet.column_dimensions['A'].width = 20
            worksheet.column_dimensions['B'].width = 30

            # Make headers bold
            for cell in worksheet[1]:
                cell.font = cell.font.copy(bold=True)

        print(f"✅ Statement period saved to sheet '{sheet_name}' in {excel_path}")

    except Exception as e:
        print(f"❌ Error saving to Excel: {str(e)}")

# Example usage function
def extract_and_display_period(pdf_path):
    """
    Extract statement period and display results

    Args:
        pdf_path (str): Path to PDF file

    Returns:
        pandas.DataFrame: Statement period information
    """

    print(f"🔍 Extracting statement period from: {pdf_path}")
    print("=" * 60)

    period_df = extract_statement_period(pdf_path)

    print("📅 STATEMENT PERIOD INFORMATION:")
    print("-" * 40)

    for _, row in period_df.iterrows():
        print(f"{row['Field']:<20}: {row['Value']}")

    print("=" * 60)

    return period_df


period_info = extract_and_display_period("bs1.pdf")

        # Save to Excel (optional)
        # excel_file = "statement_period.xlsx"
        # save_period_to_excel(period_info, excel_file)
        
print(type(period_info))
print(f"\n📊 DataFrame shape: {period_info.shape}")
print("\n📋 DataFrame contents:")
print(period_info.to_string(index=False))
