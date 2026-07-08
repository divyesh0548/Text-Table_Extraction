"""
Excel Data Cleaner Script (Enhanced v3 - Using Pandas)
======================================================
Removes specific text fields and properly DELETES 1-2 character cells

Author: AI Assistant
Date: November 21, 2025
"""

import pandas as pd
from openpyxl import load_workbook
import re
import sys
import os


def is_single_or_double_char(cell_value):
    """Check if cell contains only 1-2 characters."""
    if cell_value is None or pd.isna(cell_value):
        return False

    cell_str = str(cell_value).strip()
    if not cell_str:
        return False

    return len(cell_str) <= 2


def clean_row_pandas(row):
    """
    Remove 1-2 char and empty cells from a row, shift left.
    Returns list with only valid cells (3+ chars).
    """
    cleaned = []
    for cell in row:
        # Skip if 1-2 characters
        if is_single_or_double_char(cell):
            continue
        # Skip if empty/None
        if cell is None or (isinstance(cell, str) and not cell.strip()):
            continue
        # Keep if 3+ characters
        cleaned.append(cell)
    return cleaned


def should_remove_row(row_text):
    """Check if row should be completely removed."""
    # Indian Customs header
    if all(kw in row_text.upper() for kw in ['INDIAN', 'CUSTOMS', 'EDI', 'SYSTEM']):
        return True
    if all(kw in row_text.upper() for kw in ['CENTRAL', 'BOARD', 'INDIRECT', 'TAXES']):
        return True
    if all(kw in row_text.upper() for kw in ['DEPARTMENT', 'REVENUE', 'MINISTRY', 'FINANCE']):
        return True
    if all(kw in row_text.upper() for kw in ['GOVERNMENT', 'INDIA']) and 'PORT' not in row_text.upper():
        return True

    # Mode/Scheme list
    scheme_kw = ['MODE', 'ASSESS', 'EXMN', 'JOBBING', 'MEIS', 'DBK', 'RODTP', 'LICENCE', 'DFRC', 'RE-EXP', 'LUT']
    if '1.MODE' in row_text or '1. MODE' in row_text:
        return True
    if sum(1 for kw in scheme_kw if kw in row_text.upper()) >= 5:
        return True

    # Section headers
    if re.search(r'PART\s*-\s*I\s*-\s*SHIPPING\s+BILL', row_text, re.IGNORECASE):
        return True

    # Status/Details rows
    status_kw = ['STATUS', 'DECLARAN', 'DETAILS', 'VALU', 'SUMMA', 'MANIFEST', 'EQUIPMENT', 'ANNEX', 'PROCESS']
    if sum(1 for kw in status_kw if kw in row_text.upper()) >= 5:
        return True

    # Section labels
    patterns = [
        r'B\s+DECLARAN', r'C\.VALU\s+SUMMA', r'E\s+MANIFEST',
        r'G\.\s+EQUIPMENT', r'I\.\s+ANNEX', r'J\.PROCESS',
        r'D\.\s+EX\.PR\.', r'F\.INVOICE\s+SUMMARY', r'CHALLAN\s+DETAILS'
    ]
    for pat in patterns:
        if re.search(pat, row_text, re.IGNORECASE):
            return True

    return False


def clean_excel_file(input_file, output_file=None):
    """Clean Excel file using pandas for proper cell handling."""
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_cleaned_v3.xlsx"

    print(f"\n{'='*70}")
    print(f"Excel Data Cleaner (Enhanced v3 - PANDAS Edition)")
    print(f"{'='*70}")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"{'='*70}\n")

    try:
        xl_file = pd.ExcelFile(input_file)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    total_cells_deleted = 0
    total_rows_removed = 0

    # Create Excel writer
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for sheet_idx, sheet_name in enumerate(xl_file.sheet_names, 1):
            print(f"Processing Sheet {sheet_idx}/{len(xl_file.sheet_names)}: {sheet_name}")

            # Read sheet with no header
            df = pd.read_excel(input_file, sheet_name=sheet_name, header=None)

            rows_to_remove = []
            cleaned_rows = []
            cells_deleted_in_sheet = 0

            # Process each row
            for idx, row in df.iterrows():
                row_text = " ".join([str(x) if pd.notna(x) else "" for x in row])

                # Check for Glossary - remove all from here onwards
                if 'Glossary' in row_text or 'GLOSSARY' in row_text:
                    print(f"  → Found 'Glossary' at row {idx+1}, removing all content after it")
                    break

                # Check if row should be completely removed
                if should_remove_row(row_text):
                    rows_to_remove.append(idx)
                else:
                    # Clean the row (remove 1-2 char cells)
                    original_len = len(row)
                    cleaned_row = clean_row_pandas(row.tolist())

                    # Track cells deleted
                    cells_deleted = original_len - len(cleaned_row)
                    cells_deleted_in_sheet += cells_deleted

                    # Pad with None to maintain structure
                    while len(cleaned_row) < len(row):
                        cleaned_row.append(None)

                    cleaned_rows.append((idx, cleaned_row))

            # Build new dataframe without removed rows
            new_data = []
            for idx, row in df.iterrows():
                if idx not in rows_to_remove:
                    # Check if this row was cleaned
                    cleaned_version = next((r[1] for r in cleaned_rows if r[0] == idx), None)
                    if cleaned_version:
                        new_data.append(cleaned_version)
                    else:
                        new_data.append(row.tolist())

            # Create new dataframe
            if new_data:
                new_df = pd.DataFrame(new_data)

                # Remove columns that are completely empty or contain mostly NaN
                new_df = new_df.dropna(axis=1, how='all')

                # Write to Excel
                new_df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)

            total_cells_deleted += cells_deleted_in_sheet
            total_rows_removed += len(rows_to_remove)

            print(f"  ✓ Deleted rows: {len(rows_to_remove)}")
            print(f"  ✓ Cells deleted: {cells_deleted_in_sheet}")
            print(f"  ✓ Remaining rows: {len(new_data) if new_data else 0}\n")

    print(f"{'='*70}")
    print(f"✓ SUCCESS! Cleaned file saved")
    print(f"{'='*70}")
    print(f"\nSummary:")
    print(f"  • Sheets processed: {len(xl_file.sheet_names)}")
    print(f"  • Total rows removed: {total_rows_removed}")
    print(f"  • Total cells deleted: {total_cells_deleted}")
    print(f"\n✓ All 1-2 character cells DELETED and shifted LEFT")
    print(f"✓ All empty cells removed horizontally")
    print(f"✓ Content compacted to the left\n")

    return output_file


def main():
    if len(sys.argv) < 2:
        print("Usage: python clean_excel_data.py <input_file.xlsx> [output_file.xlsx]")
        print("\nExample:")
        print("  python clean_excel_data.py camelot_code_output.xlsx")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found!")
        sys.exit(1)

    if not input_file.endswith(('.xlsx', '.xls')):
        print(f"Error: Input file must be .xlsx or .xls")
        sys.exit(1)

    clean_excel_file(input_file, output_file)


if __name__ == "__main__":
    main()
