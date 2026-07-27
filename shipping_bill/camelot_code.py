import pandas as pd
import re
import sys
import os


def is_single_or_double_char(cell_value):
    """Return True if cell contains only 1–2 non‑space characters."""
    if cell_value is None or (isinstance(cell_value, float) and pd.isna(cell_value)):
        return False

    s = str(cell_value).strip()
    if not s:
        return False

    return len(s) <= 2


def clean_row_pandas(row):
    """
    Remove 1–2 char cells and empty cells from a row, shift remaining values left.
    Returns a list of kept values (3+ chars, non‑empty).
    """
    cleaned = []
    for v in row:
        # remove 1–2 char cells
        if is_single_or_double_char(v):
            continue
        # remove empty / None / NaN
        if v is None:
            continue
        if isinstance(v, float) and pd.isna(v):
            continue
        if isinstance(v, str) and not v.strip():
            continue
        cleaned.append(v)
    return cleaned


def should_remove_row(row_text):
    """Return True if the whole row is a layout/header row that must be dropped."""
    t = row_text.upper()

    # 1) Indian Customs header block
    if all(k in t for k in ["INDIAN", "CUSTOMS", "EDI", "SYSTEM"]):
        return True
    if all(k in t for k in ["CENTRAL", "BOARD", "INDIRECT", "TAXES"]):
        return True
    if all(k in t for k in ["DEPARTMENT", "REVENUE", "MINISTRY", "FINANCE"]):
        return True
    if all(k in t for k in ["GOVERNMENT", "INDIA"]) and "PORT" not in t:
        return True

    # 2) 1.MODE … 11.LUT list
    scheme_keywords = ["MODE", "ASSESS", "EXMN", "JOBBING", "MEIS",
                       "DBK", "RODTP", "LICENCE", "DFRC", "RE-EXP", "LUT"]
    if "1.MODE" in t or "1. MODE" in t:
        return True
    if sum(1 for k in scheme_keywords if k in t) >= 5:
        return True

    # 3) PART - I - SHIPPING BILL SUMMARY
    if re.search(r"PART\s*-\s*I\s*-\s*SHIPPING\s+BILL\s+SUMMARY", t, re.IGNORECASE):
        return True

    # 4) Complex STATUS / DECLARAN / DETAILS row
    status_kw = ["STATUS", "DECLARAN", "DETAILS", "VALU", "SUMMA",
                 "MANIFEST", "EQUIPMENT", "ANNEX", "PROCESS"]
    if sum(1 for k in status_kw if k in t) >= 5:
        return True
    if re.search(r"STATUS.*DECLARAN.*DETAILS.*VALU.*SUMMA", t, re.IGNORECASE | re.DOTALL):
        return True

    # 5) Section labels you listed
    patterns = [
        r"B\s+DECLARAN\s+DETAILS",
        r"C\.VALU\s+SUMMA",
        r"E\s+MANIFEST\s+DETAILS",
        r"G\.\s*EQUIPMENT\s+DETAILS",
        r"I\.\s*ANNEX\s+DETAILS",
        r"J\.PROCESS\s+DETAILS",
        r"D\.\s*EX\.PR\.",
        r"F\.INVOICE\s+SUMMARY",
        r"CHALLAN\s+DETAILS\s+H",
    ]
    for pat in patterns:
        if re.search(pat, t, re.IGNORECASE):
            return True

    return False


def clean_excel_file(input_file, output_file=None):
    """Main cleaning logic using pandas; operates on all sheets."""
    if output_file is None:
        base = os.path.splitext(input_file)[0]
        output_file = f"{base}_cleaned_v3.xlsx"

    print("\n" + "=" * 70)
    print("Excel Data Cleaner (Pandas, delete 1–2 char + shift left)")
    print("=" * 70)
    print(f"Input file : {input_file}")
    print(f"Output file: {output_file}")
    print("=" * 70 + "\n")

    try:
        xls = pd.ExcelFile(input_file)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    total_rows_removed = 0
    total_cells_deleted = 0

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for idx, sheet_name in enumerate(xls.sheet_names, 1):
            print(f"Processing sheet {idx}/{len(xls.sheet_names)}: {sheet_name}")
            df = pd.read_excel(input_file, sheet_name=sheet_name, header=None)

            new_rows = []
            rows_removed_here = 0
            cells_deleted_here = 0

            for r_i, row in df.iterrows():
                row_list = [x for x in row]  # keep as list
                row_text = " ".join(str(x) if not pd.isna(x) else "" for x in row_list)

                # stop at Glossary
                if "GLOSSARY" in row_text.upper():
                    print(f"  → Found 'Glossary' at row {r_i + 1}, dropping this and all below")
                    break

                # drop layout / header rows
                if should_remove_row(row_text):
                    rows_removed_here += 1
                    continue

                # clean row (delete 1–2 char & empty, shift left)
                cleaned = clean_row_pandas(row_list)
                cells_deleted_here += len(row_list) - len(cleaned)

                # pad back to original width so all rows have same length
                while len(cleaned) < len(row_list):
                    cleaned.append(None)

                new_rows.append(cleaned)

            if new_rows:
                new_df = pd.DataFrame(new_rows)
                # drop columns that are completely empty
                new_df = new_df.dropna(axis=1, how="all")
                new_df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)

            total_rows_removed += rows_removed_here
            total_cells_deleted += cells_deleted_here

            print(f"  ✓ Rows removed : {rows_removed_here}")
            print(f"  ✓ Cells deleted: {cells_deleted_here}")
            print(f"  ✓ Remaining rows in output sheet: {len(new_rows)}\n")

    print("=" * 70)
    print("✓ SUCCESS – cleaned file written")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  • Total sheets processed: {len(xls.sheet_names)}")
    print(f"  • Total rows removed    : {total_rows_removed}")
    print(f"  • Total cells deleted   : {total_cells_deleted}\n")
    print("All 1–2 character cells and empty cells were removed and rows compacted left.")
    return output_file


def main():
    # Set input/output paths here
    input_file = "C:/Users/Divyesh Parmar/Downloads/GRP SL02 PAYSHEET MAY-2026.pdf"
    output_file = None  # None → writes <input>_cleaned_v3.xlsx

    if not os.path.exists(input_file):
        print(f"Error: file '{input_file}' not found")
        sys.exit(1)

    if not input_file.lower().endswith((".xlsx", ".xls")):
        print("Error: input file must be .xlsx or .xls")
        sys.exit(1)

    clean_excel_file(input_file, output_file)


if __name__ == "__main__":
    main()
