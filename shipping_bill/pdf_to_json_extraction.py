"""
PDF to JSON Extraction - Multiple Approaches (FIXED)
Handles NaN/NA values properly for JSON serialization
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================
# FIX: Custom JSON Encoder for NaN/NA values
# ============================================

class NaNEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NaN, NA, and infinity values"""
    def default(self, obj):
        if isinstance(obj, (np.float64, np.float32)):
            if np.isnan(obj):
                return None
            elif np.isinf(obj):
                return None
        elif pd.isna(obj):  # Handles pd.NA, np.nan, None
            return None
        return super().default(obj)


def clean_for_json(data):
    """
    Clean data to be JSON serializable
    Replaces NaN, NA, None with null (None in Python)
    """
    if isinstance(data, list):
        return [clean_for_json(item) for item in data]
    elif isinstance(data, dict):
        return {key: clean_for_json(value) for key, value in data.items()}
    elif pd.isna(data):
        return None
    elif isinstance(data, (np.float64, np.float32, np.int64, np.int32)):
        if np.isnan(data) if isinstance(data, (np.float64, np.float32)) else False:
            return None
        return data.item()  # Convert numpy type to Python native type
    else:
        return data


# ============================================
# METHOD 1: Convert Your Current Camelot Output to JSON (FIXED)
# ============================================

def camelot_to_json(pdf_path, output_json, flavor='lattice'):
    """
    Extract tables using Camelot and save as JSON
    FIXED: Handles NaN/NA values properly
    """
    import camelot

    print(f"Extracting from: {pdf_path}")

    # Extract tables
    tables = camelot.read_pdf(
        str(pdf_path),
        pages='all',
        flavor=flavor,
        strip_text='\n'
    )

    print(f"Found {len(tables)} tables")

    # Structure data as JSON
    result = {
        "document": str(pdf_path),
        "total_tables": len(tables),
        "extraction_method": f"camelot-{flavor}",
        "tables": []
    }

    # Convert each table to JSON
    for i, table in enumerate(tables, start=1):
        df = table.df

        # Clean dataframe - replace NaN with None
        df = df.replace(r'^\s*$', pd.NA, regex=True)
        df = df.dropna(how='all').dropna(axis=1, how='all')

        # FIXED: Replace NA/NaN with None for JSON serialization
        df = df.fillna(value=np.nan)  # Standardize to NaN
        df = df.replace({np.nan: None})  # Replace NaN with None

        table_data = {
            "table_id": i,
            "page": int(table.page),
            "accuracy": round(float(table.accuracy), 2),
            "dimensions": {
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1])
            },
            "data": clean_for_json(df.values.tolist()),  # Clean data
            "data_dict": clean_for_json(df.to_dict(orient='records'))  # Clean data
        }

        result["tables"].append(table_data)

    # Save to JSON with custom encoder
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, cls=NaNEncoder)

    print(f"✓ Saved to: {output_json}")
    return result


# ============================================
# METHOD 2: Structured JSON (FIXED)
# ============================================

def extract_structured_json(pdf_path, output_json):
    """
    Extract and structure data with meaningful keys
    FIXED: Handles NaN/NA values properly
    """
    import camelot

    print(f"Extracting structured data from: {pdf_path}")

    tables = camelot.read_pdf(str(pdf_path), pages='all', flavor='lattice')

    # Initialize structured result
    result = {
        "metadata": {
            "source_file": str(pdf_path),
            "total_pages": int(max([t.page for t in tables])) if tables else 0,
            "total_tables": len(tables)
        },
        "shipping_bill": {
            "summary": {},
            "invoice_details": [],
            "item_details": [],
            "declarations": []
        }
    }

    # Process each table and categorize
    for i, table in enumerate(tables):
        df = table.df
        df = df.replace(r'^\s*$', pd.NA, regex=True)
        df = df.dropna(how='all').dropna(axis=1, how='all')

        # FIXED: Replace NA/NaN with None
        df = df.fillna(value=np.nan)
        df = df.replace({np.nan: None})

        # Convert to records
        records = clean_for_json(df.to_dict(orient='records'))

        # Categorize by page (you can customize this logic)
        if table.page == 1:
            result["shipping_bill"]["summary"] = {
                "page": int(table.page),
                "data": records
            }
        elif table.page == 2:
            result["shipping_bill"]["invoice_details"] = records
        elif table.page == 3:
            result["shipping_bill"]["item_details"] = records
        else:
            result["shipping_bill"]["declarations"].extend(records)

    # Save to JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, cls=NaNEncoder)

    print(f"✓ Saved structured JSON to: {output_json}")
    return result


# ============================================
# METHOD 3: Using PDFPlumber (FIXED)
# ============================================

def pdfplumber_to_json(pdf_path, output_json):
    """
    Extract using PDFPlumber and save as JSON
    FIXED: Handles NaN/NA values properly
    """
    import pdfplumber

    print(f"Extracting with PDFPlumber from: {pdf_path}")

    result = {
        "document": str(pdf_path),
        "pages": []
    }

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_data = {
                "page_number": page_num,
                "text": page.extract_text() or "",
                "tables": []
            }

            # Extract tables
            tables = page.extract_tables()
            if tables:
                for table_idx, table in enumerate(tables, start=1):
                    df = pd.DataFrame(table)

                    # FIXED: Replace NA/NaN with None
                    df = df.fillna(value=np.nan)
                    df = df.replace({np.nan: None})

                    page_data["tables"].append({
                        "table_id": table_idx,
                        "data": clean_for_json(df.values.tolist()),
                        "data_dict": clean_for_json(df.to_dict(orient='records'))
                    })

            result["pages"].append(page_data)

    # Save to JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, cls=NaNEncoder)

    print(f"✓ Saved to: {output_json}")
    return result


# ============================================
# METHOD 4: Hierarchical JSON (FIXED)
# ============================================

def extract_hierarchical_json(pdf_path, output_json):
    """
    Create deeply nested JSON structure
    FIXED: Handles NaN/NA values properly
    """
    import camelot

    print(f"Creating hierarchical JSON from: {pdf_path}")

    tables = camelot.read_pdf(str(pdf_path), pages='all', flavor='lattice')

    result = {
        "document_info": {
            "filename": Path(pdf_path).name,
            "total_tables": len(tables)
        },
        "content": {
            "by_page": {},
            "by_section": {
                "part_1_summary": [],
                "part_2_invoice": [],
                "part_3_items": [],
                "part_4_scheme": [],
                "part_5_declarations": []
            }
        }
    }

    # Group by page
    for table in tables:
        page_num = int(table.page)
        if page_num not in result["content"]["by_page"]:
            result["content"]["by_page"][page_num] = []

        df = table.df
        df = df.replace(r'^\s*$', pd.NA, regex=True)
        df = df.dropna(how='all').dropna(axis=1, how='all')

        # FIXED: Replace NA/NaN with None
        df = df.fillna(value=np.nan)
        df = df.replace({np.nan: None})

        table_info = {
            "accuracy": round(float(table.accuracy), 2),
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "data": clean_for_json(df.to_dict(orient='records'))
        }

        result["content"]["by_page"][page_num].append(table_info)

    # Save to JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, cls=NaNEncoder)

    print(f"✓ Saved hierarchical JSON to: {output_json}")
    return result


# ============================================
# METHOD 5: Flat JSON (FIXED)
# ============================================

def extract_flat_json(pdf_path, output_json):
    """
    Extract as flat JSON structure
    FIXED: Handles NaN/NA values properly
    """
    import camelot

    print(f"Creating flat JSON from: {pdf_path}")

    tables = camelot.read_pdf(str(pdf_path), pages='all', flavor='lattice')

    result = []

    for table_idx, table in enumerate(tables, start=1):
        df = table.df
        df = df.replace(r'^\s*$', pd.NA, regex=True)
        df = df.dropna(how='all').dropna(axis=1, how='all')

        # FIXED: Replace NA/NaN with None
        df = df.fillna(value=np.nan)
        df = df.replace({np.nan: None})

        # Each row becomes a JSON object
        for row_idx, row in df.iterrows():
            record = {
                "table_id": table_idx,
                "page": int(table.page),
                "row_id": int(row_idx) + 1,
                "accuracy": round(float(table.accuracy), 2)
            }

            # Add each column as a key
            for col_idx, value in enumerate(row):
                # Convert to Python native type if numpy
                if isinstance(value, (np.int64, np.int32)):
                    value = int(value)
                elif isinstance(value, (np.float64, np.float32)):
                    value = float(value) if not np.isnan(value) else None

                record[f"col_{col_idx}"] = value

            result.append(record)

    # Save to JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, cls=NaNEncoder)

    print(f"✓ Saved flat JSON to: {output_json}")
    return result


# ============================================
# METHOD 6: Complete JSON (FIXED)
# ============================================

def extract_complete_json(pdf_path, output_json):
    """
    Extract everything - tables, text, metadata
    FIXED: Handles NaN/NA values properly
    """
    import camelot
    import pdfplumber
    from datetime import datetime

    print(f"Extracting complete data from: {pdf_path}")

    # Get tables with Camelot
    tables = camelot.read_pdf(str(pdf_path), pages='all', flavor='lattice')

    result = {
        "extraction_info": {
            "timestamp": datetime.now().isoformat(),
            "source_file": str(pdf_path),
            "extraction_method": "camelot + pdfplumber"
        },
        "tables": [],
        "text_content": {}
    }

    # Process tables
    for i, table in enumerate(tables, start=1):
        df = table.df
        df = df.replace(r'^\s*$', pd.NA, regex=True)
        df = df.dropna(how='all').dropna(axis=1, how='all')

        # FIXED: Replace NA/NaN with None
        df = df.fillna(value=np.nan)
        df = df.replace({np.nan: None})

        result["tables"].append({
            "id": i,
            "page": int(table.page),
            "accuracy": round(float(table.accuracy), 2),
            "dimensions": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
            "data": clean_for_json(df.to_dict(orient='records'))
        })

    # Extract text
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            result["text_content"][f"page_{page_num}"] = page.extract_text() or ""

    # Save to JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, cls=NaNEncoder)

    print(f"✓ Saved complete JSON to: {output_json}")
    return result


# ============================================
# USAGE EXAMPLES
# ============================================

if __name__ == "__main__":

    PDF_FILE = "sample7_clean.pdf"

    print("="*70)
    print("PDF TO JSON EXTRACTION - FIXED VERSION")
    print("="*70)
    print()

    # Uncomment the method you want to use:

    # Method 1: Basic Camelot to JSON (Recommended)
    # result = camelot_to_json(PDF_FILE, "output_basic.json", flavor='lattice')

    # Method 2: Structured JSON
    result = extract_structured_json(PDF_FILE, "output_structured.json")

    # Method 3: PDFPlumber approach
    # result = pdfplumber_to_json(PDF_FILE, "output_pdfplumber.json")

    # Method 4: Hierarchical JSON
    # result = extract_hierarchical_json(PDF_FILE, "output_hierarchical.json")

    # Method 5: Flat JSON
    # result = extract_flat_json(PDF_FILE, "output_flat.json")

    # Method 6: Complete JSON
    # result = extract_complete_json(PDF_FILE, "output_complete.json")

    print()
    print("="*70)
    print("✓ Extraction Complete! (No more TypeError)")
    print("="*70)
