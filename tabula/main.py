# Complete Tabula-py PDF Table Extraction Code

import pandas as pd
import tabula
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def extract_single_table_basic(pdf_path):
    """Basic table extraction from PDF"""
    try:
        print("Basic extraction - reading first table from first page...")
        
        # Basic extraction - gets first table from first page
        df = tabula.read_pdf(pdf_path, pages=1)
        
        if isinstance(df, list) and len(df) > 0:
            table = df[0]
            print(f"Extracted table shape: {table.shape}")
            print("First few rows:")
            print(table.head())
            return table
        else:
            print("No tables found")
            return None
            
    except Exception as e:
        print(f"Error in basic extraction: {e}")
        return None

def extract_all_tables_all_pages(pdf_path):
    """Extract all tables from all pages"""
    try:
        print("Extracting all tables from all pages...")
        
        # Extract all tables from all pages
        tables = tabula.read_pdf(
            pdf_path, 
            pages='all',           # All pages
            multiple_tables=True,  # Get multiple tables per page
            stream=True,           # Stream mode for tables without clear borders
            guess=True,            # Let tabula guess table areas
            pandas_options={'header': 0}  # First row as header
        )
        
        print(f"Total tables found: {len(tables)}")
        
        for i, table in enumerate(tables, 1):
            if not table.empty:
                print(f"\nTable {i}:")
                print(f"Shape: {table.shape}")
                print("Sample data:")
                print(table.head(3))
                print("-" * 50)
        
        return tables
        
    except Exception as e:
        print(f"Error extracting all tables: {e}")
        return []

def extract_tables_specific_pages(pdf_path, pages):
    """Extract tables from specific pages"""
    try:
        print(f"Extracting tables from pages: {pages}")
        
        # Extract from specific pages
        tables = tabula.read_pdf(
            pdf_path,
            pages=pages,          # Specific pages like '1,3,5' or '1-5'
            multiple_tables=True,
            stream=True,
            guess=True
        )
        
        print(f"Tables found: {len(tables)}")
        return tables
        
    except Exception as e:
        print(f"Error extracting from specific pages: {e}")
        return []

def extract_tables_with_area_selection(pdf_path, area_coordinates):
    """Extract tables from specific areas of pages"""
    try:
        print(f"Extracting tables from specific area: {area_coordinates}")
        
        # Area format: [top, left, bottom, right] in points
        tables = tabula.read_pdf(
            pdf_path,
            pages='all',
            area=area_coordinates,  # [top, left, bottom, right]
            multiple_tables=True,
            stream=True
        )
        
        print(f"Tables found in specified area: {len(tables)}")
        return tables
        
    except Exception as e:
        print(f"Error extracting from area: {e}")
        return []

def extract_and_save_tables(pdf_path, output_dir):
    """Extract tables and save in multiple formats"""
    try:
        print("Extracting and saving tables...")
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Extract all tables
        tables = tabula.read_pdf(
            pdf_path, 
            pages='all',
            multiple_tables=True,
            stream=True,
            guess=True
        )
        
        saved_files = []
        
        for i, table in enumerate(tables, 1):
            if not table.empty:
                # Clean the table
                table = table.dropna(how='all')  # Remove empty rows
                table = table.dropna(axis=1, how='all')  # Remove empty columns
                
                # Generate filenames
                base_name = f"table_{i:02d}"
                csv_path = os.path.join(output_dir, f"{base_name}.csv")
                excel_path = os.path.join(output_dir, f"{base_name}.xlsx")
                json_path = os.path.join(output_dir, f"{base_name}.json")
                
                # Save in different formats
                table.to_csv(csv_path, index=False)
                table.to_excel(excel_path, index=False)
                table.to_json(json_path, orient='records', indent=2)
                
                saved_files.append({
                    'table_number': i,
                    'csv': csv_path,
                    'excel': excel_path,
                    'json': json_path,
                    'shape': table.shape,
                    'dataframe': table
                })
                
                print(f"Table {i} saved:")
                print(f"  CSV: {csv_path}")
                print(f"  Excel: {excel_path}")
                print(f"  JSON: {json_path}")
                print(f"  Shape: {table.shape}")
        
        return saved_files
        
    except Exception as e:
        print(f"Error saving tables: {e}")
        return []

def extract_and_combine_tables(pdf_path):
    """Extract all tables and combine into single DataFrame"""
    try:
        print("Extracting and combining all tables...")
        
        tables = tabula.read_pdf(
            pdf_path,
            pages='all',
            multiple_tables=True,
            stream=True,
            guess=True
        )
        
        if not tables:
            print("No tables found")
            return None
        
        # Clean and combine tables
        cleaned_tables = []
        for table in tables:
            if not table.empty:
                # Clean the table
                cleaned_table = table.dropna(how='all').dropna(axis=1, how='all')
                if not cleaned_table.empty:
                    cleaned_tables.append(cleaned_table)
        
        if cleaned_tables:
            # Combine all tables
            combined_df = pd.concat(cleaned_tables, ignore_index=True)
            print(f"Combined table shape: {combined_df.shape}")
            return combined_df
        else:
            print("No valid tables to combine")
            return None
            
    except Exception as e:
        print(f"Error combining tables: {e}")
        return None

def direct_pdf_to_csv_conversion(pdf_path, output_csv):
    """Directly convert PDF to CSV without DataFrame processing"""
    try:
        print(f"Converting PDF directly to CSV: {output_csv}")
        
        # Direct conversion to CSV
        tabula.convert_into(
            pdf_path,
            output_csv,
            output_format="csv",
            pages='all',
            stream=True,
            guess=True
        )
        
        print(f"PDF converted to CSV: {output_csv}")
        return True
        
    except Exception as e:
        print(f"Error in direct conversion: {e}")
        return False

def batch_process_pdfs(input_directory, output_format='csv'):
    """Process all PDFs in a directory"""
    try:
        print(f"Batch processing PDFs in: {input_directory}")
        
        # Convert all PDFs in directory
        tabula.convert_into_by_batch(
            input_directory,
            output_format=output_format,  # 'csv', 'json', 'tsv'
            pages='all',
            stream=True,
            guess=True
        )
        
        print(f"Batch processing completed. Output format: {output_format}")
        return True
        
    except Exception as e:
        print(f"Error in batch processing: {e}")
        return False

def advanced_extraction_with_options(pdf_path):
    """Advanced extraction with various options"""
    try:
        print("Advanced extraction with custom options...")
        
        # Advanced extraction with multiple options
        tables = tabula.read_pdf(
            pdf_path,
            pages='all',
            multiple_tables=True,
            
            # Extraction modes
            stream=True,           # For tables without clear borders
            lattice=False,         # For tables with clear borders (set True if needed)
            
            # Detection options
            guess=True,            # Auto-detect table areas
            
            # Area selection (uncomment if needed)
            # area=[50, 12, 500, 600],  # [top, left, bottom, right]
            
            # Column detection (uncomment if needed)
            # columns=[100, 200, 300, 400],  # X coordinates of column separators
            
            # Password protection (uncomment if needed)
            # password="your_password",
            
            # Output options
            silent=True,           # Suppress Java output
            
            # Pandas options
            pandas_options={
                'header': 0,       # First row as header
                'dtype': str       # Keep all data as strings initially
            }
        )
        
        print(f"Advanced extraction found {len(tables)} tables")
        
        for i, table in enumerate(tables, 1):
            if not table.empty:
                print(f"\nTable {i} info:")
                print(f"Shape: {table.shape}")
                print(f"Columns: {list(table.columns)}")
                print("Sample:")
                print(table.head(2))
        
        return tables
        
    except Exception as e:
        print(f"Error in advanced extraction: {e}")
        return []

def extract_tables_with_error_handling(pdf_path):
    """Robust extraction with comprehensive error handling"""
    try:
        print("Starting robust table extraction...")
        
        # Check if file exists
        if not os.path.exists(pdf_path):
            print(f"Error: File not found - {pdf_path}")
            return None
        
        print(f"Processing: {pdf_path}")
        
        # Try different extraction methods
        methods = [
            {'stream': True, 'lattice': False, 'guess': True},
            {'stream': False, 'lattice': True, 'guess': True},
            {'stream': True, 'lattice': False, 'guess': False},
        ]
        
        for i, method in enumerate(methods, 1):
            try:
                print(f"Trying extraction method {i}: {method}")
                
                tables = tabula.read_pdf(
                    pdf_path,
                    pages='all',
                    multiple_tables=True,
                    **method
                )
                
                if tables and len(tables) > 0:
                    valid_tables = [t for t in tables if not t.empty]
                    if valid_tables:
                        print(f"Success with method {i}! Found {len(valid_tables)} valid tables")
                        return valid_tables
                
            except Exception as method_error:
                print(f"Method {i} failed: {method_error}")
                continue
        
        print("All extraction methods failed")
        return None
        
    except Exception as e:
        print(f"Critical error: {e}")
        return None

def main_extraction_demo():
    """Main function demonstrating various extraction methods"""
    
    # Get PDF file from user
    pdf_path = input("Enter path to PDF file: ").strip().strip('"')
    
    if not os.path.exists(pdf_path):
        print("PDF file not found!")
        return
    
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_dir = f"{base_name}_extracted_tables"
    
    print("="*60)
    print("TABULA PDF TABLE EXTRACTION DEMO")
    print("="*60)
    
    # Method 1: Basic extraction
    print("\n1. BASIC EXTRACTION")
    print("-" * 30)
    basic_table = extract_single_table_basic(pdf_path)
    
    # Method 2: Extract all tables from all pages
    print("\n2. ALL TABLES FROM ALL PAGES")
    print("-" * 30)
    all_tables = extract_all_tables_all_pages(pdf_path)
    
    # Method 3: Extract and save tables
    print("\n3. EXTRACT AND SAVE TABLES")
    print("-" * 30)
    saved_files = extract_and_save_tables(pdf_path, output_dir)
    
    # Method 4: Combine all tables
    print("\n4. COMBINE ALL TABLES")
    print("-" * 30)
    combined_df = extract_and_combine_tables(pdf_path)
    if combined_df is not None:
        combined_csv = f"{base_name}_combined_tables.csv"
        combined_df.to_csv(combined_csv, index=False)
        print(f"Combined table saved: {combined_csv}")
    
    # Method 5: Direct PDF to CSV
    print("\n5. DIRECT PDF TO CSV CONVERSION")
    print("-" * 30)
    direct_csv = f"{base_name}_direct_conversion.csv"
    direct_pdf_to_csv_conversion(pdf_path, direct_csv)
    
    # Method 6: Advanced extraction
    print("\n6. ADVANCED EXTRACTION WITH OPTIONS")
    print("-" * 30)
    advanced_tables = advanced_extraction_with_options(pdf_path)
    
    # Method 7: Robust extraction
    print("\n7. ROBUST EXTRACTION WITH ERROR HANDLING")
    print("-" * 30)
    robust_tables = extract_tables_with_error_handling(pdf_path)
    
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE!")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Files created:")
    print(f"  - Individual table files in: {output_dir}/")
    if combined_df is not None:
        print(f"  - Combined tables: {combined_csv}")
    print(f"  - Direct conversion: {direct_csv}")

if __name__ == "__main__":
    print("Tabula-py PDF Table Extraction Tool")
    print("Requirements: pip install tabula-py pandas")
    print("Note: Java must be installed and accessible")
    print()
    
    main_extraction_demo()
