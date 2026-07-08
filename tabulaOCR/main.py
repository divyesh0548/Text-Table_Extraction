# Complete OCR + Tabula workflow for extracting tables from scanned PDFs

import os
import pandas as pd
import tabula
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import pytesseract
import img2pdf
import shutil
from pypdf import PdfWriter, PdfReader

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Divyesh Parmar\Downloads\Tesserect OCR\tesseract.exe'

scanned_pdf = r"bs_cleaned.pdf"

def enhance_image_for_ocr(image_path):
    """Enhance image quality for better OCR results"""
    with Image.open(image_path) as img:
        # Convert to grayscale
        if img.mode != 'L':
            gray = ImageOps.grayscale(img)
        else:
            gray = img
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(2.0)
        
        # Enhance sharpness
        sharpness_enhancer = ImageEnhance.Sharpness(enhanced)
        sharpened = sharpness_enhancer.enhance(2.0)
        
        # Resize for better quality (300 DPI equivalent)
        width, height = sharpened.size
        new_width = int(width * 2)
        new_height = int(height * 2)
        resized = sharpened.resize((new_width, new_height), Image.LANCZOS)
        
        # Apply noise reduction
        filtered = resized.filter(ImageFilter.MedianFilter(size=3))
        
        # Enhance brightness
        brightness_enhancer = ImageEnhance.Brightness(filtered)
        brightened = brightness_enhancer.enhance(1.1)
        
        return brightened

def ocr_pdf_with_tesseract(scanned_pdf_path, output_searchable_pdf):
    """Convert scanned PDF to searchable PDF using OCR"""
    
    # Create temporary directory
    temp_dir = "temp_ocr"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        print("Converting PDF pages to images...")
        # Convert PDF to high-quality images
        pages = convert_from_path(scanned_pdf_path, dpi=300)
        
        processed_images = []
        
        for i, page in enumerate(pages):
            # Save original page
            original_path = os.path.join(temp_dir, f"page_{i+1:03d}.jpg")
            page.save(original_path, "JPEG", quality=95)
            
            # Enhance image
            enhanced_image = enhance_image_for_ocr(original_path)
            enhanced_path = os.path.join(temp_dir, f"enhanced_page_{i+1:03d}.jpg")
            enhanced_image.save(enhanced_path, "JPEG", quality=95)
            
            # Create searchable PDF using tesseract
            pdf_path = os.path.join(temp_dir, f"page_{i+1:03d}.pdf")
            
            # Use tesseract to create searchable PDF
            pdf_data = pytesseract.image_to_pdf_or_hocr(enhanced_path, extension='pdf')
            with open(pdf_path, 'wb') as f:
                f.write(pdf_data)
            
            processed_images.append(pdf_path)
            print(f"Processed page {i+1}")
        
        # Merge all searchable PDF pages
        merger = PdfWriter()
        for pdf_path in processed_images:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                merger.add_page(page)
        
        # Save merged searchable PDF
        with open(output_searchable_pdf, 'wb') as output_file:
            merger.write(output_file)
        
        print(f"Searchable PDF created: {output_searchable_pdf}")
        return True
        
    except Exception as e:
        print(f"Error in OCR processing: {e}")
        return False
        
    finally:
        # Clean up temporary files
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def extract_tables_with_tabula(searchable_pdf_path, output_dir):
    """Extract tables from searchable PDF using Tabula"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print("Extracting tables with Tabula...")
        
        # Extract tables from all pages
        tables = tabula.read_pdf(
            searchable_pdf_path, 
            pages='all',
            multiple_tables=True,
            stream=True,  # Use stream mode for better detection
            guess=True,   # Let tabula guess the table areas
            pandas_options={'header': 0}  # Use first row as header
        )
        
        print(f"Found {len(tables)} tables")
        
        extracted_tables = []
        
        for i, table in enumerate(tables):
            if not table.empty:
                # Clean the table
                table = table.dropna(how='all')  # Remove empty rows
                table = table.dropna(axis=1, how='all')  # Remove empty columns
                
                # Save table as CSV
                csv_path = os.path.join(output_dir, f"table_{i+1}.csv")
                table.to_csv(csv_path, index=False)
                
                # Save table as Excel
                excel_path = os.path.join(output_dir, f"table_{i+1}.xlsx")
                table.to_excel(excel_path, index=False)
                
                extracted_tables.append({
                    'table_number': i+1,
                    'dataframe': table,
                    'csv_path': csv_path,
                    'excel_path': excel_path,
                    'shape': table.shape
                })
                
                print(f"Table {i+1}: {table.shape[0]} rows x {table.shape[1]} columns")
                print(f"Saved: {csv_path}")
                print(f"Saved: {excel_path}")
                print("-" * 50)
        
        return extracted_tables
        
    except Exception as e:
        print(f"Error extracting tables: {e}")
        return []

def complete_ocr_tabula_workflow(scanned_pdf_path):
    """Complete workflow: OCR + Tabula table extraction"""
    
    base_name = os.path.splitext(os.path.basename(scanned_pdf_path))[0]
    
    # Step 1: Create searchable PDF using OCR
    searchable_pdf = f"{base_name}_searchable.pdf"
    print("Step 1: Converting scanned PDF to searchable PDF using OCR...")
    
    if not ocr_pdf_with_tesseract(scanned_pdf_path, searchable_pdf):
        print("Failed to create searchable PDF")
        return None
    
    # Step 2: Extract tables using Tabula
    output_dir = f"{base_name}_tables"
    print("Step 2: Extracting tables using Tabula...")
    
    extracted_tables = extract_tables_with_tabula(searchable_pdf, output_dir)
    
    if extracted_tables:
        print(f"\n✅ Successfully extracted {len(extracted_tables)} tables!")
        print(f"📁 Tables saved in: {output_dir}")
        print(f"📄 Searchable PDF: {searchable_pdf}")
        
        # Display first few rows of each table
        for table_info in extracted_tables:
            print(f"\nTable {table_info['table_number']} Preview:")
            print(table_info['dataframe'].head())
    else:
        print("❌ No tables found or extraction failed")
    
    return extracted_tables


complete_ocr_tabula_workflow(scanned_pdf)