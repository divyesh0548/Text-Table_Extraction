# Complete PDF Rotation and Enhancement Program

import os
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import img2pdf
import shutil
from pypdf import PdfReader, PdfWriter

def rotate_pdf_pages(input_pdf_path, output_pdf_path, rotation_direction):
    """
    Rotate all PDF pages by 90 degrees
    rotation_direction: 'left' (counter-clockwise) or 'right' (clockwise)
    """
    try:
        reader = PdfReader(input_pdf_path)
        writer = PdfWriter()
        
        # Determine rotation angle
        if rotation_direction.lower() == 'left':
            angle = -90  # Counter-clockwise (left)
        elif rotation_direction.lower() == 'right':
            angle = 90   # Clockwise (right)
        else:
            print("Invalid rotation direction. Use 'left' or 'right'")
            return False
        
        print(f"Rotating PDF {angle} degrees ({'left' if angle < 0 else 'right'})...")
        
        # Rotate each page
        for i, page in enumerate(reader.pages):
            rotated_page = page.rotate(angle)
            writer.add_page(rotated_page)
            print(f"Rotated page {i+1}")
        
        # Save rotated PDF
        with open(output_pdf_path, 'wb') as output_file:
            writer.write(output_file)
        
        print(f"Rotated PDF saved: {output_pdf_path}")
        return True
        
    except Exception as e:
        print(f"Error rotating PDF: {e}")
        return False

def extract_pages_from_pdf(input_pdf_path, output_dir, page_range=None):
    """
    Extract specific pages or all pages from PDF
    page_range: tuple (start, end) or None for all pages (1-indexed)
    """
    try:
        reader = PdfReader(input_pdf_path)
        total_pages = len(reader.pages)
        
        if page_range:
            start_page, end_page = page_range
            start_page = max(1, start_page) - 1  # Convert to 0-indexed
            end_page = min(total_pages, end_page)
            pages_to_extract = range(start_page, end_page)
        else:
            pages_to_extract = range(total_pages)
        
        extracted_files = []
        
        for i in pages_to_extract:
            writer = PdfWriter()
            writer.add_page(reader.pages[i])
            
            page_filename = f"page_{i+1:03d}.pdf"
            page_path = os.path.join(output_dir, page_filename)
            
            with open(page_path, 'wb') as output_file:
                writer.write(output_file)
            
            extracted_files.append(page_path)
            print(f"Extracted page {i+1} to {page_filename}")
        
        return extracted_files
        
    except Exception as e:
        print(f"Error extracting pages: {e}")
        return []

def pdf_to_images(pdf_path, temp_dir, dpi=100):
    """Convert PDF pages to high-resolution images"""
    try:
        pages = convert_from_path(pdf_path, dpi=dpi)
        image_paths = []
        
        for i, page in enumerate(pages):
            image_name = f"page_{i+1:03d}.jpg"
            image_path = os.path.join(temp_dir, image_name)
            page.save(image_path, "JPEG", quality=95)
            image_paths.append(image_path)
        
        return image_paths
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return []

def preprocess_image_pil_enhanced(image_path, scale_factor=1.5):
    """Enhanced PIL preprocessing for better quality"""
    with Image.open(image_path) as img:
        if img.mode != 'L':
            gray = ImageOps.grayscale(img)
        else:
            gray = img
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(1.8)
        
        # Enhance sharpness
        sharpness_enhancer = ImageEnhance.Sharpness(enhanced)
        sharpened = sharpness_enhancer.enhance(2.0)
        
        # Resize for better quality
        width, height = sharpened.size
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        resized = sharpened.resize((new_width, new_height), Image.LANCZOS)
        
        # Apply noise reduction filter
        filtered = resized.filter(ImageFilter.MedianFilter(size=3))
        
        # Enhance brightness slightly
        brightness_enhancer = ImageEnhance.Brightness(filtered)
        brightened = brightness_enhancer.enhance(1.1)
        
        return brightened

def process_and_save_enhanced_images(image_paths, temp_dir):
    """Process images and save enhanced versions"""
    enhanced_paths = []
    
    for image_path in image_paths:
        try:
            pil_processed = preprocess_image_pil_enhanced(image_path)
            
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            enhanced_path = os.path.join(temp_dir, f"{base_name}_enhanced.jpg")
            pil_processed.save(enhanced_path, "JPEG", quality=95)
            enhanced_paths.append(enhanced_path)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            enhanced_paths.append(image_path)
    
    return enhanced_paths

def images_to_pdf(image_paths, output_pdf_path):
    """Convert enhanced images back to PDF"""
    try:
        image_paths.sort()
        
        # Try img2pdf first
        try:
            with open(output_pdf_path, "wb") as pdf_file:
                pdf_file.write(img2pdf.convert(image_paths))
            return True
        except:
            # Fallback to PIL method
            images = []
            for img_path in image_paths:
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                images.append(img)
            
            if images:
                images[0].save(
                    output_pdf_path, 
                    "PDF", 
                    resolution=300.0, 
                    save_all=True, 
                    append_images=images[1:],
                    optimize=True
                )
                
                for img in images:
                    img.close()
                return True
        
    except Exception as e:
        print(f"Error creating PDF: {e}")
        return False

def enhance_pdf_quality(input_pdf_path, output_pdf_path):
    """Enhance PDF quality using image preprocessing"""
    temp_dir = "temp_enhancement"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Convert PDF to images
        image_paths = pdf_to_images(input_pdf_path, temp_dir)
        
        if not image_paths:
            return False
        
        # Process and enhance images
        enhanced_paths = process_and_save_enhanced_images(image_paths, temp_dir)
        
        # Convert enhanced images back to PDF
        success = images_to_pdf(enhanced_paths, output_pdf_path)
        
        return success
        
    finally:
        # Clean up temporary files
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def get_user_choice():
    """Get user's rotation choice"""
    print("\nPDF Rotation and Enhancement Tool")
    print("=" * 40)
    print("Choose rotation option:")
    print("1. Rotate 90 degrees LEFT (counter-clockwise)")
    print("2. Rotate 90 degrees RIGHT (clockwise)")
    print("3. No rotation - just enhance quality")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            if choice in ['1', '2', '3', '4']:
                return choice
            else:
                print("Invalid choice. Please enter 1, 2, 3, or 4.")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return '4'

def get_page_range():
    """Get page range from user"""
    print("\nPage extraction options:")
    print("1. Extract all pages")
    print("2. Extract specific page range")
    
    while True:
        try:
            choice = input("Enter your choice (1-2): ").strip()
            if choice == '1':
                return None
            elif choice == '2':
                start = int(input("Enter start page number: "))
                end = int(input("Enter end page number: "))
                return (start, end)
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except ValueError:
            print("Please enter valid page numbers.")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return None

def main():
    """Main program function"""
    # Get input PDF file
    input_pdf = input("Enter path to input PDF file: ").strip().strip('"')
    
    if not os.path.exists(input_pdf):
        print(f"Error: File '{input_pdf}' not found.")
        return
    
    base_name = os.path.splitext(os.path.basename(input_pdf))[0]
    
    # Get user choice for rotation
    choice = get_user_choice()
    
    if choice == '4':
        print("Goodbye!")
        return
    
    # Create working directories
    work_dir = "pdf_processing_work"
    os.makedirs(work_dir, exist_ok=True)
    
    try:
        if choice == '3':
            # No rotation - just enhance
            print("Enhancing PDF quality...")
            output_pdf = f"{base_name}_enhanced.pdf"
            success = enhance_pdf_quality(input_pdf, output_pdf)
            
            if success:
                print(f"✅ Enhanced PDF saved: {output_pdf}")
            else:
                print("❌ Failed to enhance PDF")
        
        else:
            # Rotation required
            rotation_direction = 'left' if choice == '1' else 'right'
            
            # Step 1: Rotate PDF
            rotated_pdf = os.path.join(work_dir, f"{base_name}_rotated.pdf")
            print(f"\nStep 1: Rotating PDF {rotation_direction}...")
            
            if not rotate_pdf_pages(input_pdf, rotated_pdf, rotation_direction):
                print("❌ Failed to rotate PDF")
                return
            
            # Step 2: Get page range for extraction
            page_range = get_page_range()
            
            if page_range is None:
                print("Processing all pages...")
            else:
                print(f"Processing pages {page_range[0]} to {page_range[1]}...")
            
            # Step 3: Extract pages
            extract_dir = os.path.join(work_dir, "extracted_pages")
            os.makedirs(extract_dir, exist_ok=True)
            
            print(f"\nStep 2: Extracting pages...")
            extracted_files = extract_pages_from_pdf(rotated_pdf, extract_dir, page_range)
            
            if not extracted_files:
                print("❌ Failed to extract pages")
                return
            
            # Step 4: Enhance each extracted page and combine
            print(f"\nStep 3: Enhancing PDF quality...")
            enhanced_files = []
            
            for i, page_file in enumerate(extracted_files):
                page_num = i + 1
                enhanced_page = os.path.join(extract_dir, f"enhanced_page_{page_num:03d}.pdf")
                
                print(f"Enhancing page {page_num}...")
                if enhance_pdf_quality(page_file, enhanced_page):
                    enhanced_files.append(enhanced_page)
                else:
                    print(f"Warning: Failed to enhance page {page_num}, using original")
                    enhanced_files.append(page_file)
            
            # Step 5: Combine enhanced pages into final PDF
            print(f"\nStep 4: Combining enhanced pages...")
            final_pdf = f"{base_name}_rotated_{rotation_direction}_enhanced.pdf"
            
            writer = PdfWriter()
            for enhanced_file in enhanced_files:
                reader = PdfReader(enhanced_file)
                for page in reader.pages:
                    writer.add_page(page)
            
            with open(final_pdf, 'wb') as output_file:
                writer.write(output_file)
            
            print(f"✅ Final enhanced PDF saved: {final_pdf}")
    
    finally:
        # Clean up working directory
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
            print("🧹 Cleaned up temporary files")

if __name__ == "__main__":
    print("PDF Rotation and Enhancement Tool")
    print("Requires: pip install pypdf opencv-python pdf2image pillow img2pdf numpy")
    print()
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
