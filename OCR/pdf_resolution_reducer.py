import os
from pdf2image import convert_from_path
from PIL import Image
import img2pdf

def reduce_pdf_resolution(input_pdf_path, output_pdf_path, target_dpi=72):
    # Convert PDF to images with original resolution
    images = convert_from_path(input_pdf_path)
    
    reduced_images = []
    for i, img in enumerate(images):
        # Calculate new size based on target DPI and original DPI
        orig_dpi = img.info.get('dpi', (300,300))[0]
        scale_factor = target_dpi / orig_dpi
        
        new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
        
        # Resize image to new target resolution
        reduced_img = img.resize(new_size, Image.LANCZOS)
        
        # Save reduced image temporarily in JPEG format to reduce size
        temp_img_path = f"temp_page_{i}.jpg"
        reduced_img.save(temp_img_path, "JPEG", quality=85)
        reduced_images.append(temp_img_path)
    
    # Convert list of JPEG images back to PDF
    with open(output_pdf_path, "wb") as f:
        f.write(img2pdf.convert(reduced_images))
    
    # Clean up temporary images
    for img_path in reduced_images:
        os.remove(img_path)

if __name__ == "__main__":
    input_pdf = "invoice5_enhanced.pdf"        # Path to your input PDF
    output_pdf = "reduced_output.pdf"  # Output PDF path
    target_resolution = 56         # DPI to reduce to (default 72 dpi)
    
    reduce_pdf_resolution(input_pdf, output_pdf, target_resolution)
    print(f"Reduced resolution PDF saved as {output_pdf}")
