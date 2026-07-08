import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Divyesh Parmar\Downloads\Tesserect OCR\tesseract.exe'

def extract_with_tesseract(pdf_path):
    pages = convert_from_path(pdf_path, dpi=300)
    
    text = ""
    for page in pages:
        # Enhanced preprocessing for better accuracy
        img = np.array(page)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Extract text with custom configuration
        custom_config = r'--oem 3 --psm 6'
        page_text = pytesseract.image_to_string(thresh, config=custom_config)
        text += page_text + "\n"
    
    return text

extracted_text = extract_with_tesseract("invoice1.pdf")
print(extracted_text)