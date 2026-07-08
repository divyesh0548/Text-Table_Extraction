from paddleocr import PaddleOCR
from pdf2image import convert_from_path
import numpy as np

def extract_with_paddleocr(pdf_path, page_numbers=None):
    pages = convert_from_path(pdf_path, dpi=300)
    
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    
    if page_numbers is None:
        page_numbers = list(range(1, len(pages) + 1))
    
    text = ""
    for page_num in page_numbers:
        if 1 <= page_num <= len(pages):
            # Convert PIL Image to numpy array
            page_image = np.array(pages[page_num - 1])
            
            results = ocr.ocr(page_image)
            
            if results and results[0]:
                for line in results[0]:
                    text += line[1][0] + " "
                text += "\n"
    
    return text


# Usage Examples:
extracted_text = extract_with_paddleocr("invoice1.pdf", [1])
print(extracted_text)
