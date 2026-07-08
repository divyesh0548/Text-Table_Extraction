import easyocr
from pdf2image import convert_from_path
import numpy as np

def extract_with_easyocr(pdf_path, page_numbers=None):
    pages = convert_from_path(pdf_path, dpi=300)
    reader = easyocr.Reader(['en'])
    
    if page_numbers is None:
        page_numbers = list(range(1, len(pages) + 1))
    
    text = ""
    for page_num in page_numbers:
        if 1 <= page_num <= len(pages):
            img_array = np.array(pages[page_num - 1])  # convert PIL to numpy array
            results = reader.readtext(img_array, detail=0)
            text += " ".join(results) + "\n"
        else:
            print(f"Warning: Page {page_num} is out of range and will be skipped.")
    
    return text

# Example usage:
extracted_text = extract_with_easyocr("AK.pdf", [1])  # Extract pages 1 and 3
print(extracted_text)
    