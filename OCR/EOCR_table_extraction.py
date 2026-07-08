import easyocr
from pdf2image import convert_from_path
import numpy as np
import cv2
import pandas as pd
from io import BytesIO
import os

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

def detect_tables_in_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to detect table-like grid structures
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Detect horizontal and vertical lines using Hough Transform
    horizontal_lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, threshold=200, minLineLength=100, maxLineGap=10)
    vertical_lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, threshold=200, minLineLength=100, maxLineGap=10)

    # Draw lines on the image to visualize table detection
    for line in horizontal_lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for line in vertical_lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return image, horizontal_lines, vertical_lines

def extract_table_data(image, horizontal_lines, vertical_lines):
    # Sort the lines by their coordinates (you may need to fine-tune this part)
    horizontal_lines = sorted(horizontal_lines, key=lambda x: x[0][1])
    vertical_lines = sorted(vertical_lines, key=lambda x: x[0][0])

    table_data = []
    
    for i in range(len(horizontal_lines) - 1):
        row_data = []
        for j in range(len(vertical_lines) - 1):
            # Crop the region where the table cell is
            x1, y1, x2, y2 = vertical_lines[j][0][0], horizontal_lines[i][0][1], vertical_lines[j + 1][0][0], horizontal_lines[i + 1][0][1]
            cell_img = image[y1:y2, x1:x2]

            # Use EasyOCR to extract text from the cell image
            reader = easyocr.Reader(['en'])
            result = reader.readtext(cell_img, detail=0)
            row_data.append(" ".join(result))
        
        table_data.append(row_data)
    
    return table_data

def export_to_excel(table_data, output_filename="output.xlsx"):
    # Convert the table data into a pandas DataFrame
    df = pd.DataFrame(table_data)
    
    # Export the data to an Excel file
    df.to_excel(output_filename, index=False, header=False)

# Example usage
pdf_path = "CH-012026-0062-INVOICE.pdf"
page_numbers = [1]  # Extract from page 1

# Step 1: Extract text using EasyOCR
extracted_text = extract_with_easyocr(pdf_path, page_numbers)
print(extracted_text)

# Step 2: Process the image to detect tables
pages = convert_from_path(pdf_path, dpi=300)
image = np.array(pages[page_numbers[0] - 1])  # Convert page to numpy array

# Detect table-like structures (lines)
image_with_lines, horizontal_lines, vertical_lines = detect_tables_in_image(image)

# Step 3: Extract the data from detected table cells
table_data = extract_table_data(image, horizontal_lines, vertical_lines)

# Step 4: Export the extracted table data to Excel
export_to_excel(table_data, "output_table.xlsx")
