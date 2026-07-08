
import cv2
import numpy as np
import pandas as pd
import pytesseract
import os
import sys
from typing import List, Tuple, Optional
import re

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Divyesh Parmar\Downloads\Tesserect OCR\tesseract.exe'

class ImageTableExtractor:
    """
    A comprehensive class for extracting tables from images using OpenCV and Tesseract OCR.
    This approach uses computer vision to detect table structure and OCR for text extraction.
    """

    def __init__(self, tesseract_cmd: Optional[str] = None):
        """
        Initialize the table extractor.

        Args:
            tesseract_cmd: Path to tesseract executable (if not in PATH)
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        # Configure Tesseract for better table OCR
        self.tesseract_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!?@#$%^&*()_+-=[]{}|\:";\'<>?/~` '

    def preprocess_image(self, image: np.ndarray, enhance: bool = True) -> np.ndarray:
        """
        Preprocess the image for better table detection and OCR.

        Args:
            image: Input image as numpy array
            enhance: Whether to apply image enhancement

        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if enhance:
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)

            # Noise removal
            gray = cv2.medianBlur(gray, 3)

            # Sharpening
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]])
            gray = cv2.filter2D(gray, -1, kernel)

        # Adaptive thresholding for better text extraction
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        return binary

    def detect_table_lines(self, binary_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect horizontal and vertical lines in the image.

        Args:
            binary_image: Binary image

        Returns:
            Tuple of (horizontal_lines_image, vertical_lines_image)
        """
        # Create kernels for line detection
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

        # Detect horizontal lines
        horizontal_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, horizontal_kernel)

        # Detect vertical lines  
        vertical_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, vertical_kernel)

        return horizontal_lines, vertical_lines

    def find_table_contours(self, binary_image: np.ndarray) -> List[np.ndarray]:
        """
        Find table contours in the image.

        Args:
            binary_image: Binary image

        Returns:
            List of contours representing potential table regions
        """
        # Detect lines
        horizontal_lines, vertical_lines = self.detect_table_lines(binary_image)

        # Combine horizontal and vertical lines
        table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)

        # Dilate to connect nearby lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        table_mask = cv2.dilate(table_mask, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by area and aspect ratio
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 0.2 < aspect_ratio < 10:  # Reasonable aspect ratio for tables
                    valid_contours.append(contour)

        return valid_contours

    def extract_table_cells(self, image: np.ndarray, binary_image: np.ndarray) -> List[List[dict]]:
        """
        Extract individual cells from detected table regions.

        Args:
            image: Original image
            binary_image: Binary processed image

        Returns:
            List of tables, each containing rows of cell information
        """
        contours = self.find_table_contours(binary_image)
        tables = []

        for contour in contours:
            # Get bounding rectangle of table
            x, y, w, h = cv2.boundingRect(contour)
            table_region = binary_image[y:y+h, x:x+w]
            original_region = image[y:y+h, x:x+w]

            # Detect grid structure
            cells = self._detect_grid_cells(table_region, original_region, x, y)

            if cells:
                tables.append(cells)

        return tables

    def _detect_grid_cells(self, table_region: np.ndarray, original_region: np.ndarray, 
                          offset_x: int, offset_y: int) -> List[List[dict]]:
        """
        Detect individual cells within a table region using line intersection analysis.

        Args:
            table_region: Binary image of table region
            original_region: Original image of table region
            offset_x, offset_y: Offset coordinates of the table region

        Returns:
            2D list of cell information dictionaries
        """
        # Detect horizontal and vertical lines
        horizontal_lines, vertical_lines = self.detect_table_lines(table_region)

        # Find line coordinates
        h_lines = self._find_line_coordinates(horizontal_lines, horizontal=True)
        v_lines = self._find_line_coordinates(vertical_lines, horizontal=False)

        if len(h_lines) < 2 or len(v_lines) < 2:
            return []

        # Sort lines
        h_lines = sorted(h_lines)
        v_lines = sorted(v_lines)

        # Extract cells
        cells = []
        for i in range(len(h_lines) - 1):
            row = []
            for j in range(len(v_lines) - 1):
                y1, y2 = h_lines[i], h_lines[i + 1]
                x1, x2 = v_lines[j], v_lines[j + 1]

                # Extract cell content
                cell_img = original_region[y1:y2, x1:x2]
                text = self._extract_text_from_cell(cell_img)

                cell_info = {
                    'text': text.strip(),
                    'bbox': (offset_x + x1, offset_y + y1, x2 - x1, y2 - y1),
                    'row': i,
                    'col': j
                }
                row.append(cell_info)

            if row:  # Only add non-empty rows
                cells.append(row)

        return cells

    def _find_line_coordinates(self, line_img: np.ndarray, horizontal: bool = True) -> List[int]:
        """
        Find coordinates of lines in the image.

        Args:
            line_img: Binary image containing lines
            horizontal: True for horizontal lines, False for vertical lines

        Returns:
            List of line coordinates
        """
        if horizontal:
            # Sum along horizontal axis to find horizontal lines
            line_sum = np.sum(line_img, axis=1)
        else:
            # Sum along vertical axis to find vertical lines
            line_sum = np.sum(line_img, axis=0)

        # Find peaks (lines)
        threshold = np.max(line_sum) * 0.3
        lines = []

        for i, val in enumerate(line_sum):
            if val > threshold:
                lines.append(i)

        # Group nearby lines
        if not lines:
            return []

        grouped_lines = [lines[0]]
        for line in lines[1:]:
            if line - grouped_lines[-1] > 10:  # Minimum gap between lines
                grouped_lines.append(line)

        return grouped_lines

    def _extract_text_from_cell(self, cell_img: np.ndarray) -> str:
        """
        Extract text from a single cell using OCR.

        Args:
            cell_img: Image of the cell

        Returns:
            Extracted text
        """
        if cell_img.size == 0:
            return ""

        # Resize cell for better OCR if it's too small
        # height, width = cell_img.shape
        # if height < 20 or width < 20:
        #     scale_factor = max(2, 20 // min(height, width))
        #     cell_img = cv2.resize(cell_img, None, fx=scale_factor, fy=scale_factor, 
        #                         interpolation=cv2.INTER_CUBIC)

        # Additional preprocessing for cell
        cell_img = cv2.medianBlur(cell_img, 3)

        try:
            # Use Tesseract to extract text
            text = pytesseract.image_to_string(cell_img, config=self.tesseract_config)
            return text
        except Exception as e:
            print(f"OCR error for cell: {e}")
            return ""

    def cells_to_dataframe(self, cells: List[List[dict]]) -> pd.DataFrame:
        """
        Convert extracted cells to a pandas DataFrame.

        Args:
            cells: 2D list of cell information

        Returns:
            DataFrame representation of the table
        """
        if not cells or not cells[0]:
            return pd.DataFrame()

        # Create a 2D array of text values
        max_cols = max(len(row) for row in cells)
        table_data = []

        for row in cells:
            row_data = []
            for j in range(max_cols):
                if j < len(row):
                    text = row[j]['text'].strip()
                    # Clean text
                    text = re.sub(r'\s+', ' ', text)
                    text = re.sub(r'[^\w\s.,!?@#$%^&*()_+\-=\[\]{}|\\:";\'<>?/~`]', '', text)
                    row_data.append(text)
                else:
                    row_data.append('')
            table_data.append(row_data)

        return pd.DataFrame(table_data)

    def extract_tables_from_image(self, image_path: str, output_dir: str = None, 
                                 save_debug_images: bool = False) -> List[pd.DataFrame]:
        """
        Main method to extract tables from an image.

        Args:
            image_path: Path to input image
            output_dir: Directory to save results and debug images
            save_debug_images: Whether to save intermediate processing images

        Returns:
            List of DataFrames representing extracted tables
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        print(f"Processing image: {image_path}")
        print(f"Image shape: {image.shape}")

        # Preprocess image
        binary_image = self.preprocess_image(image, enhance=True)

        # Save debug images if requested
        if save_debug_images and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(os.path.join(output_dir, "01_binary.png"), binary_image)

            # Save line detection results
            h_lines, v_lines = self.detect_table_lines(binary_image)
            cv2.imwrite(os.path.join(output_dir, "02_horizontal_lines.png"), h_lines)
            cv2.imwrite(os.path.join(output_dir, "03_vertical_lines.png"), v_lines)

            table_mask = cv2.addWeighted(h_lines, 0.5, v_lines, 0.5, 0.0)
            cv2.imwrite(os.path.join(output_dir, "04_table_mask.png"), table_mask)

        # Extract table cells
        tables_cells = self.extract_table_cells(image, binary_image)

        print(f"Found {len(tables_cells)} table(s)")

        # Convert to DataFrames
        dataframes = []
        for i, cells in enumerate(tables_cells):
            df = self.cells_to_dataframe(cells)
            print(f"Table {i+1} shape: {df.shape}")
            dataframes.append(df)

            # Save to CSV if output directory specified
            if output_dir and not df.empty:
                os.makedirs(output_dir, exist_ok=True)
                csv_path = os.path.join(output_dir, f"table_{i+1}.csv")
                df.to_csv(csv_path, index=False)
                print(f"Saved table {i+1} to: {csv_path}")

        return dataframes

# Example usage and testing function
def main():
    """
    Example usage of the ImageTableExtractor class
    """
    # Example paths (update with your actual paths)
    image_path = "page1.jpg"  # Your input image
    output_dir = "extracted_tables"  # Output directory

    # Initialize extractor
    # On Windows, you might need to specify tesseract path:
    # extractor = ImageTableExtractor(tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe')
    extractor = ImageTableExtractor()

    if not os.path.exists(image_path):
        print(f"Example usage - please replace '{image_path}' with your actual image file")
        print("\nUsage:")
        print("extractor = ImageTableExtractor()")
        print("tables = extractor.extract_tables_from_image('your_image.jpg', 'output_dir')")
        return

    try:
        # Extract tables
        tables = extractor.extract_tables_from_image(
            image_path=image_path,
            output_dir=output_dir,
            save_debug_images=True  # Save intermediate processing images
        )

        # Display results
        print(f"\n=== EXTRACTION RESULTS ===")
        for i, df in enumerate(tables):
            print(f"\nTable {i+1}:")
            print(f"Shape: {df.shape}")
            if not df.empty:
                print("Preview:")
                print(df.head())
                print("-" * 50)
            else:
                print("Empty table")

    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
