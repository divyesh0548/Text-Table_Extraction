import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import os

def enhance_pdf_tables(
    pdf_path: str,
    dpi: int = 300,
    line_strength: int = 2,
    show_progress: bool = True
    ):

    # Temporary folder for page images
    temp_dir = "temp_pdf_images"
    os.makedirs(temp_dir, exist_ok=True)

    # Convert PDF pages to images
    pages = convert_from_path(pdf_path, dpi=dpi)
    enhanced_images = []

    for i, page in enumerate(pages):
        if show_progress:
            print(f"[INFO] Processing page {i+1}/{len(pages)}...")

        # Convert PIL image to OpenCV format
        img = np.array(page)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # --- Step 1: Grayscale ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- Step 2: Improve contrast ---
        gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)

        # --- Step 3: Reduce noise ---
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # --- Step 4: Adaptive threshold ---
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            15, 10
        )

        # --- Step 5: Detect horizontal & vertical lines ---
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

        detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

        # --- Step 6: Combine and strengthen lines ---
        table_mask = cv2.add(detect_horizontal, detect_vertical)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_strength, line_strength))
        table_mask = cv2.dilate(table_mask, kernel, iterations=2)

        # --- Step 7: Smooth edges and repair small gaps ---
        table_mask = cv2.morphologyEx(table_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # --- Step 8: Invert and merge with original ---
        inverted_mask = cv2.bitwise_not(table_mask)
        enhanced_gray = cv2.bitwise_and(gray, gray, mask=inverted_mask)
        enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

        # Save temporary enhanced image
        enhanced_path = os.path.join(temp_dir, f"enhanced_page_{i+1}.png")
        cv2.imwrite(enhanced_path, enhanced_bgr)
        enhanced_images.append(Image.open(enhanced_path).convert("RGB"))

    # --- Step 9: Save final enhanced PDF in current directory ---
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    pdf_output_path = os.path.join(os.getcwd(), f"{base_name}_enhanced.pdf")

    enhanced_images[0].save(pdf_output_path, save_all=True, append_images=enhanced_images[1:])

    if show_progress:
        print(f"[DONE] Enhanced PDF saved as: {pdf_output_path}")

    # --- Step 10: Cleanup temporary files ---
    for file in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, file)
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"[WARN] Could not delete {file_path}: {e}")

    os.rmdir(temp_dir)

    if show_progress:
        print("[CLEANUP] Temporary images deleted.")

    return pdf_output_path


# Example usage
if __name__ == "__main__":
    input_pdf = "invoice5.pdf"  # 👈 Replace with your PDF path
    enhance_pdf_tables(input_pdf, dpi=300)
