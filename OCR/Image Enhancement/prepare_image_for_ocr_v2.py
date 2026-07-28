"""
OCR image preprocessing with bold text and restored table lines.

This version creates seven outputs, including:

    07_final_bold_text_with_table_lines.png

The final image combines:
- bold text from output 06; and
- detected table lines from output 04.

Edit the CONFIGURATION section and run:
    python prepare_image_for_ocr_v2.py

Install:
    pip install opencv-python numpy
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_IMAGE = SCRIPT_DIR / "MBP 1 MK 2026_page_2_table_1_crop.png"
OUTPUT_DIR = SCRIPT_DIR / "ocr_preprocessed"

UPSCALE_FACTOR = 2.0
ENABLE_DESKEW = True

ADAPTIVE_BLOCK_SIZE = 35
ADAPTIVE_C = 15

TEXT_BOLD_ITERATIONS = 1
TEXT_BOLD_KERNEL_SIZE = 2

REMOVE_TABLE_LINES = True
HORIZONTAL_LINE_SCALE = 25
VERTICAL_LINE_SCALE = 25

# Used only while removing table lines from the text copy.
# The unpadded table-line mask is restored in output 07.
LINE_REMOVAL_PADDING = 1

MIN_COMPONENT_AREA = 4

# =============================================================================
# END CONFIGURATION
# =============================================================================


def make_odd(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 == 1 else value + 1


def load_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Input image not found: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"OpenCV could not read the image: {path}")

    return image


def estimate_skew_angle(gray: np.ndarray) -> float:
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )[1]

    coordinates = np.column_stack(np.where(binary > 0))

    if len(coordinates) < 100:
        return 0.0

    angle = cv2.minAreaRect(
        coordinates[:, ::-1].astype(np.float32)
    )[-1]

    if angle < -45:
        angle = 90 + angle

    correction = -angle

    if abs(correction) > 10:
        return 0.0

    return correction


def rotate_without_cropping(
    image: np.ndarray,
    angle: float,
) -> np.ndarray:
    if abs(angle) < 0.05:
        return image.copy()

    height, width = image.shape[:2]
    centre = (width / 2.0, height / 2.0)

    matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)

    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])

    new_width = int(height * sine + width * cosine)
    new_height = int(height * cosine + width * sine)

    matrix[0, 2] += new_width / 2 - centre[0]
    matrix[1, 2] += new_height / 2 - centre[1]

    return cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def normalize_background(gray: np.ndarray) -> np.ndarray:
    sigma = max(15, min(gray.shape[:2]) // 40)

    background = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
    )

    return cv2.divide(gray, background, scale=255)


def enhance_grayscale(gray: np.ndarray) -> np.ndarray:
    denoised = cv2.fastNlMeansDenoising(
        gray,
        None,
        h=7,
        templateWindowSize=7,
        searchWindowSize=21,
    )

    normalized = normalize_background(denoised)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )
    contrasted = clahe.apply(normalized)

    blurred = cv2.GaussianBlur(
        contrasted,
        (0, 0),
        1.0,
    )

    return cv2.addWeighted(
        contrasted,
        1.5,
        blurred,
        -0.5,
        0,
    )


def make_binary_ink_mask(gray: np.ndarray) -> np.ndarray:
    """
    Return white foreground ink on a black background.
    """
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        make_odd(ADAPTIVE_BLOCK_SIZE),
        ADAPTIVE_C,
    )


def detect_table_lines(
    ink_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return:
        raw_line_mask:
            Detected table lines without removal padding.
        padded_line_mask:
            Slightly expanded line mask used to remove borders from text.
    """
    height, width = ink_mask.shape

    horizontal_length = max(
        20,
        width // max(HORIZONTAL_LINE_SCALE, 1),
    )
    vertical_length = max(
        20,
        height // max(VERTICAL_LINE_SCALE, 1),
    )

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (horizontal_length, 1),
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, vertical_length),
    )

    horizontal = cv2.morphologyEx(
        ink_mask,
        cv2.MORPH_OPEN,
        horizontal_kernel,
        iterations=1,
    )

    vertical = cv2.morphologyEx(
        ink_mask,
        cv2.MORPH_OPEN,
        vertical_kernel,
        iterations=1,
    )

    raw_line_mask = cv2.bitwise_or(
        horizontal,
        vertical,
    )

    padded_line_mask = raw_line_mask.copy()

    if LINE_REMOVAL_PADDING > 0:
        padding_size = LINE_REMOVAL_PADDING * 2 + 1
        padding_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (padding_size, padding_size),
        )

        padded_line_mask = cv2.dilate(
            raw_line_mask,
            padding_kernel,
            iterations=1,
        )

    return raw_line_mask, padded_line_mask


def remove_small_components(
    mask: np.ndarray,
    minimum_area: int,
) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8),
        connectivity=8,
    )

    cleaned = np.zeros_like(mask)

    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])

        if area >= minimum_area:
            cleaned[labels == label] = 255

    return cleaned


def thicken_text(text_mask: np.ndarray) -> np.ndarray:
    size = max(1, int(TEXT_BOLD_KERNEL_SIZE))

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (size, size),
    )

    return cv2.dilate(
        text_mask,
        kernel,
        iterations=max(0, int(TEXT_BOLD_ITERATIONS)),
    )


def foreground_mask_to_document(mask: np.ndarray) -> np.ndarray:
    """
    Convert white foreground on black into black content on white.
    """
    return cv2.bitwise_not(mask)


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(path), image)

    if not success:
        raise RuntimeError(f"Failed to save image: {path}")

    if not path.exists():
        raise RuntimeError(
            f"OpenCV reported success, but output file is missing: {path}"
        )

    print(f"Created: {path.name}")


def process_image() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image = load_image(INPUT_IMAGE)

    if ENABLE_DESKEW:
        initial_gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )
        skew_angle = estimate_skew_angle(initial_gray)
        image = rotate_without_cropping(image, skew_angle)
        print(f"Skew correction: {skew_angle:.3f} degrees")

    if UPSCALE_FACTOR != 1.0:
        image = cv2.resize(
            image,
            None,
            fx=UPSCALE_FACTOR,
            fy=UPSCALE_FACTOR,
            interpolation=cv2.INTER_CUBIC,
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    enhanced_gray = enhance_grayscale(gray)
    ink_mask = make_binary_ink_mask(enhanced_gray)

    raw_line_mask = np.zeros_like(ink_mask)
    padded_line_mask = np.zeros_like(ink_mask)

    if REMOVE_TABLE_LINES:
        raw_line_mask, padded_line_mask = detect_table_lines(
            ink_mask
        )

        text_mask = cv2.bitwise_and(
            ink_mask,
            cv2.bitwise_not(padded_line_mask),
        )
    else:
        text_mask = ink_mask.copy()

    text_mask = remove_small_components(
        text_mask,
        MIN_COMPONENT_AREA,
    )

    bold_text_mask = thicken_text(text_mask)

    # Combine bold text and the original, unpadded table-line mask.
    final_combined_mask = cv2.bitwise_or(
        bold_text_mask,
        raw_line_mask,
    )

    binary_image = foreground_mask_to_document(ink_mask)
    line_removed_image = foreground_mask_to_document(text_mask)
    bold_text_image = foreground_mask_to_document(bold_text_mask)
    table_lines_image = foreground_mask_to_document(raw_line_mask)
    final_image = foreground_mask_to_document(
        final_combined_mask
    )

    save_image(
        OUTPUT_DIR / "01_upscaled_original.png",
        image,
    )
    save_image(
        OUTPUT_DIR / "02_enhanced_grayscale.png",
        enhanced_gray,
    )
    save_image(
        OUTPUT_DIR / "03_binary.png",
        binary_image,
    )
    save_image(
        OUTPUT_DIR / "04_detected_table_lines.png",
        table_lines_image,
    )
    save_image(
        OUTPUT_DIR / "05_table_lines_removed.png",
        line_removed_image,
    )
    save_image(
        OUTPUT_DIR / "06_bold_text_for_ocr.png",
        bold_text_image,
    )
    save_image(
        OUTPUT_DIR / "07_final_bold_text_with_table_lines.png",
        final_image,
    )

    final_path = (
        OUTPUT_DIR
        / "07_final_bold_text_with_table_lines.png"
    )

    print("\nProcessing completed successfully.")
    print(f"Final image: {final_path.resolve()}")


if __name__ == "__main__":
    process_image()
