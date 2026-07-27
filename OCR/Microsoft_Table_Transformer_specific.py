from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Callable

import fitz  # PyMuPDF
import pytesseract
import torch
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from PIL import Image, ImageDraw, ImageOps
from transformers import AutoImageProcessor, TableTransformerForObjectDetection


# =============================================================================
# CONFIGURATION
# =============================================================================

# INPUT_PDF = Path("Current-Test/MBP 1 RKB 2026.pdf")
INPUT_PDF = Path(r"Aarti-Mandays May 26.pdf")
OUTPUT_XLSX = Path("extracted_tables.xlsx")
DEBUG_DIR = Path(__file__).resolve().parent / "table_debug"

DETECTION_MODEL_NAME = "microsoft/table-transformer-detection"
STRUCTURE_MODEL_NAME = (
    "microsoft/table-transformer-structure-recognition-v1.1-all"
)

PDF_DPI = 250
DETECTION_THRESHOLD = 0.75
STRUCTURE_THRESHOLD = 0.60

TABLE_PADDING = 12
CELL_INSET = 2

# Uncomment and update this on Windows only when tesseract.exe is not in PATH.
# This is unnecessary after replacing default_ocr() with your existing OCR.
#
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Users\Divyesh Parmar\Downloads\Tesserect OCR\tesseract.exe"
)


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box: tuple[float, float, float, float]


# =============================================================================
# GENERAL GEOMETRY
# =============================================================================

def safe_sheet_name(name: str) -> str:
    """Return a legal Excel worksheet name."""
    name = re.sub(r"[:\\/?*\[\]]", "_", name)
    return name[:31] or "Table"


def clamp_box(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    padding: int = 0,
     ) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box

    return (
        max(0, int(round(x1)) - padding),
        max(0, int(round(y1)) - padding),
        min(width, int(round(x2)) + padding),
        min(height, int(round(y2)) + padding),
    )


def box_area(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
    ) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
    ) -> float:
    intersection = intersection_area(box_a, box_b)
    union = box_area(box_a) + box_area(box_b) - intersection
    return intersection / union if union > 0 else 0.0


def remove_duplicate_detections(
    detections: list[Detection],
    iou_threshold: float,
    ) -> list[Detection]:
    """
    Remove near-identical model boxes while retaining the highest-confidence box.
    """
    kept: list[Detection] = []

    for item in sorted(detections, key=lambda detection: detection.score, reverse=True):
        duplicate = any(
            item.label == old.label
            and box_iou(item.box, old.box) >= iou_threshold
            for old in kept
        )
        if not duplicate:
            kept.append(item)

    return kept


def axis_length(
    detection: Detection,
    axis: str,
) -> float:
    if axis == "row":
        return max(0.0, detection.box[3] - detection.box[1])
    return max(0.0, detection.box[2] - detection.box[0])


def axis_center(
    detection: Detection,
    axis: str,
) -> float:
    if axis == "row":
        return (detection.box[1] + detection.box[3]) / 2
    return (detection.box[0] + detection.box[2]) / 2


def orthogonal_overlap_ratio(
    first: Detection,
    second: Detection,
    axis: str,
) -> float:
    if axis == "row":
        start = max(first.box[0], second.box[0])
        end = min(first.box[2], second.box[2])
        base = min(first.box[2] - first.box[0], second.box[2] - second.box[0])
    else:
        start = max(first.box[1], second.box[1])
        end = min(first.box[3], second.box[3])
        base = min(first.box[3] - first.box[1], second.box[3] - second.box[1])

    if base <= 0:
        return 0.0
    return max(0.0, end - start) / base


def collapse_nearby_structure_detections(
    detections: list[Detection],
    axis: str,
    center_tolerance_ratio: float = 0.4,
    overlap_ratio_threshold: float = 0.9,
) -> list[Detection]:
    """
    Remove nearly identical row/column detections that survive IoU-based filtering.
    """
    kept: list[Detection] = []

    for item in sorted(detections, key=lambda detection: detection.score, reverse=True):
        item_length = axis_length(item, axis)
        duplicate = False

        for existing in kept:
            existing_length = axis_length(existing, axis)
            tolerance = max(
                3.0,
                min(item_length, existing_length) * center_tolerance_ratio,
            )
            centers_are_close = (
                abs(axis_center(item, axis) - axis_center(existing, axis))
                <= tolerance
            )
            overlap_ratio = orthogonal_overlap_ratio(item, existing, axis)

            if centers_are_close and overlap_ratio >= overlap_ratio_threshold:
                duplicate = True
                break

        if not duplicate:
            kept.append(item)

    return kept


# =============================================================================
# PDF RENDERING
# =============================================================================

def render_pdf_pages(pdf_path: Path, dpi: int) -> list[Image.Image]:
    """
    Render all PDF pages to RGB PIL images.

    A higher DPI improves small-text OCR and structure recognition, but increases
    processing time and memory usage.
    """
    images: list[Image.Image] = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    with fitz.open(pdf_path) as document:
        for page in document:
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image = Image.frombytes(
                "RGB",
                (pixmap.width, pixmap.height),
                pixmap.samples,
            )
            images.append(image)

    return images


# =============================================================================
# TABLE TRANSFORMER
# =============================================================================

class TableTransformerEngine:
    def __init__(self) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Using device: {self.device}")

        self.detection_processor = AutoImageProcessor.from_pretrained(
            DETECTION_MODEL_NAME
        )
        self.detection_model = (
            TableTransformerForObjectDetection.from_pretrained(
                DETECTION_MODEL_NAME
            )
            .to(self.device)
            .eval()
        )

        self.structure_processor = AutoImageProcessor.from_pretrained(
            STRUCTURE_MODEL_NAME
        )
        # transformers>=4.4x expects both shortest_edge and longest_edge.
        # v1.1 structure configs only ship longest_edge and crash on resize.
        self._normalize_processor_size(self.detection_processor)
        self._normalize_processor_size(self.structure_processor)

        self.structure_model = (
            TableTransformerForObjectDetection.from_pretrained(
                STRUCTURE_MODEL_NAME
            )
            .to(self.device)
            .eval()
        )

    @staticmethod
    def _normalize_processor_size(processor) -> None:
        size = getattr(processor, "size", None)
        if not isinstance(size, dict):
            return

        if "shortest_edge" in size and "longest_edge" in size:
            return
        if "height" in size and "width" in size:
            return

        longest = int(size.get("longest_edge") or size.get("max_size") or 800)
        shortest = int(size.get("shortest_edge") or min(800, longest))
        processor.size = {
            "shortest_edge": shortest,
            "longest_edge": longest,
        }

    @staticmethod
    def _decode_results(
        result: dict,
        id2label: dict,
    ) -> list[Detection]:
        decoded: list[Detection] = []

        for score, label_tensor, box_tensor in zip(
            result["scores"],
            result["labels"],
            result["boxes"],
        ):
            label_id = int(label_tensor.item())
            label = id2label.get(
                label_id,
                id2label.get(str(label_id), str(label_id)),
            )

            decoded.append(
                Detection(
                    label=str(label).lower(),
                    score=float(score.item()),
                    box=tuple(
                        float(coordinate)
                        for coordinate in box_tensor.tolist()
                    ),
                )
            )

        return decoded

    @staticmethod
    def _move_inputs_to_device(inputs: dict, device: torch.device) -> dict:
        return {
            key: value.to(device)
            for key, value in inputs.items()
        }

    def detect_tables(self, page_image: Image.Image) -> list[Detection]:
        inputs = self.detection_processor(
            images=page_image,
            return_tensors="pt",
        )
        inputs = self._move_inputs_to_device(inputs, self.device)

        with torch.inference_mode():
            outputs = self.detection_model(**inputs)

        target_sizes = torch.tensor(
            [[page_image.height, page_image.width]],
            device=self.device,
        )

        result = self.detection_processor.post_process_object_detection(
            outputs,
            threshold=DETECTION_THRESHOLD,
            target_sizes=target_sizes,
        )[0]

        detections = self._decode_results(
            result,
            self.detection_model.config.id2label,
        )

        detections = [
            item
            for item in detections
            if item.label in {"table", "table rotated"}
        ]

        return remove_duplicate_detections(
            detections,
            iou_threshold=0.70,
        )

    def recognize_structure(
        self,
        table_image: Image.Image,
    ) -> list[Detection]:
        inputs = self.structure_processor(
            images=table_image,
            return_tensors="pt",
        )
        inputs = self._move_inputs_to_device(inputs, self.device)

        with torch.inference_mode():
            outputs = self.structure_model(**inputs)

        target_sizes = torch.tensor(
            [[table_image.height, table_image.width]],
            device=self.device,
        )

        result = self.structure_processor.post_process_object_detection(
            outputs,
            threshold=STRUCTURE_THRESHOLD,
            target_sizes=target_sizes,
        )[0]

        return self._decode_results(
            result,
            self.structure_model.config.id2label,
        )


# =============================================================================
# OCR ADAPTER
# =============================================================================

def cell_has_visible_ink(
    cell_image: Image.Image,
    dark_threshold: int = 170,
    minimum_ink_ratio: float = 0.005,
    minimum_dark_pixels: int = 8,
) -> bool:
    """
    Return True when a cell crop contains enough dark pixels to justify OCR.

    Ignores a thin border so table grid lines alone do not count as content.
    Thresholds are intentionally soft so faint/small text is still OCR'd.
    """
    gray = cell_image.convert("L")
    width, height = gray.size
    if width < 2 or height < 2:
        return False

    inset_x = max(1, min(3, width // 10))
    inset_y = max(1, min(3, height // 10))
    if width > 2 * inset_x and height > 2 * inset_y:
        gray = gray.crop(
            (inset_x, inset_y, width - inset_x, height - inset_y)
        )

    histogram = gray.histogram()
    total_pixels = sum(histogram) or 1
    dark_pixels = sum(histogram[:dark_threshold])
    if dark_pixels < minimum_dark_pixels:
        return False
    return (dark_pixels / total_pixels) >= minimum_ink_ratio


def ocr_with_confidence(
    image: Image.Image,
    config: str,
    ) -> tuple[str, float]:
    """Run Tesseract and return recognized text plus mean word confidence."""
    data = pytesseract.image_to_data(
        image,
        config=config,
        output_type=pytesseract.Output.DICT,
    )

    texts: list[str] = []
    confidences: list[float] = []

    for text, confidence in zip(data.get("text", []), data.get("conf", [])):
        cleaned = str(text or "").strip()
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            continue

        if not cleaned or confidence_value < 0:
            continue

        texts.append(cleaned)
        confidences.append(confidence_value)

    if not confidences:
        return "", 0.0

    return " ".join(texts), sum(confidences) / len(confidences)


def is_gibberish_token(token: str) -> bool:
    """Heuristic check for OCR hallucinated alphabetic tokens."""
    word = token.lower()
    if not word.isalpha():
        return False

    protected = {
        "and", "the", "for", "not", "yes", "nil", "ltd", "pvt", "huf",
        "mr", "mrs", "ms", "dr", "no", "sr", "qty", "nos", "amt", "ref",
        "date", "name", "code", "type", "unit", "rate", "total", "page",
        "from", "with", "this", "that", "each", "paid", "free", "none",
        "cash", "bank", "bill", "item", "amount", "number", "invoice",
        "address", "details", "description", "particulars", "quantity",
        "value", "price", "weight", "remarks", "month", "year", "days",
    }
    if word in protected:
        return False

    # Keep common uppercase-looking codes/acronyms (INR, HSN, GST...).
    if token.isupper() and 2 <= len(token) <= 5:
        return False

    # Keep normal short Title Case tokens (names/labels); only reject tiny crumbs.
    if len(word) <= 2:
        return True

    vowels = sum(character in "aeiou" for character in word)
    if len(word) >= 4 and vowels == 0:
        return True

    # Very low vowel density is typical of OCR junk.
    if len(word) >= 5 and vowels / len(word) < 0.2:
        return True

    # Repeated/near-repeated characters: "taee", "aaaa", "llll".
    if len(word) >= 4 and len(set(word)) <= 2:
        return True

    # Rare-letter heavy fragments are usually noise.
    rare = sum(character in "qzxjvwk" for character in word)
    if len(word) >= 4 and rare / len(word) >= 0.4:
        return True

    # Short random-looking words with awkward consonant clusters.
    if len(word) <= 6 and re.search(r"[bcdfghjklmnpqrstvwxyz]{4,}", word):
        return True

    # Weird mixed casing (e.g. "iRae", "aFeE") is typical OCR junk.
    if (
        len(token) >= 4
        and not token.islower()
        and not token.isupper()
        and not token.istitle()
        and len(token) <= 8
    ):
        return True

    return False


def default_ocr(cell_image: Image.Image) -> str:
    """
    Standalone OCR implementation using Tesseract.

    Replace only this function with your existing OCR implementation.
    The replacement must:
      1. Accept a PIL.Image.Image.
      2. Return the recognized text as a string.
    """
    if cell_image.width < 2 or cell_image.height < 2:
        return ""

    # Blank / near-blank cells: Tesseract invents random strings from noise.
    if not cell_has_visible_ink(cell_image):
        return ""

    enlarged = cell_image.resize(
        (cell_image.width * 2, cell_image.height * 2),
        Image.Resampling.LANCZOS,
    ).convert("L")
    contrasted = ImageOps.autocontrast(enlarged)
    thresholded = contrasted.point(
        lambda pixel: 255 if pixel > 180 else 0,
        mode="1",
    ).convert("L")

    def normalize_text(text: str) -> str:
        normalized_lines = [
            " ".join(line.split())
            for line in text.splitlines()
            if line.strip()
        ]
        return "\n".join(normalized_lines).strip()

    def score_text(text: str) -> tuple[int, int, int]:
        cleaned = cleanup_extracted_cell_text(text)
        if not cleaned:
            return (-1, -1, -1)

        # Prefer meaningful content, but still accept valid short cleaned values.
        meaningful_bonus = 10 if has_meaningful_cell_content(cleaned) else 1
        alnum_count = sum(character.isalnum() for character in cleaned)
        length_score = len(cleaned.replace(" ", ""))
        return (meaningful_bonus, alnum_count, length_score)

    best_text = ""
    best_score = (-1, -1, -1)

    for image_variant in (contrasted, thresholded):
        # Avoid PSM 11 (sparse text): it hallucinates on blank cells.
        for psm in (6, 7):
            config = f"--oem 3 --psm {psm}"
            text, mean_confidence = ocr_with_confidence(
                image_variant,
                config=config,
            )
            # Soft threshold so faint but real digits/text are kept.
            if text and mean_confidence < 35.0:
                continue

            normalized = normalize_text(text)
            score = score_text(normalized)
            if score > best_score:
                best_score = score
                best_text = normalized

            # Fallback when confidence path is empty/too strict.
            if score <= (-1, -1, -1):
                fallback = normalize_text(
                    pytesseract.image_to_string(
                        image_variant,
                        config=config,
                    )
                )
                fallback_score = score_text(fallback)
                if fallback_score > best_score:
                    best_score = fallback_score
                    best_text = fallback

    return best_text


def is_probable_ocr_noise(line: str) -> bool:
    """Identify OCR fragments without removing normal cell values."""
    tokens = re.findall(r"[A-Za-z0-9]+", line.lower())
    if not tokens:
        # Symbols-only cells such as "{" are not real table values.
        return bool(line.strip())

    if all(token.isdigit() for token in tokens):
        return False

    # Keep numeric-looking values that include separators/units crumbs.
    if re.fullmatch(r"[\d\s,./:%()\-A-Za-z]+", line) and re.search(r"\d", line):
        alpha_tokens = [token for token in tokens if not token.isdigit()]
        if not alpha_tokens or all(len(token) <= 3 for token in alpha_tokens):
            return False

    # Keep common short, valid forms such as "Mr. A. B.".
    if tokens[0] in {"mr", "mrs", "ms", "dr"}:
        return False

    original_tokens = re.findall(r"[A-Za-z0-9]+", line)
    alpha_original = [token for token in original_tokens if token.isalpha()]
    if (
        alpha_original
        and all(is_gibberish_token(token) for token in alpha_original)
        and not any(token.isdigit() for token in tokens)
    ):
        return True

    # Single short alphabetic token from a phantom/narrow column (e.g. ira, ode).
    if len(tokens) == 1:
        token = tokens[0]
        if token.isdigit():
            return False
        if len(token) <= 2:
            return True
        return is_gibberish_token(original_tokens[0] if original_tokens else token)

    if len(tokens) < 3:
        # Two short leftovers such as "fe i" / "ae fe".
        if all(len(token) <= 2 for token in tokens):
            return True
        return all(
            (not token.isdigit()) and is_gibberish_token(original)
            for token, original in zip(tokens, original_tokens)
        )

    lengths = [len(token) for token in tokens]
    if max(lengths) <= 2:
        return True

    # Examples: "i. tae ee" and similar short, disconnected fragments.
    protected_words = {"and", "the", "for", "not", "yes", "nil"}
    if alpha_original and sum(
        is_gibberish_token(token) for token in alpha_original
    ) >= max(1, len(alpha_original) - 1):
        return True

    return (
        max(lengths) <= 3
        and sum(length <= 2 for length in lengths) >= 2
        and not any(token in protected_words for token in tokens)
    )


def cleanup_extracted_cell_text(value: str | None) -> str:
    """Remove OCR-only symbols and clearly meaningless text lines."""
    text = str(value or "")
    # Vertical grid lines are OCR artefacts, not table content.
    text = re.sub(r"[|¦\u2502\u2503\u2551\u254E\u254F]+", " ", text)

    kept_lines = []
    for line in text.splitlines():
        normalized = " ".join(line.split())
        if normalized and not is_probable_ocr_noise(normalized):
            kept_lines.append(normalized)

    return "\n".join(kept_lines).strip()


def has_meaningful_cell_content(value: str | None) -> bool:
    """Return whether a cleaned cell contains real table content."""
    # Use lightweight cleanup here to avoid recursion with cleanup_extracted_cell_text.
    text = str(value or "")
    text = re.sub(r"[|¦\u2502\u2503\u2551\u254E\u254F]+", " ", text)
    text = "\n".join(
        " ".join(line.split())
        for line in text.splitlines()
        if line.strip() and not is_probable_ocr_noise(" ".join(line.split()))
    ).strip()
    if not text:
        return False

    words = re.findall(r"[A-Za-z]+", text)
    real_words = [
        word
        for word in words
        if len(word) >= 3 and not is_gibberish_token(word)
    ]
    # Prefer real words; ignore hallucinated OCR crumbs.
    if real_words:
        return True

    # Allow uppercase codes such as INR / USA / HUF.
    if any(len(word) == 3 and word.isupper() for word in words):
        return True

    protected = {
        "and", "the", "for", "not", "yes", "nil", "ltd", "pvt", "huf",
        "mr", "mrs", "ms", "dr", "no", "sr", "qty", "nos", "amt", "ref",
    }
    if any(word.lower() in protected for word in words):
        return True

    # Do not discard a valid date, amount, percentage, or other numeric field
    # merely because its column header was missed by OCR.
    return bool(
        re.search(r"\d", text)
        and re.fullmatch(r"[\d\s,./:%()\-A-Za-z]+", text)
    )


def _column_values(data: list[list[str]], column_index: int) -> list[str]:
    return [
        row[column_index] if column_index < len(row) else ""
        for row in data
    ]


def is_junk_generated_column(values: list[str]) -> bool:
    """
    Detect phantom columns created from grid-line OCR crumbs.

    Typical pattern: empty header + short fragments (ira / fe / mit / ode).
    """
    if not values:
        return True

    header = cleanup_extracted_cell_text(values[0])
    body = values[1:] if len(values) > 1 else []

    meaningful_body = [
        cell for cell in body if has_meaningful_cell_content(cell)
    ]
    if meaningful_body:
        return False

    # No meaningful body cells. Drop when header is also empty/noise, or when
    # the body only had short fragments / empties.
    if not header:
        return True

    # Header exists but body is empty/noise — still drop narrow phantom cols
    # whose "header" is itself a short OCR crumb.
    return is_probable_ocr_noise(header) or len(header) <= 3


def prune_generated_columns(
    data: list[list[str]],
    merge_ranges: list[tuple[int, int, int, int]],
    ) -> tuple[list[list[str]], list[tuple[int, int, int, int]]]:
    """Drop non-index columns containing only empty or OCR-noise values."""
    if not data or not data[0]:
        return data, merge_ranges

    column_count = len(data[0])
    if column_count <= 1:
        return data, merge_ranges

    # The first column is normally the serial/index column and is always kept.
    keep_indices = [0]
    for column_index in range(1, column_count):
        values = _column_values(data, column_index)
        if is_junk_generated_column(values):
            continue
        if any(has_meaningful_cell_content(cell) for cell in values):
            keep_indices.append(column_index)

    if len(keep_indices) == column_count:
        return data, merge_ranges

    index_map = {
        old_index: new_index
        for new_index, old_index in enumerate(keep_indices)
    }
    new_data = [
        [row[index] if index < len(row) else "" for index in keep_indices]
        for row in data
    ]

    new_merges: list[tuple[int, int, int, int]] = []
    for row_start, row_end, column_start, column_end in merge_ranges:
        mapped_columns = [
            index_map[index]
            for index in range(column_start, column_end + 1)
            if index in index_map
        ]
        if not mapped_columns:
            continue

        # Preserve text from a merged-cell anchor if its original column was
        # removed and the merge still has a retained column.
        if (
            column_start not in index_map
            and row_start < len(data)
            and column_start < len(data[row_start])
        ):
            new_data[row_start][mapped_columns[0]] = data[row_start][column_start]

        new_merges.append(
            (
                row_start,
                row_end,
                min(mapped_columns),
                max(mapped_columns),
            )
        )

    print(
        f"  Removed {column_count - len(keep_indices)} empty/noise column(s)."
    )
    return new_data, new_merges


def rows_look_like_duplicates(
    first_row: list[str],
    second_row: list[str],
) -> bool:
    comparable_pairs = 0
    matching_pairs = 0
    meaningful_first = 0
    meaningful_second = 0

    for first_cell, second_cell in zip(first_row, second_row, strict=False):
        first_text = cleanup_extracted_cell_text(first_cell)
        second_text = cleanup_extracted_cell_text(second_cell)

        first_meaningful = has_meaningful_cell_content(first_text)
        second_meaningful = has_meaningful_cell_content(second_text)
        meaningful_first += int(first_meaningful)
        meaningful_second += int(second_meaningful)

        if not first_meaningful and not second_meaningful:
            continue

        comparable_pairs += 1
        if first_text == second_text:
            matching_pairs += 1
            continue

        if first_text and second_text and (
            first_text in second_text or second_text in first_text
        ):
            matching_pairs += 1

    minimum_meaningful = min(meaningful_first, meaningful_second)
    if minimum_meaningful < 2 or comparable_pairs == 0:
        return False

    coverage = matching_pairs / comparable_pairs
    return coverage >= 0.75


def merge_similar_rows(
    first_row: list[str],
    second_row: list[str],
) -> list[str]:
    merged: list[str] = []
    column_count = max(len(first_row), len(second_row))

    for index in range(column_count):
        first_cell = first_row[index] if index < len(first_row) else ""
        second_cell = second_row[index] if index < len(second_row) else ""
        first_text = cleanup_extracted_cell_text(first_cell)
        second_text = cleanup_extracted_cell_text(second_cell)

        if not first_text:
            merged.append(second_text)
            continue
        if not second_text:
            merged.append(first_text)
            continue

        first_meaningful = has_meaningful_cell_content(first_text)
        second_meaningful = has_meaningful_cell_content(second_text)

        if first_meaningful and not second_meaningful:
            merged.append(first_text)
        elif second_meaningful and not first_meaningful:
            merged.append(second_text)
        elif len(second_text) > len(first_text) and first_text in second_text:
            merged.append(second_text)
        else:
            merged.append(first_text)

    return merged


def deduplicate_adjacent_rows(
    data: list[list[str]],
    merge_ranges: list[tuple[int, int, int, int]],
) -> tuple[list[list[str]], list[tuple[int, int, int, int]]]:
    if len(data) < 2:
        return data, merge_ranges

    new_data: list[list[str]] = []
    index_map: dict[int, int] = {}
    source_index = 0

    while source_index < len(data):
        current_row = data[source_index]

        if (
            source_index + 1 < len(data)
            and rows_look_like_duplicates(current_row, data[source_index + 1])
        ):
            current_row = merge_similar_rows(current_row, data[source_index + 1])
            index_map[source_index] = len(new_data)
            index_map[source_index + 1] = len(new_data)
            new_data.append(current_row)
            source_index += 2
            continue

        index_map[source_index] = len(new_data)
        new_data.append(current_row)
        source_index += 1

    if len(new_data) == len(data):
        return data, merge_ranges

    new_merges: list[tuple[int, int, int, int]] = []
    for row_start, row_end, column_start, column_end in merge_ranges:
        mapped_rows = [
            index_map[index]
            for index in range(row_start, row_end + 1)
            if index in index_map
        ]
        if not mapped_rows:
            continue

        new_merges.append(
            (
                min(mapped_rows),
                max(mapped_rows),
                column_start,
                column_end,
            )
        )

    print(f"  Collapsed {len(data) - len(new_data)} duplicate/noisy row(s).")
    return new_data, new_merges


# Example adapter for your existing OCR:
#
# def my_existing_ocr(cell_image: Image.Image) -> str:
#     result = your_ocr_model.recognize(cell_image)
#     return result["text"].strip()
#
# Then change the last function call to:
#
# extract_tables_from_pdf(
#     INPUT_PDF,
#     OUTPUT_XLSX,
#     ocr_function=my_existing_ocr,
# )


# =============================================================================
# TABLE STRUCTURE TO GRID
# =============================================================================

def prepare_rows_columns_and_spans(
    structure: list[Detection],
    ) -> tuple[list[Detection], list[Detection], list[Detection]]:
    rows = [
        item
        for item in structure
        if item.label == "table row"
    ]
    columns = [
        item
        for item in structure
        if item.label == "table column"
    ]
    spans = [
        item
        for item in structure
        if item.label == "table spanning cell"
    ]

    rows = remove_duplicate_detections(rows, iou_threshold=0.85)
    columns = remove_duplicate_detections(columns, iou_threshold=0.85)
    spans = remove_duplicate_detections(spans, iou_threshold=0.85)

    rows = collapse_nearby_structure_detections(rows, axis="row")
    columns = collapse_nearby_structure_detections(columns, axis="column")

    rows.sort(key=lambda item: (item.box[1] + item.box[3]) / 2)
    columns.sort(key=lambda item: (item.box[0] + item.box[2]) / 2)

    if columns:
        widths = [max(1.0, column.box[2] - column.box[0]) for column in columns]
        median_width = median(widths)
        filtered_columns: list[Detection] = []

        for index, column in enumerate(columns):
            width = column.box[2] - column.box[0]
            if index == 0:
                filtered_columns.append(column)
                continue

            suspiciously_narrow = width <= max(5.0, median_width * 0.18)
            if suspiciously_narrow:
                continue
            filtered_columns.append(column)

        columns = filtered_columns or columns

    return rows, columns, spans


def grid_cell_box(
    row: Detection,
    column: Detection,
    image_width: int,
    image_height: int,
    ) -> tuple[int, int, int, int]:
    """
    Build a cell using the intersection of one predicted row and one column.
    """
    x1 = max(0, int(round(column.box[0])))
    y1 = max(0, int(round(row.box[1])))
    x2 = min(image_width, int(round(column.box[2])))
    y2 = min(image_height, int(round(row.box[3])))

    if x2 <= x1 or y2 <= y1:
        return (0, 0, 0, 0)

    return (x1, y1, x2, y2)


def find_span_range(
    span: Detection,
    rows: list[Detection],
    columns: list[Detection],
    table_width: int,
    table_height: int,
    minimum_cell_overlap: float = 0.35,
    ) -> tuple[int, int, int, int] | None:
    """
    Convert a predicted spanning-cell box into zero-based grid coordinates.

    Returns:
        row_start, row_end, column_start, column_end
    """
    matched_rows: set[int] = set()
    matched_columns: set[int] = set()

    for row_index, row in enumerate(rows):
        for column_index, column in enumerate(columns):
            cell_box = grid_cell_box(
                row,
                column,
                table_width,
                table_height,
            )
            area = box_area(cell_box)

            if area <= 0:
                continue

            overlap_ratio = intersection_area(span.box, cell_box) / area

            if overlap_ratio >= minimum_cell_overlap:
                matched_rows.add(row_index)
                matched_columns.add(column_index)

    if not matched_rows or not matched_columns:
        return None

    row_start = min(matched_rows)
    row_end = max(matched_rows)
    column_start = min(matched_columns)
    column_end = max(matched_columns)

    if row_start == row_end and column_start == column_end:
        return None

    return row_start, row_end, column_start, column_end


def ranges_overlap(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
    ) -> bool:
    first_row_start, first_row_end, first_col_start, first_col_end = first
    second_row_start, second_row_end, second_col_start, second_col_end = second

    rows_overlap = not (
        first_row_end < second_row_start
        or second_row_end < first_row_start
    )
    columns_overlap = not (
        first_col_end < second_col_start
        or second_col_end < first_col_start
    )

    return rows_overlap and columns_overlap


def extract_grid(
    table_image: Image.Image,
    rows: list[Detection],
    columns: list[Detection],
    spans: list[Detection],
    ocr_function: Callable[[Image.Image], str],
    ) -> tuple[
    list[list[str]],
    list[tuple[int, int, int, int]],
    ]:
    """
    OCR every ordinary cell, then replace predicted spanning ranges with one
    OCR result for the complete merged region.
    """
    data = [
        ["" for _ in columns]
        for _ in rows
    ]

    for row_index, row in enumerate(rows):
        for column_index, column in enumerate(columns):
            x1, y1, x2, y2 = grid_cell_box(
                row,
                column,
                table_image.width,
                table_image.height,
            )

            if x2 <= x1 or y2 <= y1:
                continue

            x1 = min(x2, x1 + CELL_INSET)
            y1 = min(y2, y1 + CELL_INSET)
            x2 = max(x1, x2 - CELL_INSET)
            y2 = max(y1, y2 - CELL_INSET)

            cell_image = table_image.crop((x1, y1, x2, y2))
            cell_text = ocr_function(cell_image)

            if not cleanup_extracted_cell_text(cell_text):
                # Retry 1: slightly expanded crop (helps clipped characters).
                retry_box = clamp_box(
                    (
                        x1 - CELL_INSET,
                        y1 - CELL_INSET,
                        x2 + CELL_INSET,
                        y2 + CELL_INSET,
                    ),
                    table_image.width,
                    table_image.height,
                )
                retry_image = table_image.crop(retry_box)
                retry_text = ocr_function(retry_image)
                if cleanup_extracted_cell_text(retry_text):
                    cell_text = retry_text
                else:
                    # Retry 2: original intersection without inset, for tiny values.
                    raw_box = grid_cell_box(
                        row,
                        column,
                        table_image.width,
                        table_image.height,
                    )
                    if raw_box[2] > raw_box[0] and raw_box[3] > raw_box[1]:
                        raw_image = table_image.crop(raw_box)
                        raw_text = ocr_function(raw_image)
                        if cleanup_extracted_cell_text(raw_text):
                            cell_text = raw_text

            data[row_index][column_index] = cell_text

    merge_ranges: list[tuple[int, int, int, int]] = []

    for span in sorted(
        spans,
        key=lambda item: item.score,
        reverse=True,
    ):
        merge_range = find_span_range(
            span,
            rows,
            columns,
            table_image.width,
            table_image.height,
        )

        if merge_range is None:
            continue

        if any(
            ranges_overlap(merge_range, existing)
            for existing in merge_ranges
        ):
            continue

        row_start, row_end, column_start, column_end = merge_range

        x1 = min(
            columns[index].box[0]
            for index in range(column_start, column_end + 1)
        )
        x2 = max(
            columns[index].box[2]
            for index in range(column_start, column_end + 1)
        )
        y1 = min(
            rows[index].box[1]
            for index in range(row_start, row_end + 1)
        )
        y2 = max(
            rows[index].box[3]
            for index in range(row_start, row_end + 1)
        )

        merged_box = clamp_box(
            (x1, y1, x2, y2),
            table_image.width,
            table_image.height,
        )
        merged_image = table_image.crop(merged_box)
        merged_text = ocr_function(merged_image)

        data[row_start][column_start] = merged_text

        for row_index in range(row_start, row_end + 1):
            for column_index in range(column_start, column_end + 1):
                if (
                    row_index != row_start
                    or column_index != column_start
                ):
                    data[row_index][column_index] = ""

        merge_ranges.append(merge_range)

    return data, merge_ranges


# =============================================================================
# DEBUG IMAGES
# =============================================================================

def save_structure_debug_image(
    table_image: Image.Image,
    rows: list[Detection],
    columns: list[Detection],
    spans: list[Detection],
    output_path: Path,
    ) -> None:
    """
    Red = rows, blue = columns, green = spanning cells.
    """
    debug_image = table_image.copy()
    draw = ImageDraw.Draw(debug_image)

    for row in rows:
        draw.rectangle(row.box, outline="red", width=2)

    for column in columns:
        draw.rectangle(column.box, outline="blue", width=2)

    for span in spans:
        draw.rectangle(span.box, outline="green", width=3)

    debug_image.save(output_path)


# =============================================================================
# EXCEL OUTPUT
# =============================================================================

def write_table_sheet(
    workbook: Workbook,
    sheet_name: str,
    data: list[list[str]],
    merge_ranges: list[tuple[int, int, int, int]],
     ) -> None:
    worksheet = workbook.create_sheet(
        title=safe_sheet_name(sheet_name)
    )
    worksheet.freeze_panes = "A2"

    for row_number, values in enumerate(data, start=1):
        for column_number, value in enumerate(values, start=1):
            cell = worksheet.cell(
                row=row_number,
                column=column_number,
                value=value,
            )
            cell.alignment = Alignment(
                wrap_text=True,
                vertical="top",
            )

    # Treat the first detected row as a probable header.
    if data:
        for cell in worksheet[1]:
            cell.font = Font(bold=True)
            cell.fill = PatternFill(
                fill_type="solid",
                fgColor="D9EAF7",
            )

    # Determine widths before applying Excel merges.
    for column_number in range(1, len(data[0]) + 1 if data else 1):
        maximum_length = 0

        for row_number in range(1, len(data) + 1):
            value = worksheet.cell(
                row=row_number,
                column=column_number,
            ).value

            if value is None:
                continue

            maximum_length = max(
                maximum_length,
                max(
                    (
                        len(line)
                        for line in str(value).splitlines()
                    ),
                    default=0,
                ),
            )

        worksheet.column_dimensions[
            get_column_letter(column_number)
        ].width = min(max(maximum_length + 2, 10), 45)

    for row_start, row_end, column_start, column_end in merge_ranges:
        worksheet.merge_cells(
            start_row=row_start + 1,
            end_row=row_end + 1,
            start_column=column_start + 1,
            end_column=column_end + 1,
        )


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def extract_tables_from_pdf(
    input_pdf: Path,
    output_xlsx: Path,
    ocr_function: Callable[[Image.Image], str] = default_ocr,
    ) -> None:
    if not input_pdf.exists():
        raise FileNotFoundError(
            f"Input PDF was not found: {input_pdf.resolve()}"
        )

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    engine = TableTransformerEngine()
    pages = render_pdf_pages(input_pdf, PDF_DPI)

    workbook = Workbook()
    workbook.remove(workbook.active)

    extracted_table_count = 0

    for page_number, page_image in enumerate(pages, start=1):
        tables = engine.detect_tables(page_image)

        print(
            f"Page {page_number}: "
            f"{len(tables)} table region(s) detected."
        )

        for table_number, table in enumerate(tables, start=1):
            crop_box = clamp_box(
                table.box,
                page_image.width,
                page_image.height,
                padding=TABLE_PADDING,
            )
            table_image = page_image.crop(crop_box)

            if table.label == "table rotated":
                table_image = table_image.rotate(
                    90,
                    expand=True,
                )

            table_crop_path = (
                DEBUG_DIR
                / f"page_{page_number}_table_{table_number}.png"
            )
            DEBUG_DIR.mkdir(parents=True, exist_ok=True)
            table_image.save(table_crop_path)

            structure = engine.recognize_structure(table_image)
            rows, columns, spans = prepare_rows_columns_and_spans(
                structure
            )

            structure_debug_path = DEBUG_DIR / (
                f"page_{page_number}_table_{table_number}"
                "_structure.png"
            )
            DEBUG_DIR.mkdir(parents=True, exist_ok=True)
            save_structure_debug_image(
                table_image,
                rows,
                columns,
                spans,
                structure_debug_path,
            )

            if not rows or not columns:
                print(
                    f"  Skipped table {table_number}: "
                    "no usable row/column structure."
                )
                continue

            data, merge_ranges = extract_grid(
                table_image,
                rows,
                columns,
                spans,
                ocr_function,
            )
            data = [
                [cleanup_extracted_cell_text(value) for value in row]
                for row in data
            ]
            data, merge_ranges = prune_generated_columns(data, merge_ranges)
            data, merge_ranges = deduplicate_adjacent_rows(data, merge_ranges)

            write_table_sheet(
                workbook,
                sheet_name=f"P{page_number}_Table{table_number}",
                data=data,
                merge_ranges=merge_ranges,
            )

            extracted_table_count += 1
            print(
                f"  Extracted table {table_number}: "
                f"{len(rows)} row(s), {len(data[0]) if data else 0} column(s)."
            )

    if extracted_table_count == 0:
        worksheet = workbook.create_sheet("No tables found")
        worksheet["A1"] = (
            "No table with usable row and column structure was detected."
        )

    workbook.save(output_xlsx)

    print()
    print(f"Tables extracted: {extracted_table_count}")
    print(f"Excel file: {output_xlsx.resolve()}")
    print(f"Debug images: {DEBUG_DIR.resolve()}")


if __name__ == "__main__":
    extract_tables_from_pdf(
        input_pdf=INPUT_PDF,
        output_xlsx=OUTPUT_XLSX,
        ocr_function=default_ocr,
    )
