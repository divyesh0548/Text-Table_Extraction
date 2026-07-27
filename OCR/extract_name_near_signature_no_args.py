#!/usr/bin/env python3
"""
Detect signatures in a PDF and extract the probable printed name nearby.

Pipeline:
1. Render each PDF page as an image.
2. Detect signature bounding boxes with a custom Ultralytics YOLO model.
3. Run Tesseract OCR on the page, and also OCR a band under each signature.
4. Prefer text printed below the signature as the person's name.
5. Strip role phrases (director, president, authorised signatory, etc.).
6. Rank the remaining nearby lines and return the most probable name.
7. Save JSON, CSV, annotated page images, and an annotated PDF.

Important:
- The YOLO model must be trained to detect a class named "signature",
  unless --signature-class-id is supplied.
- This extracts printed text near a signature. It does not identify a person
  from the signature strokes themselves.

Install Python dependencies:
    pip install ultralytics pymupdf pytesseract opencv-python numpy

Tesseract itself must also be installed separately:
    Windows: install Tesseract and set TESSERACT_CMD below.
    Ubuntu:  sudo apt install tesseract-ocr

Edit the CONFIGURATION section below, then run:
    python extract_name_near_signature_no_args.py
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import fitz
import numpy as np
import pytesseract
from pytesseract import Output
from ultralytics import YOLO


DEFAULT_EXCLUDED_PHRASES = (
    "director",
    "managing director",
    "whole time director",
    "whole-time director",
    "executive director",
    "independent director",
    "additional director",
    "nominee director",
    "president",
    "vice president",
    "vice-president",
    "chairman",
    "chairperson",
    "secretary",
    "company secretary",
    "authorised signatory",
    "authorized signatory",
    "authorised signatories",
    "authorized signatories",
    "signatory",
    "partner",
    "proprietor",
)


# =============================================================================
# CONFIGURATION: EDIT THESE VALUES
# =============================================================================

# Input PDF that contains signatures.
INPUT_PDF = Path("Current-test/MBP 1 AKB 2026.pdf")

# Trained Ultralytics YOLO model that detects signatures.
SIGNATURE_MODEL = Path("Sign-Detection-YOLO8s.pt")

# Folder in which JSON, CSV, annotated images, and annotated PDF will be saved.
OUTPUT_DIR = Path("signature_results")

# Windows example:
# TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
#
# Use None on Linux or when tesseract is already available in PATH.
TESSERACT_CMD: str | None = (
    r"C:\Users\Divyesh Parmar\Downloads\Tesserect OCR\tesseract.exe"
)

# PDF rendering resolution.
DPI = 300

# YOLO settings.
SIGNATURE_CONFIDENCE = 0.35
YOLO_IMAGE_SIZE = 1280

# Set this to the numeric YOLO signature class ID, such as 0.
# Keep it as None when the YOLO class name contains the word "signature".
SIGNATURE_CLASS_ID: int | None = None

# Tesseract OCR settings.
OCR_LANGUAGE = "eng"
OCR_PSM = 6
MINIMUM_OCR_CONFIDENCE = 25.0

# Role / designation words to strip so only the person's name remains.
EXCLUDED_PHRASES = (
    "director",
    "managing director",
    "whole time director",
    "whole-time director",
    "executive director",
    "independent director",
    "additional director",
    "nominee director",
    "president",
    "vice president",
    "vice-president",
    "chairman",
    "chairperson",
    "secretary",
    "company secretary",
    "authorised signatory",
    "authorized signatory",
    "authorised signatories",
    "authorized signatories",
    "signatory",
    "partner",
    "proprietor",
)

# Search band BELOW the signature (name is usually printed under the ink).
# Distances are in rendered image pixels at DPI above.
SEARCH_DOWN = 420
SEARCH_UP = 40
SEARCH_SIDE = 80
# Extra padding around the below-signature crop used for a second OCR pass.
BELOW_OCR_PAD_X = 40
BELOW_OCR_PAD_TOP = 8
BELOW_OCR_HEIGHT = 220

# Candidate must reach this score to be selected as the probable name.
MINIMUM_NAME_SCORE = 0.35

# =============================================================================
# END OF CONFIGURATION
# =============================================================================


@dataclass
class TextLine:
    text: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2


@dataclass
class SignatureResult:
    page: int
    signature_number: int
    signature_confidence: float
    signature_box: dict[str, int]
    probable_name: str | None
    name_confidence: float
    name_score: float
    name_region: str | None
    name_box: dict[str, int] | None
    nearby_candidates: list[dict]


def normalize_spaces(text: str) -> str:
    return " ".join(text.split()).strip()


def normalize_for_matching(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return normalize_spaces(text)


def remove_excluded_phrases(
    text: str,
    excluded_phrases: Iterable[str],
    ) -> str:
    """
    Remove role phrases without discarding a name on the same OCR line.

    Examples:
        "Rajesh Kumar Director" -> "Rajesh Kumar"
        "Authorized Signatory: Rajesh Kumar" -> "Rajesh Kumar"
    """
    cleaned = normalize_spaces(text)

    for phrase in sorted(excluded_phrases, key=len, reverse=True):
        words = [re.escape(word) for word in normalize_for_matching(phrase).split()]
        if not words:
            continue

        # Allows OCR punctuation or multiple spaces between words.
        pattern = r"\b" + r"[\W_]+".join(words) + r"\b"
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(
        r"^[\s:;,\-–—|./\\]+|[\s:;,\-–—|./\\]+$",
        "",
        cleaned,
    )
    return normalize_spaces(cleaned)


def looks_like_date_or_identifier(text: str) -> bool:
    normalized = normalize_spaces(text)

    date_patterns = (
        r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b",
        r"\b\d{1,2}\s+"
        r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)"
        r"[a-z]*\s+\d{2,4}\b",
    )

    if any(re.search(pattern, normalized, re.IGNORECASE) for pattern in date_patterns):
        return True

    # DIN, employee number, registration number, phone number, etc.
    if re.search(r"\b(?:din|id|no|number|date|place)\s*[:\-]?\s*\d+", normalized, re.I):
        return True

    digits = sum(character.isdigit() for character in normalized)
    letters = sum(character.isalpha() for character in normalized)

    return digits >= 4 and digits > letters


def name_likeness(text: str) -> float:
    """
    Heuristic score from 0 to 1 indicating whether text resembles a person's name.
    """
    text = normalize_spaces(text)

    if not text or looks_like_date_or_identifier(text):
        return 0.0

    words = re.findall(r"[A-Za-z][A-Za-z.'’-]*", text)
    if not words:
        return 0.0

    compact = re.sub(r"\s+", "", text)
    letters = sum(character.isalpha() for character in compact)
    alphabetic_ratio = letters / max(len(compact), 1)

    if alphabetic_ratio < 0.65:
        return 0.0

    score = 0.25 + 0.35 * alphabetic_ratio

    # Most printed personal names have between 2 and 5 components.
    if 2 <= len(words) <= 5:
        score += 0.25
    elif len(words) == 1:
        score += 0.08
    elif len(words) > 7:
        score -= 0.30

    title_case_count = sum(
        word[0].isupper() and (len(word) == 1 or word[1:].islower())
        for word in words
    )
    uppercase_count = sum(word.isupper() and len(word) > 1 for word in words)

    if title_case_count / len(words) >= 0.60:
        score += 0.15
    elif uppercase_count / len(words) >= 0.60:
        # Many scanned forms print names in uppercase.
        score += 0.10

    # Sentences and role descriptions are usually longer.
    if len(text) > 70:
        score -= 0.25

    common_non_name_words = {
        "signed",
        "signature",
        "signatory",
        "company",
        "firm",
        "partner",
        "chairman",
        "chairperson",
        "secretary",
        "member",
        "director",
        "president",
        "managing",
        "executive",
        "independent",
        "authorised",
        "authorized",
        "date",
        "place",
        "name",
        "for",
        "and",
        "the",
        "of",
    }
    normalized_words = {word.casefold().strip(".'’-") for word in words}

    if normalized_words & common_non_name_words:
        score -= 0.20

    return max(0.0, min(1.0, score))


def line_box(line: TextLine) -> dict[str, int]:
    return {
        "x1": line.x1,
        "y1": line.y1,
        "x2": line.x2,
        "y2": line.y2,
    }


def intersection_area(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
     ) -> int:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    width = max(0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0, min(ay2, by2) - max(ay1, by1))
    return width * height


def horizontal_overlap_ratio(
    line: TextLine,
    signature_box: tuple[int, int, int, int],
     ) -> float:
    sx1, _, sx2, _ = signature_box
    overlap = max(0, min(line.x2, sx2) - max(line.x1, sx1))
    return overlap / max(1, min(line.width, sx2 - sx1))


def vertical_overlap_ratio(
    line: TextLine,
    signature_box: tuple[int, int, int, int],
     ) -> float:
    _, sy1, _, sy2 = signature_box
    overlap = max(0, min(line.y2, sy2) - max(line.y1, sy1))
    return overlap / max(1, min(line.height, sy2 - sy1))


def classify_relative_region(
    line: TextLine,
    signature_box: tuple[int, int, int, int],
    search_down: int,
    search_up: int,
    search_side: int,
    ) -> tuple[str, float] | None:
    """
    Return the candidate region and spatial closeness score.

    Printed names are almost always under the signature, so "below" is
    preferred strongly over left / right / above.
    """
    sx1, sy1, sx2, sy2 = signature_box
    signature_center_x = (sx1 + sx2) / 2
    signature_center_y = (sy1 + sy2) / 2
    signature_width = max(1, sx2 - sx1)

    h_overlap = horizontal_overlap_ratio(line, signature_box)
    v_overlap = vertical_overlap_ratio(line, signature_box)

    candidates: list[tuple[str, float]] = []

    # Text below the signature (primary region for the person's name).
    below_gap = line.y1 - sy2
    horizontally_near = (
        line.center_x >= sx1 - search_side - signature_width * 0.25
        and line.center_x <= sx2 + search_side + signature_width * 0.25
    )
    if -25 <= below_gap <= search_down and horizontally_near:
        alignment = max(
            0.0,
            1.0 - abs(line.center_x - signature_center_x)
            / max(search_side + signature_width * 0.5, 1),
        )
        distance = max(0.0, 1.0 - max(0, below_gap) / max(search_down, 1))
        candidates.append(
            ("below", 0.55 * distance + 0.45 * max(h_overlap, alignment))
        )

    # Text to the right.
    right_gap = line.x1 - sx2
    if (
        -10 <= right_gap <= search_side
        and line.center_y >= sy1 - search_up
        and line.center_y <= sy2 + search_down
    ):
        alignment = max(
            0.0,
            1.0 - abs(line.center_y - signature_center_y)
            / max(search_up + search_down, 1),
        )
        distance = max(0.0, 1.0 - max(0, right_gap) / max(search_side, 1))
        candidates.append(("right", 0.55 * distance + 0.45 * max(v_overlap, alignment)))

    # Text to the left.
    left_gap = sx1 - line.x2
    if (
        -10 <= left_gap <= search_side
        and line.center_y >= sy1 - search_up
        and line.center_y <= sy2 + search_down
    ):
        alignment = max(
            0.0,
            1.0 - abs(line.center_y - signature_center_y)
            / max(search_up + search_down, 1),
        )
        distance = max(0.0, 1.0 - max(0, left_gap) / max(search_side, 1))
        candidates.append(("left", 0.50 * distance + 0.50 * max(v_overlap, alignment)))

    # Text above.
    above_gap = sy1 - line.y2
    if (
        -10 <= above_gap <= search_up
        and line.center_x >= sx1 - search_side
        and line.center_x <= sx2 + search_side
    ):
        alignment = max(
            0.0,
            1.0 - abs(line.center_x - signature_center_x) / max(search_side, 1),
        )
        distance = max(0.0, 1.0 - max(0, above_gap) / max(search_up, 1))
        candidates.append(("above", 0.55 * distance + 0.45 * max(h_overlap, alignment)))

    if not candidates:
        return None

    region_priority = {
        "below": 0.22,
        "right": 0.04,
        "left": 0.03,
        "above": 0.01,
    }

    region, spatial_score = max(
        candidates,
        key=lambda item: item[1] + region_priority[item[0]],
    )
    return region, min(1.0, spatial_score + region_priority[region])


def extract_ocr_lines(
    image_bgr: np.ndarray,
    language: str,
    minimum_ocr_confidence: float,
    psm: int,
    ) -> list[TextLine]:
    """
    Run Tesseract on the whole page and group OCR words into text lines.
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    data = pytesseract.image_to_data(
        rgb,
        lang=language,
        config=f"--oem 3 --psm {psm}",
        output_type=Output.DICT,
    )

    grouped: dict[tuple[int, int, int], list[dict]] = {}

    count = len(data["text"])

    for index in range(count):
        text = normalize_spaces(str(data["text"][index]))

        try:
            confidence = float(data["conf"][index])
        except (TypeError, ValueError):
            confidence = -1.0

        if not text or confidence < minimum_ocr_confidence:
            continue

        key = (
            int(data["block_num"][index]),
            int(data["par_num"][index]),
            int(data["line_num"][index]),
        )

        grouped.setdefault(key, []).append(
            {
                "text": text,
                "confidence": confidence,
                "x": int(data["left"][index]),
                "y": int(data["top"][index]),
                "width": int(data["width"][index]),
                "height": int(data["height"][index]),
                "word_num": int(data["word_num"][index]),
            }
        )

    lines: list[TextLine] = []

    for words in grouped.values():
        words.sort(key=lambda item: item["word_num"])

        text = normalize_spaces(" ".join(word["text"] for word in words))
        x1 = min(word["x"] for word in words)
        y1 = min(word["y"] for word in words)
        x2 = max(word["x"] + word["width"] for word in words)
        y2 = max(word["y"] + word["height"] for word in words)
        confidence = sum(word["confidence"] for word in words) / len(words)

        lines.append(
            TextLine(
                text=text,
                confidence=confidence,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
        )

    return lines


def line_mostly_inside_signature(
    line: TextLine,
    signature_box: tuple[int, int, int, int],
    overlap_threshold: float = 0.55,
    ) -> bool:
    """True when most of the OCR line sits inside the signature ink box."""
    line_area = max(1, line.width * line.height)
    overlap = intersection_area(
        signature_box,
        (line.x1, line.y1, line.x2, line.y2),
    )
    return (overlap / line_area) >= overlap_threshold


def ocr_band_below_signature(
    image_bgr: np.ndarray,
    signature_box: tuple[int, int, int, int],
    language: str,
    excluded_phrases: tuple[str, ...],
    pad_x: int = BELOW_OCR_PAD_X,
    pad_top: int = BELOW_OCR_PAD_TOP,
    band_height: int = BELOW_OCR_HEIGHT,
    ) -> list[dict]:
    """
    OCR a crop immediately under the signature box.

    This recovers names when page-level OCR lines miss the printed text below
    a tight signature detection.
    """
    sx1, _, sx2, sy2 = signature_box
    height, width = image_bgr.shape[:2]

    x1 = max(0, sx1 - pad_x)
    y1 = max(0, sy2 - pad_top)
    x2 = min(width, sx2 + pad_x)
    y2 = min(height, sy2 + band_height)

    if x2 - x1 < 8 or y2 - y1 < 8:
        return []

    crop = image_bgr[y1:y2, x1:x2]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    # Upscale sparse label text under signatures.
    rgb = cv2.resize(
        rgb,
        (rgb.shape[1] * 2, rgb.shape[0] * 2),
        interpolation=cv2.INTER_CUBIC,
    )

    candidates: list[dict] = []
    for psm in (6, 7, 4):
        text = pytesseract.image_to_string(
            rgb,
            lang=language,
            config=f"--oem 3 --psm {psm}",
        )
        for raw_line in text.splitlines():
            original = normalize_spaces(raw_line)
            if not original:
                continue
            cleaned = remove_excluded_phrases(original, excluded_phrases)
            if not cleaned:
                continue
            likeness = name_likeness(cleaned)
            if likeness < 0.20:
                continue

            # Prefer the first good name-like line in this below band.
            combined = 0.55 * 0.95 + 0.40 * likeness + 0.05
            candidates.append(
                {
                    "original_text": original,
                    "cleaned_text": cleaned,
                    "region": "below",
                    "ocr_confidence": 70.0,
                    "spatial_score": 0.95,
                    "name_likeness": round(likeness, 4),
                    "combined_score": round(combined, 4),
                    "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "source": "below_crop",
                }
            )

    candidates.sort(key=lambda item: item["combined_score"], reverse=True)
    return candidates


def choose_name_near_signature(
    signature_box: tuple[int, int, int, int],
    ocr_lines: list[TextLine],
    excluded_phrases: tuple[str, ...],
    search_down: int,
    search_up: int,
    search_side: int,
    minimum_name_score: float,
    image_bgr: np.ndarray | None = None,
    language: str = "eng",
    maximum_candidates: int = 10,
    ) -> tuple[dict | None, list[dict]]:
    ranked_candidates: list[dict] = []

    for line in ocr_lines:
        # Ignore OCR that is mostly the signature ink itself.
        if line_mostly_inside_signature(line, signature_box):
            continue

        relative = classify_relative_region(
            line=line,
            signature_box=signature_box,
            search_down=search_down,
            search_up=search_up,
            search_side=search_side,
        )

        if relative is None:
            continue

        region, spatial_score = relative
        cleaned_text = remove_excluded_phrases(line.text, excluded_phrases)
        if not cleaned_text:
            continue

        likeness = name_likeness(cleaned_text)
        ocr_score = max(0.0, min(1.0, line.confidence / 100.0))

        # Strongly prefer text found under the signature.
        region_bonus = 0.12 if region == "below" else 0.0
        combined_score = (
            0.48 * spatial_score
            + 0.37 * likeness
            + 0.15 * ocr_score
            + region_bonus
        )

        ranked_candidates.append(
            {
                "original_text": line.text,
                "cleaned_text": cleaned_text,
                "region": region,
                "ocr_confidence": round(line.confidence, 2),
                "spatial_score": round(spatial_score, 4),
                "name_likeness": round(likeness, 4),
                "combined_score": round(min(1.0, combined_score), 4),
                "box": line_box(line),
                "source": "page_ocr",
            }
        )

    # Dedicated OCR of the band under the signature box.
    if image_bgr is not None:
        ranked_candidates.extend(
            ocr_band_below_signature(
                image_bgr=image_bgr,
                signature_box=signature_box,
                language=language,
                excluded_phrases=excluded_phrases,
            )
        )

    ranked_candidates.sort(
        key=lambda candidate: (
            1 if candidate.get("region") == "below" else 0,
            candidate["combined_score"],
        ),
        reverse=True,
    )
    ranked_candidates = ranked_candidates[:maximum_candidates]

    if (
        not ranked_candidates
        or ranked_candidates[0]["combined_score"] < minimum_name_score
    ):
        return None, ranked_candidates

    return ranked_candidates[0], ranked_candidates


def pixmap_to_bgr(pixmap: fitz.Pixmap) -> np.ndarray:
    image = np.frombuffer(pixmap.samples, dtype=np.uint8)
    image = image.reshape(pixmap.height, pixmap.width, pixmap.n)

    if pixmap.n == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if pixmap.n == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

    raise ValueError(f"Unsupported pixmap channel count: {pixmap.n}")


def detect_signature_boxes(
    model: YOLO,
    image_bgr: np.ndarray,
    confidence_threshold: float,
    image_size: int,
    signature_class_id: int | None,
    ) -> list[tuple[tuple[int, int, int, int], float, int]]:
    results = model.predict(
        source=image_bgr,
        conf=confidence_threshold,
        imgsz=image_size,
        verbose=False,
    )

    detections: list[tuple[tuple[int, int, int, int], float, int]] = []

    if not results:
        return detections

    result = results[0]

    if result.boxes is None:
        return detections

    class_names = result.names

    for box in result.boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())

        if signature_class_id is not None:
            is_signature = class_id == signature_class_id
        else:
            class_name = str(class_names.get(class_id, class_id)).casefold()
            is_signature = "signature" in class_name

        if not is_signature:
            continue

        x1, y1, x2, y2 = [
            int(round(value))
            for value in box.xyxy[0].tolist()
        ]

        if x2 <= x1 or y2 <= y1:
            continue

        detections.append(((x1, y1, x2, y2), confidence, class_id))

    # Read in normal document order.
    detections.sort(key=lambda item: (item[0][1], item[0][0]))
    return detections


def annotate_result(
    page_image: np.ndarray,
    signature_box: tuple[int, int, int, int],
    signature_number: int,
    signature_confidence: float,
    best_candidate: dict | None,
       ) -> None:
    sx1, sy1, sx2, sy2 = signature_box

    cv2.rectangle(
        page_image,
        (sx1, sy1),
        (sx2, sy2),
        (0, 0, 255),
        3,
    )
    cv2.putText(
        page_image,
        f"Signature {signature_number}: {signature_confidence:.2f}",
        (sx1, max(25, sy1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    if best_candidate is None:
        cv2.putText(
            page_image,
            "Nearby name not found",
            (sx1, min(page_image.shape[0] - 10, sy2 + 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return

    name_box = best_candidate["box"]
    nx1, ny1, nx2, ny2 = (
        name_box["x1"],
        name_box["y1"],
        name_box["x2"],
        name_box["y2"],
    )

    cv2.rectangle(
        page_image,
        (nx1, ny1),
        (nx2, ny2),
        (0, 180, 0),
        3,
    )
    cv2.putText(
        page_image,
        best_candidate["cleaned_text"][:80],
        (nx1, max(25, ny1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 180, 0),
        2,
        cv2.LINE_AA,
    )


def save_annotated_pdf(
    image_paths: list[Path],
    page_sizes: list[tuple[float, float]],
    output_pdf: Path,
) -> None:
    document = fitz.open()

    try:
        for image_path, (page_width, page_height) in zip(
            image_paths,
            page_sizes,
            strict=True,
        ):
            page = document.new_page(
                width=page_width,
                height=page_height,
            )
            page.insert_image(
                page.rect,
                filename=str(image_path),
                keep_proportion=False,
            )

        document.save(
            output_pdf,
            garbage=4,
            deflate=True,
        )

    finally:
        document.close()


def save_results_csv(results: list[SignatureResult], output_path: Path) -> None:
    fieldnames = [
        "page",
        "signature_number",
        "signature_confidence",
        "signature_x1",
        "signature_y1",
        "signature_x2",
        "signature_y2",
        "probable_name",
        "name_confidence",
        "name_score",
        "name_region",
        "name_x1",
        "name_y1",
        "name_x2",
        "name_y2",
    ]

    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            signature_box = result.signature_box
            name_box = result.name_box or {}

            writer.writerow(
                {
                    "page": result.page,
                    "signature_number": result.signature_number,
                    "signature_confidence": result.signature_confidence,
                    "signature_x1": signature_box["x1"],
                    "signature_y1": signature_box["y1"],
                    "signature_x2": signature_box["x2"],
                    "signature_y2": signature_box["y2"],
                    "probable_name": result.probable_name or "",
                    "name_confidence": result.name_confidence,
                    "name_score": result.name_score,
                    "name_region": result.name_region or "",
                    "name_x1": name_box.get("x1", ""),
                    "name_y1": name_box.get("y1", ""),
                    "name_x2": name_box.get("x2", ""),
                    "name_y2": name_box.get("y2", ""),
                }
            )


def process_pdf(
    input_pdf: Path,
    model_path: Path,
    output_dir: Path,
    dpi: int,
    signature_confidence: float,
    image_size: int,
    signature_class_id: int | None,
    language: str,
    ocr_psm: int,
    minimum_ocr_confidence: float,
    excluded_phrases: tuple[str, ...],
    search_down: int,
    search_up: int,
    search_side: int,
    minimum_name_score: float,
) -> list[SignatureResult]:
    if not input_pdf.exists():
        raise FileNotFoundError(f"PDF not found: {input_pdf}")

    if not model_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir = output_dir / "annotated_pages"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    source_pdf = fitz.open(input_pdf)

    results: list[SignatureResult] = []
    annotated_page_paths: list[Path] = []
    original_page_sizes: list[tuple[float, float]] = []

    try:
        for page_index in range(source_pdf.page_count):
            page_number = page_index + 1
            page = source_pdf.load_page(page_index)
            original_page_sizes.append((page.rect.width, page.rect.height))

            pixmap = page.get_pixmap(
                dpi=dpi,
                colorspace=fitz.csRGB,
                alpha=False,
            )
            page_image = pixmap_to_bgr(pixmap)

            print(
                f"Page {page_number}/{source_pdf.page_count}: running OCR...",
                flush=True,
            )
            ocr_lines = extract_ocr_lines(
                image_bgr=page_image,
                language=language,
                minimum_ocr_confidence=minimum_ocr_confidence,
                psm=ocr_psm,
            )

            print(
                f"Page {page_number}/{source_pdf.page_count}: detecting signatures...",
                flush=True,
            )
            signatures = detect_signature_boxes(
                model=model,
                image_bgr=page_image,
                confidence_threshold=signature_confidence,
                image_size=image_size,
                signature_class_id=signature_class_id,
            )

            annotated = page_image.copy()

            for signature_index, (
                signature_box,
                detection_confidence,
                _class_id,
            ) in enumerate(signatures, start=1):
                best, candidates = choose_name_near_signature(
                    signature_box=signature_box,
                    ocr_lines=ocr_lines,
                    excluded_phrases=excluded_phrases,
                    search_down=search_down,
                    search_up=search_up,
                    search_side=search_side,
                    minimum_name_score=minimum_name_score,
                    image_bgr=page_image,
                    language=language,
                )

                sx1, sy1, sx2, sy2 = signature_box

                if best is None:
                    result = SignatureResult(
                        page=page_number,
                        signature_number=signature_index,
                        signature_confidence=round(detection_confidence, 4),
                        signature_box={
                            "x1": sx1,
                            "y1": sy1,
                            "x2": sx2,
                            "y2": sy2,
                        },
                        probable_name=None,
                        name_confidence=0.0,
                        name_score=0.0,
                        name_region=None,
                        name_box=None,
                        nearby_candidates=candidates,
                    )
                else:
                    result = SignatureResult(
                        page=page_number,
                        signature_number=signature_index,
                        signature_confidence=round(detection_confidence, 4),
                        signature_box={
                            "x1": sx1,
                            "y1": sy1,
                            "x2": sx2,
                            "y2": sy2,
                        },
                        probable_name=best["cleaned_text"],
                        name_confidence=round(
                            best["ocr_confidence"] / 100.0,
                            4,
                        ),
                        name_score=best["combined_score"],
                        name_region=best["region"],
                        name_box=best["box"],
                        nearby_candidates=candidates,
                    )

                results.append(result)

                annotate_result(
                    page_image=annotated,
                    signature_box=signature_box,
                    signature_number=signature_index,
                    signature_confidence=detection_confidence,
                    best_candidate=best,
                )

            annotated_path = annotated_dir / f"page_{page_number:04d}.png"

            if not cv2.imwrite(str(annotated_path), annotated):
                raise RuntimeError(
                    f"Failed to save annotated page: {annotated_path}"
                )

            annotated_page_paths.append(annotated_path)

            print(
                f"Page {page_number}: {len(signatures)} signature(s) found.",
                flush=True,
            )

    finally:
        source_pdf.close()

    json_path = output_dir / "signature_names.json"
    csv_path = output_dir / "signature_names.csv"
    annotated_pdf_path = output_dir / "annotated_signatures.pdf"

    json_path.write_text(
        json.dumps(
            [asdict(result) for result in results],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    save_results_csv(results, csv_path)

    save_annotated_pdf(
        image_paths=annotated_page_paths,
        page_sizes=original_page_sizes,
        output_pdf=annotated_pdf_path,
    )

    print(f"\nJSON:          {json_path.resolve()}")
    print(f"CSV:           {csv_path.resolve()}")
    print(f"Annotated PDF: {annotated_pdf_path.resolve()}")

    return results


def validate_configuration() -> None:
    """Check configured paths and values before processing."""
    if not INPUT_PDF:
        raise ValueError("INPUT_PDF is not configured.")

    if not SIGNATURE_MODEL:
        raise ValueError("SIGNATURE_MODEL is not configured.")

    if not OUTPUT_DIR:
        raise ValueError("OUTPUT_DIR is not configured.")

    if INPUT_PDF.suffix.lower() != ".pdf":
        raise ValueError(
            f"INPUT_PDF must point to a PDF file: {INPUT_PDF}"
        )

    if DPI < 72:
        raise ValueError("DPI must be at least 72.")

    if not 0.0 <= SIGNATURE_CONFIDENCE <= 1.0:
        raise ValueError(
            "SIGNATURE_CONFIDENCE must be between 0 and 1."
        )

    if not 0.0 <= MINIMUM_NAME_SCORE <= 1.0:
        raise ValueError(
            "MINIMUM_NAME_SCORE must be between 0 and 1."
        )


def main() -> int:
    try:
        validate_configuration()

        if TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

        results = process_pdf(
            input_pdf=INPUT_PDF,
            model_path=SIGNATURE_MODEL,
            output_dir=OUTPUT_DIR,
            dpi=DPI,
            signature_confidence=SIGNATURE_CONFIDENCE,
            image_size=YOLO_IMAGE_SIZE,
            signature_class_id=SIGNATURE_CLASS_ID,
            language=OCR_LANGUAGE,
            ocr_psm=OCR_PSM,
            minimum_ocr_confidence=MINIMUM_OCR_CONFIDENCE,
            excluded_phrases=EXCLUDED_PHRASES,
            search_down=SEARCH_DOWN,
            search_up=SEARCH_UP,
            search_side=SEARCH_SIDE,
            minimum_name_score=MINIMUM_NAME_SCORE,
        )

        print(f"\nCompleted. Total signatures detected: {len(results)}")
        return 0

    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
