#!/usr/bin/env python3
"""
Safely enhance table borders in scanned/image PDFs without blacking out text.

Why this version is safer
-------------------------
The earlier morphology-first approach could join letters in the same text row
and misclassify the entire row as a horizontal line.

This version instead:
1. Detects straight segments with the probabilistic Hough transform.
2. Keeps only near-horizontal and near-vertical segments.
3. Merges duplicate/broken collinear segments.
4. Keeps only groups that form a real grid with at least two horizontal and
   two vertical lines.
5. Draws only those verified table-grid lines on the original rendered page.

Always run this script on the ORIGINAL PDF, not on an already damaged output.
Black bars added by an earlier program cannot be reliably removed afterward.

Install:
    pip install pymupdf opencv-python numpy

For a server/headless machine:
    pip install pymupdf opencv-python-headless numpy

Basic usage:
    1. Set INPUT_PDF (and optional OUTPUT_PDF) in the CONFIGURATION section.
    2. Run: python enhance_pdf_table_lines_safe.py
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import cv2
import fitz  # PyMuPDF
import numpy as np


OutputMode = Literal["color", "grayscale"]
Orientation = Literal["horizontal", "vertical"]


# =============================================================================
# CONFIGURATION — set paths here, then run the script
# =============================================================================

INPUT_PDF = Path("Current-Test/MBP_1_CRA.pdf")
# Leave None to auto-name: <input_stem>_safe_table_lines.pdf
OUTPUT_PDF: Path | None = None

DPI = 300
MODE: OutputMode = "color"  # "color" | "grayscale"
LINE_THICKNESS = 2
MIN_HORIZONTAL_FRACTION = 0.10
MIN_VERTICAL_FRACTION = 0.055
ANGLE_TOLERANCE = 1.5
HOUGH_THRESHOLD = 45
REPAIR_GAP = 10
COORDINATE_TOLERANCE = 4
INTERSECTION_TOLERANCE = 10
MINIMUM_SUPPORT = 0.18
# Optional folder for debug images. None = do not save debug images.
DEBUG_DIR: Path | None = None


@dataclass
class AxisLine:
    """Axis-aligned representation of a detected line segment."""

    orientation: Orientation
    coordinate: float  # y for horizontal, x for vertical
    start: float       # x1 for horizontal, y1 for vertical
    end: float         # x2 for horizontal, y2 for vertical
    weight: float = 1.0

    @property
    def length(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class GridComponent:
    horizontal_indices: set[int]
    vertical_indices: set[int]
    edge_count: int


def pixmap_to_rgb(pixmap: fitz.Pixmap) -> np.ndarray:
    data = np.frombuffer(pixmap.samples, dtype=np.uint8)
    data = data.reshape(pixmap.height, pixmap.width, pixmap.n)

    if pixmap.n == 3:
        return data.copy()
    if pixmap.n == 4:
        return cv2.cvtColor(data, cv2.COLOR_RGBA2RGB)

    raise ValueError(f"Unexpected pixmap channel count: {pixmap.n}")


def encode_png(rgb_image: np.ndarray) -> bytes:
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(
        ".png",
        bgr,
        [cv2.IMWRITE_PNG_COMPRESSION, 3],
    )
    if not ok:
        raise RuntimeError("Could not encode processed page as PNG.")
    return encoded.tobytes()


def prepare_edges(rgb_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create contrast-enhanced grayscale and an edge map for line detection."""
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # Fixed thresholds are intentionally conservative. Adaptive thresholding of
    # the full page was the primary reason text rows became false black lines.
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3, L2gradient=True)

    return enhanced, edges


def angle_from_horizontal_degrees(x1: int, y1: int, x2: int, y2: int) -> float:
    angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
    if angle > 90.0:
        angle = 180.0 - angle
    return angle


def detect_hough_segments(
    edges: np.ndarray,
    horizontal_min_length: int,
    vertical_min_length: int,
    angle_tolerance: float,
    hough_threshold: int,
    repair_gap: int,
) -> tuple[list[AxisLine], list[AxisLine]]:
    """Detect conservative horizontal and vertical straight-line candidates."""
    minimum_hough_length = max(20, min(horizontal_min_length, vertical_min_length))

    raw = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 1800.0,  # 0.1-degree angular resolution
        threshold=max(10, hough_threshold),
        minLineLength=minimum_hough_length,
        maxLineGap=max(0, repair_gap),
    )

    horizontal: list[AxisLine] = []
    vertical: list[AxisLine] = []

    if raw is None:
        return horizontal, vertical

    for entry in raw[:, 0, :]:
        x1, y1, x2, y2 = map(int, entry)
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length <= 0:
            continue

        angle = angle_from_horizontal_degrees(x1, y1, x2, y2)

        if angle <= angle_tolerance and length >= horizontal_min_length:
            horizontal.append(
                AxisLine(
                    orientation="horizontal",
                    coordinate=(y1 + y2) / 2.0,
                    start=float(min(x1, x2)),
                    end=float(max(x1, x2)),
                    weight=length,
                )
            )

        elif abs(90.0 - angle) <= angle_tolerance and length >= vertical_min_length:
            vertical.append(
                AxisLine(
                    orientation="vertical",
                    coordinate=(x1 + x2) / 2.0,
                    start=float(min(y1, y2)),
                    end=float(max(y1, y2)),
                    weight=length,
                )
            )

    return horizontal, vertical


def intervals_touch(a: AxisLine, b: AxisLine, gap: float) -> bool:
    return b.start <= a.end + gap and a.start <= b.end + gap


def merge_axis_lines(
    lines: Sequence[AxisLine],
    coordinate_tolerance: int,
    merge_gap: int,
) -> list[AxisLine]:
    """
    Merge duplicate Hough detections and small breaks on the same straight line.

    Important: this happens AFTER straight-line detection. It never performs a
    close operation across all text on the page.
    """
    if not lines:
        return []

    remaining = sorted(lines, key=lambda item: (item.coordinate, item.start))
    groups: list[list[AxisLine]] = []

    for line in remaining:
        placed = False

        for group in groups:
            total_weight = sum(item.weight for item in group)
            group_coordinate = (
                sum(item.coordinate * item.weight for item in group)
                / max(total_weight, 1e-9)
            )
            group_start = min(item.start for item in group)
            group_end = max(item.end for item in group)
            group_proxy = AxisLine(
                orientation=line.orientation,
                coordinate=group_coordinate,
                start=group_start,
                end=group_end,
                weight=total_weight,
            )

            if (
                abs(line.coordinate - group_coordinate) <= coordinate_tolerance
                and intervals_touch(group_proxy, line, merge_gap)
            ):
                group.append(line)
                placed = True
                break

        if not placed:
            groups.append([line])

    merged: list[AxisLine] = []
    for group in groups:
        total_weight = sum(item.weight for item in group)
        coordinate = (
            sum(item.coordinate * item.weight for item in group)
            / max(total_weight, 1e-9)
        )
        merged.append(
            AxisLine(
                orientation=group[0].orientation,
                coordinate=coordinate,
                start=min(item.start for item in group),
                end=max(item.end for item in group),
                weight=total_weight,
            )
        )

    return merged


def lines_intersect(
    horizontal: AxisLine,
    vertical: AxisLine,
    tolerance: int,
) -> bool:
    return (
        horizontal.start - tolerance
        <= vertical.coordinate
        <= horizontal.end + tolerance
        and vertical.start - tolerance
        <= horizontal.coordinate
        <= vertical.end + tolerance
    )


def build_intersection_graph(
    horizontal: Sequence[AxisLine],
    vertical: Sequence[AxisLine],
    tolerance: int,
) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    h_to_v: dict[int, set[int]] = {index: set() for index in range(len(horizontal))}
    v_to_h: dict[int, set[int]] = {index: set() for index in range(len(vertical))}

    for h_index, h_line in enumerate(horizontal):
        for v_index, v_line in enumerate(vertical):
            if lines_intersect(h_line, v_line, tolerance):
                h_to_v[h_index].add(v_index)
                v_to_h[v_index].add(h_index)

    return h_to_v, v_to_h


def find_grid_components(
    h_to_v: dict[int, set[int]],
    v_to_h: dict[int, set[int]],
) -> list[GridComponent]:
    """Find connected components in the horizontal/vertical intersection graph."""
    components: list[GridComponent] = []
    visited_h: set[int] = set()
    visited_v: set[int] = set()

    for start_h in h_to_v:
        if start_h in visited_h or not h_to_v[start_h]:
            continue

        queue: list[tuple[str, int]] = [("h", start_h)]
        component_h: set[int] = set()
        component_v: set[int] = set()

        while queue:
            kind, index = queue.pop()

            if kind == "h":
                if index in visited_h:
                    continue
                visited_h.add(index)
                component_h.add(index)
                for neighbor in h_to_v[index]:
                    if neighbor not in visited_v:
                        queue.append(("v", neighbor))
            else:
                if index in visited_v:
                    continue
                visited_v.add(index)
                component_v.add(index)
                for neighbor in v_to_h[index]:
                    if neighbor not in visited_h:
                        queue.append(("h", neighbor))

        edge_count = sum(len(h_to_v[index] & component_v) for index in component_h)
        components.append(
            GridComponent(
                horizontal_indices=component_h,
                vertical_indices=component_v,
                edge_count=edge_count,
            )
        )

    return components


def ink_support_ratio(
    gray: np.ndarray,
    line: AxisLine,
    band_radius: int = 2,
) -> float:
    """Measure how much dark-image support exists along a proposed line."""
    height, width = gray.shape

    # Otsu is only used to measure support along an already detected straight
    # segment. It is never used to construct long full-page line masks.
    _, ink = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )

    if line.orientation == "horizontal":
        y = int(round(line.coordinate))
        x1 = max(0, int(math.floor(line.start)))
        x2 = min(width - 1, int(math.ceil(line.end)))
        y1 = max(0, y - band_radius)
        y2 = min(height - 1, y + band_radius)

        if x2 <= x1:
            return 0.0

        band = ink[y1 : y2 + 1, x1 : x2 + 1]
        supported_positions = np.any(band > 0, axis=0)

    else:
        x = int(round(line.coordinate))
        y1 = max(0, int(math.floor(line.start)))
        y2 = min(height - 1, int(math.ceil(line.end)))
        x1 = max(0, x - band_radius)
        x2 = min(width - 1, x + band_radius)

        if y2 <= y1:
            return 0.0

        band = ink[y1 : y2 + 1, x1 : x2 + 1]
        supported_positions = np.any(band > 0, axis=1)

    return float(np.mean(supported_positions)) if supported_positions.size else 0.0


def select_verified_grid_lines(
    gray: np.ndarray,
    horizontal: Sequence[AxisLine],
    vertical: Sequence[AxisLine],
    intersection_tolerance: int,
    minimum_support: float,
    minimum_horizontal_lines: int = 2,
    minimum_vertical_lines: int = 2,
) -> tuple[list[AxisLine], list[AxisLine]]:
    """Retain only line groups that form a real table-like grid."""
    h_to_v, v_to_h = build_intersection_graph(
        horizontal,
        vertical,
        tolerance=intersection_tolerance,
    )
    components = find_grid_components(h_to_v, v_to_h)

    accepted_h_indices: set[int] = set()
    accepted_v_indices: set[int] = set()

    for component in components:
        h_count = len(component.horizontal_indices)
        v_count = len(component.vertical_indices)

        # A real rectangular grid with at least 2 horizontal and 2 vertical
        # lines normally has at least 4 intersections.
        if (
            h_count >= minimum_horizontal_lines
            and v_count >= minimum_vertical_lines
            and component.edge_count >= 4
        ):
            accepted_h_indices.update(component.horizontal_indices)
            accepted_v_indices.update(component.vertical_indices)

    accepted_h = [
        horizontal[index]
        for index in sorted(accepted_h_indices)
        if ink_support_ratio(gray, horizontal[index]) >= minimum_support
    ]
    accepted_v = [
        vertical[index]
        for index in sorted(accepted_v_indices)
        if ink_support_ratio(gray, vertical[index]) >= minimum_support
    ]

    # Re-run the graph after support filtering. This prevents a weak isolated
    # segment from surviving only because it belonged to a larger component.
    h_to_v_2, v_to_h_2 = build_intersection_graph(
        accepted_h,
        accepted_v,
        tolerance=intersection_tolerance,
    )

    final_h = [line for index, line in enumerate(accepted_h) if len(h_to_v_2[index]) >= 2]
    final_v = [line for index, line in enumerate(accepted_v) if len(v_to_h_2[index]) >= 2]

    return final_h, final_v


def draw_verified_lines(
    rgb_image: np.ndarray,
    horizontal: Sequence[AxisLine],
    vertical: Sequence[AxisLine],
    thickness: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw only verified table-grid lines and return image plus line mask."""
    output = rgb_image.copy()
    mask = np.zeros(rgb_image.shape[:2], dtype=np.uint8)
    thickness = max(1, thickness)

    for line in horizontal:
        point1 = (int(round(line.start)), int(round(line.coordinate)))
        point2 = (int(round(line.end)), int(round(line.coordinate)))
        cv2.line(output, point1, point2, (0, 0, 0), thickness, cv2.LINE_AA)
        cv2.line(mask, point1, point2, 255, thickness, cv2.LINE_AA)

    for line in vertical:
        point1 = (int(round(line.coordinate)), int(round(line.start)))
        point2 = (int(round(line.coordinate)), int(round(line.end)))
        cv2.line(output, point1, point2, (0, 0, 0), thickness, cv2.LINE_AA)
        cv2.line(mask, point1, point2, 255, thickness, cv2.LINE_AA)

    return output, mask


def draw_debug_overlay(
    rgb_image: np.ndarray,
    horizontal: Sequence[AxisLine],
    vertical: Sequence[AxisLine],
    horizontal_color: tuple[int, int, int],
    vertical_color: tuple[int, int, int],
    thickness: int = 2,
) -> np.ndarray:
    overlay = rgb_image.copy()

    for line in horizontal:
        cv2.line(
            overlay,
            (int(round(line.start)), int(round(line.coordinate))),
            (int(round(line.end)), int(round(line.coordinate))),
            horizontal_color,
            thickness,
            cv2.LINE_AA,
        )

    for line in vertical:
        cv2.line(
            overlay,
            (int(round(line.coordinate)), int(round(line.start))),
            (int(round(line.coordinate)), int(round(line.end))),
            vertical_color,
            thickness,
            cv2.LINE_AA,
        )

    return overlay


def process_page(
    rgb_image: np.ndarray,
    output_mode: OutputMode,
    line_thickness: int,
    horizontal_min_fraction: float,
    vertical_min_fraction: float,
    angle_tolerance: float,
    hough_threshold: int,
    repair_gap: int,
    coordinate_tolerance: int,
    intersection_tolerance: int,
    minimum_support: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, int]]:
    enhanced_gray, edges = prepare_edges(rgb_image)
    height, width = enhanced_gray.shape

    horizontal_min_length = max(30, int(round(width * horizontal_min_fraction)))
    vertical_min_length = max(30, int(round(height * vertical_min_fraction)))

    raw_h, raw_v = detect_hough_segments(
        edges=edges,
        horizontal_min_length=horizontal_min_length,
        vertical_min_length=vertical_min_length,
        angle_tolerance=angle_tolerance,
        hough_threshold=hough_threshold,
        repair_gap=repair_gap,
    )

    merged_h = merge_axis_lines(
        raw_h,
        coordinate_tolerance=coordinate_tolerance,
        merge_gap=repair_gap,
    )
    merged_v = merge_axis_lines(
        raw_v,
        coordinate_tolerance=coordinate_tolerance,
        merge_gap=repair_gap,
    )

    merged_h = [line for line in merged_h if line.length >= horizontal_min_length]
    merged_v = [line for line in merged_v if line.length >= vertical_min_length]

    verified_h, verified_v = select_verified_grid_lines(
        gray=enhanced_gray,
        horizontal=merged_h,
        vertical=merged_v,
        intersection_tolerance=intersection_tolerance,
        minimum_support=minimum_support,
    )

    if output_mode == "grayscale":
        base = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
    else:
        base = rgb_image.copy()

    processed, line_mask = draw_verified_lines(
        base,
        verified_h,
        verified_v,
        thickness=line_thickness,
    )

    debug_images = {
        "edges": edges,
        "all_candidates": draw_debug_overlay(
            rgb_image,
            merged_h,
            merged_v,
            horizontal_color=(255, 0, 0),
            vertical_color=(0, 0, 255),
        ),
        "verified_grid": draw_debug_overlay(
            rgb_image,
            verified_h,
            verified_v,
            horizontal_color=(0, 180, 0),
            vertical_color=(0, 180, 0),
            thickness=max(2, line_thickness),
        ),
        "line_mask": line_mask,
        "processed": processed,
    }

    counts = {
        "raw_horizontal": len(raw_h),
        "raw_vertical": len(raw_v),
        "merged_horizontal": len(merged_h),
        "merged_vertical": len(merged_v),
        "verified_horizontal": len(verified_h),
        "verified_vertical": len(verified_v),
    }

    return processed, debug_images, counts


def save_debug_images(
    debug_dir: Path,
    page_number: int,
    images: dict[str, np.ndarray],
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)

    for name, image in images.items():
        path = debug_dir / f"page_{page_number:04d}_{name}.png"

        if image.ndim == 3:
            image_to_save = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_to_save = image

        if not cv2.imwrite(str(path), image_to_save):
            raise RuntimeError(f"Could not save debug image: {path}")


def process_pdf(
    input_pdf: Path,
    output_pdf: Path,
    dpi: int,
    output_mode: OutputMode,
    line_thickness: int,
    horizontal_min_fraction: float,
    vertical_min_fraction: float,
    angle_tolerance: float,
    hough_threshold: int,
    repair_gap: int,
    coordinate_tolerance: int,
    intersection_tolerance: int,
    minimum_support: float,
    debug_dir: Path | None,
) -> None:
    if not input_pdf.exists():
        raise FileNotFoundError(f"Input PDF does not exist: {input_pdf}")
    if input_pdf.suffix.lower() != ".pdf":
        raise ValueError("Input file must be a PDF.")
    if dpi < 100:
        raise ValueError("DPI must be at least 100.")
    if not 0.0 < horizontal_min_fraction <= 1.0:
        raise ValueError("--min-horizontal-fraction must be between 0 and 1.")
    if not 0.0 < vertical_min_fraction <= 1.0:
        raise ValueError("--min-vertical-fraction must be between 0 and 1.")
    if not 0.0 <= minimum_support <= 1.0:
        raise ValueError("--minimum-support must be between 0 and 1.")

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    source = fitz.open(input_pdf)

    if source.needs_pass:
        source.close()
        raise PermissionError("Password-protected PDFs are not supported.")

    result = fitz.open()
    result.set_metadata(source.metadata)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    try:
        for page_index in range(source.page_count):
            page = source.load_page(page_index)
            pixmap = page.get_pixmap(
                matrix=matrix,
                colorspace=fitz.csRGB,
                alpha=False,
            )
            rgb_page = pixmap_to_rgb(pixmap)

            processed, debug_images, counts = process_page(
                rgb_image=rgb_page,
                output_mode=output_mode,
                line_thickness=line_thickness,
                horizontal_min_fraction=horizontal_min_fraction,
                vertical_min_fraction=vertical_min_fraction,
                angle_tolerance=angle_tolerance,
                hough_threshold=hough_threshold,
                repair_gap=repair_gap,
                coordinate_tolerance=coordinate_tolerance,
                intersection_tolerance=intersection_tolerance,
                minimum_support=minimum_support,
            )

            new_page = result.new_page(
                width=page.rect.width,
                height=page.rect.height,
            )
            new_page.insert_image(
                new_page.rect,
                stream=encode_png(processed),
                keep_proportion=False,
                overlay=True,
            )

            if debug_dir is not None:
                save_debug_images(debug_dir, page_index + 1, debug_images)

            print(
                f"Page {page_index + 1}/{source.page_count}: "
                f"verified H={counts['verified_horizontal']}, "
                f"V={counts['verified_vertical']}",
                flush=True,
            )

        result.save(output_pdf, garbage=4, clean=True, deflate=True)

    finally:
        result.close()
        source.close()


def main() -> int:
    input_pdf = Path(INPUT_PDF).expanduser()
    output_pdf = (
        Path(OUTPUT_PDF).expanduser()
        if OUTPUT_PDF is not None
        else input_pdf.with_name(f"{input_pdf.stem}_safe_table_lines.pdf")
    )
    debug_dir = Path(DEBUG_DIR).expanduser() if DEBUG_DIR is not None else None

    if not input_pdf.exists():
        print(
            f"Error: Input PDF was not found: {input_pdf.resolve()}",
            file=sys.stderr,
        )
        return 1

    try:
        process_pdf(
            input_pdf=input_pdf,
            output_pdf=output_pdf,
            dpi=DPI,
            output_mode=MODE,
            line_thickness=LINE_THICKNESS,
            horizontal_min_fraction=MIN_HORIZONTAL_FRACTION,
            vertical_min_fraction=MIN_VERTICAL_FRACTION,
            angle_tolerance=ANGLE_TOLERANCE,
            hough_threshold=HOUGH_THRESHOLD,
            repair_gap=REPAIR_GAP,
            coordinate_tolerance=COORDINATE_TOLERANCE,
            intersection_tolerance=INTERSECTION_TOLERANCE,
            minimum_support=MINIMUM_SUPPORT,
            debug_dir=debug_dir,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"\nCreated: {output_pdf.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
