"""
Visualize PaddleOCR / PaddleX table structure recognition output.

Input JSON fields:
  - input_path : source image path
  - bbox       : list of cell polygons [x1,y1,x2,y2,x3,y3,x4,y4]
  - structure  : tokenized HTML table structure (SLANet)
  - structure_score : model confidence

Outputs (in output_dir):
  - *_cells_overlay.png   : cell polygons on source image with indices
  - *_structure_grid.png  : schematic logical table grid
  - *_combined.png        : overlay + grid side by side
  - *_structure.html      : reconstructed HTML table
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib import colors as mcolors
from PIL import Image


ROW_COLORS = list(mcolors.TABLEAU_COLORS.values())


def load_result(json_path: str | Path) -> dict[str, Any]:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def resolve_image_path(result: dict[str, Any], json_path: Path) -> Path:
    raw = result.get("input_path", "")
    candidate = Path(raw)
    if candidate.is_file():
        return candidate.resolve()

    # Try relative to JSON file, then its parent folders
    bases = [json_path.parent, json_path.parent.parent, Path.cwd()]
    for base in bases:
        probe = (base / raw).resolve()
        if probe.is_file():
            return probe
        probe = (base / candidate.name).resolve()
        if probe.is_file():
            return probe

    raise FileNotFoundError(f"Could not locate source image for: {raw}")


def structure_tokens_to_html(tokens: list[str]) -> str:
    return "".join(tokens)


def parse_table_rows(structure_tokens: list[str]) -> list[list[dict[str, Any]]]:
    """Parse tokenized HTML into rows of cells with colspan/rowspan."""
    html = structure_tokens_to_html(structure_tokens)
    rows: list[list[dict[str, Any]]] = []

    for tr_match in re.finditer(r"<tr>(.*?)</tr>", html, flags=re.IGNORECASE | re.DOTALL):
        row_cells: list[dict[str, Any]] = []
        for td_match in re.finditer(
            r"<td([^>]*)>(.*?)</td>", tr_match.group(1), flags=re.IGNORECASE | re.DOTALL
        ):
            attrs = td_match.group(1)
            colspan = 1
            rowspan = 1
            if m := re.search(r'colspan=["\']?(\d+)', attrs, re.IGNORECASE):
                colspan = int(m.group(1))
            if m := re.search(r'rowspan=["\']?(\d+)', attrs, re.IGNORECASE):
                rowspan = int(m.group(1))
            row_cells.append(
                {
                    "colspan": colspan,
                    "rowspan": rowspan,
                    "text": td_match.group(2).strip(),
                }
            )
        rows.append(row_cells)
    return rows


def build_cell_layout(rows: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """
    Assign each cell a grid position (row, col) accounting for spans.
    Cell order matches bbox order in the JSON.
    """
    grid: list[list[int | None]] = []
    cells: list[dict[str, Any]] = []

    for row_idx, row in enumerate(rows):
        while len(grid) <= row_idx:
            grid.append([])

        col_idx = 0
        for cell in row:
            while col_idx < len(grid[row_idx]) and grid[row_idx][col_idx] is not None:
                col_idx += 1

            rs, cs = cell["rowspan"], cell["colspan"]
            cell_id = len(cells)
            cells.append(
                {
                    **cell,
                    "id": cell_id,
                    "row": row_idx,
                    "col": col_idx,
                }
            )

            for r in range(row_idx, row_idx + rs):
                while len(grid) <= r:
                    grid.append([])
                for c in range(col_idx, col_idx + cs):
                    while len(grid[r]) <= c:
                        grid[r].append(None)
                    grid[r][c] = cell_id
            col_idx += cs

    max_cols = max((len(r) for r in grid), default=0)
    return cells


def bbox_to_polygon(bbox: list[float | int]) -> np.ndarray:
    pts = np.array(bbox, dtype=np.float32).reshape(-1, 2)
    return pts.astype(np.int32)


def draw_cell_overlay(
    image_path: Path,
    bboxes: list[list[float | int]],
    output_path: Path,
    structure_rows: list[list[dict[str, Any]]] | None = None,
) -> Path:
    """Draw detected cell polygons on the source image with index labels."""
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    canvas = image_bgr.copy()
    flat_cells = [c for row in (structure_rows or []) for c in row]

    for idx, bbox in enumerate(bboxes):
        pts = bbox_to_polygon(bbox)
        rgb = mcolors.to_rgb(ROW_COLORS[idx % len(ROW_COLORS)])
        color_bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))

        tint = canvas.copy()
        cv2.fillPoly(tint, [pts], color_bgr)
        cv2.addWeighted(tint, 0.22, canvas, 0.78, 0, canvas)
        cv2.polylines(canvas, [pts], isClosed=True, color=color_bgr, thickness=2)

        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        label = str(idx)
        if idx < len(flat_cells) and flat_cells[idx]["colspan"] > 1:
            label += f" (c{flat_cells[idx]['colspan']})"

        cv2.putText(
            canvas,
            label,
            (cx - 12, cy + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            (cx - 12, cy + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    return output_path


def draw_structure_grid(
    structure_tokens: list[str],
    output_path: Path,
    title: str = "Predicted table structure",
) -> Path:
    """Render the logical HTML table as a schematic grid."""
    rows = parse_table_rows(structure_tokens)
    cells = build_cell_layout(rows)

    if not cells:
        raise ValueError("No table cells found in structure tokens.")

    max_row = max(c["row"] + c["rowspan"] for c in cells)
    max_col = max(c["col"] + c["colspan"] for c in cells)

    fig_w = max(6, max_col * 1.4)
    fig_h = max(4, max_row * 0.9)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, max_col)
    ax.set_ylim(0, max_row)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=12)

    for cell in cells:
        cid = cell["id"]
        color = ROW_COLORS[cid % len(ROW_COLORS)]
        rect = patches.Rectangle(
            (cell["col"], cell["row"]),
            cell["colspan"],
            cell["rowspan"],
            linewidth=1.5,
            edgecolor="#222222",
            facecolor=color,
            alpha=0.35,
        )
        ax.add_patch(rect)

        label_lines = [f"#{cid}"]
        if cell["colspan"] > 1:
            label_lines.append(f"colspan={cell['colspan']}")
        if cell["rowspan"] > 1:
            label_lines.append(f"rowspan={cell['rowspan']}")
        if cell["text"]:
            label_lines.append(cell["text"][:18])

        ax.text(
            cell["col"] + cell["colspan"] / 2,
            cell["row"] + cell["rowspan"] / 2,
            "\n".join(label_lines),
            ha="center",
            va="center",
            fontsize=8,
            color="#111111",
        )

    # Light grid lines
    for x in range(max_col + 1):
        ax.axvline(x, color="#dddddd", linewidth=0.5)
    for y in range(max_row + 1):
        ax.axhline(y, color="#dddddd", linewidth=0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def draw_combined(
    overlay_path: Path,
    grid_path: Path,
    output_path: Path,
    score: float | None = None,
) -> Path:
    """Stack cell overlay and structure grid side by side."""
    left = Image.open(overlay_path).convert("RGB")
    right = Image.open(grid_path).convert("RGB")

    # Match heights
    target_h = max(left.height, right.height)

    def resize_to_height(img: Image.Image, h: int) -> Image.Image:
        ratio = h / img.height
        return img.resize((int(img.width * ratio), h), Image.Resampling.LANCZOS)

    left = resize_to_height(left, target_h)
    right = resize_to_height(right, target_h)

    gap = 20
    canvas = Image.new("RGB", (left.width + gap + right.width, target_h + 40), "white")
    canvas.paste(left, (0, 30))
    canvas.paste(right, (left.width + gap, 30))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)

    if score is not None:
        # Re-open and add title via matplotlib for simplicity
        fig, ax = plt.subplots(figsize=(canvas.width / 100, canvas.height / 100))
        ax.imshow(canvas)
        ax.axis("off")
        ax.set_title(f"Table structure visualization  |  score={score:.3f}", fontsize=11)
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    return output_path


def export_structure_html(structure_tokens: list[str], output_path: Path) -> Path:
    html = structure_tokens_to_html(structure_tokens)
    if not html.lower().startswith("<html"):
        html = f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>table,td,th{{border:1px solid #333;border-collapse:collapse;padding:8px;}}table{{border-collapse:collapse;}}</style></head><body>{html}</body></html>"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def visualize_result(
    result: dict[str, Any],
    output_dir: str | Path,
    stem: str | None = None,
    image_path: str | Path | None = None,
) -> dict[str, str]:
    """
    Create all visualizations from a result dict or loaded JSON.
    Returns paths of generated files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if stem is None:
        src = result.get("input_path", "table")
        stem = Path(src).stem

    bboxes = result.get("bbox", [])
    structure = result.get("structure", [])
    score = result.get("structure_score")

    if image_path is None:
        image_path = resolve_image_path(result, output_dir / "res.json")
    else:
        image_path = Path(image_path)

    structure_rows = parse_table_rows(structure)

    paths = {
        "cells_overlay": str(
            draw_cell_overlay(
                image_path,
                bboxes,
                output_dir / f"{stem}_cells_overlay.png",
                structure_rows=structure_rows,
            )
        ),
        "structure_grid": str(
            draw_structure_grid(
                structure,
                output_dir / f"{stem}_structure_grid.png",
                title=f"Table structure ({stem})",
            )
        ),
        "structure_html": str(
            export_structure_html(structure, output_dir / f"{stem}_structure.html")
        ),
    }
    paths["combined"] = str(
        draw_combined(
            Path(paths["cells_overlay"]),
            Path(paths["structure_grid"]),
            output_dir / f"{stem}_combined.png",
            score=score,
        )
    )
    return paths


def visualize_from_json(json_path: str | Path, output_dir: str | Path | None = None) -> dict[str, str]:
    json_path = Path(json_path)
    result = load_result(json_path)
    out = Path(output_dir) if output_dir else json_path.parent
    stem = json_path.stem.replace("_res", "") if json_path.stem.endswith("_res") else json_path.stem
    return visualize_result(result, out, stem=stem, image_path=resolve_image_path(result, json_path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize table structure recognition JSON.")
    parser.add_argument("json_path", help="Path to res.json from TableStructureRecognition")
    parser.add_argument("-o", "--output-dir", default=None, help="Output directory")
    args = parser.parse_args()

    paths = visualize_from_json(args.json_path, args.output_dir)
    print("Saved visualizations:")
    for key, path in paths.items():
        print(f"  {key:16} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
