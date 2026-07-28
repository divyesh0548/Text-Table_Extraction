from __future__ import annotations

import io
import os
import shutil
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download
from openpyxl import Workbook
from PIL import Image, ImageDraw

# Allow importing sibling modules from this OCR folder.
OCR_DIR = Path(__file__).resolve().parent
if str(OCR_DIR) not in sys.path:
    sys.path.insert(0, str(OCR_DIR))

import Microsoft_Table_Transformer_specific as mtt


st.set_page_config(layout="wide", page_title="Table Extraction")

mode = st.sidebar.radio(
    "Mode",
    ["Single image", "Dataset browser"],
    index=0,
)


@st.cache_resource(show_spinner="Loading Table Transformer models...")
def get_table_engine() -> mtt.TableTransformerEngine:
    return mtt.TableTransformerEngine()


def draw_detections(
    image: Image.Image,
    detections: list[mtt.Detection],
    color: str = "red",
    width: int = 3,
) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for item in detections:
        draw.rectangle(item.box, outline=color, width=width)
        label = f"{item.label} ({item.score:.2f})"
        x1, y1, _, _ = item.box
        draw.text((x1 + 4, max(0, y1 - 14)), label, fill=color)
    return annotated


def extract_tables_from_image(
    page_image: Image.Image,
    ocr_function=mtt.default_ocr,
) -> list[dict]:
    """
    Detect tables on one image, recognize structure, OCR cells.
    Returns one dict per extracted table.
    """
    engine = get_table_engine()
    tables = engine.detect_tables(page_image)
    results: list[dict] = []

    # If the upload is already a cropped table, detection can miss it.
    # Fall back to treating the full image as one table.
    if not tables:
        tables = [
            mtt.Detection(
                label="table",
                score=1.0,
                box=(0.0, 0.0, float(page_image.width), float(page_image.height)),
            )
        ]
        used_full_image_fallback = True
    else:
        used_full_image_fallback = False

    for table_number, table in enumerate(tables, start=1):
        crop_box = mtt.clamp_box(
            table.box,
            page_image.width,
            page_image.height,
            padding=0 if used_full_image_fallback else mtt.TABLE_PADDING,
        )
        table_image = page_image.crop(crop_box)

        if table.label == "table rotated":
            table_image = table_image.rotate(90, expand=True)

        structure = engine.recognize_structure(table_image)
        rows, columns, spans = mtt.prepare_rows_columns_and_spans(structure)

        structure_image = table_image.copy()
        draw = ImageDraw.Draw(structure_image)
        for row in rows:
            draw.rectangle(row.box, outline="red", width=2)
        for column in columns:
            draw.rectangle(column.box, outline="blue", width=2)
        for span in spans:
            draw.rectangle(span.box, outline="green", width=3)

        data: list[list[str]] = []
        merge_ranges: list[tuple[int, int, int, int]] = []

        if rows and columns:
            data, merge_ranges = mtt.extract_grid(
                table_image,
                rows,
                columns,
                spans,
                ocr_function,
            )
            data = [
                [mtt.cleanup_extracted_cell_text(value) for value in row]
                for row in data
            ]
            data, merge_ranges = mtt.prune_generated_columns(data, merge_ranges)
            data, merge_ranges = mtt.deduplicate_adjacent_rows(data, merge_ranges)

        results.append(
            {
                "table_number": table_number,
                "label": table.label,
                "score": table.score,
                "box": table.box,
                "used_full_image_fallback": used_full_image_fallback,
                "table_image": table_image,
                "structure_image": structure_image,
                "row_count": len(rows),
                "column_count": len(columns) if not data else (len(data[0]) if data else 0),
                "data": data,
                "merge_ranges": merge_ranges,
            }
        )

    return results


def tables_to_excel_bytes(tables: list[dict]) -> bytes:
    workbook = Workbook()
    workbook.remove(workbook.active)

    for item in tables:
        data = item["data"]
        if not data:
            worksheet = workbook.create_sheet(
                title=mtt.safe_sheet_name(f"Table{item['table_number']}_empty")
            )
            worksheet["A1"] = "No usable row/column structure was detected."
            continue

        mtt.write_table_sheet(
            workbook,
            sheet_name=f"Table{item['table_number']}",
            data=data,
            merge_ranges=item["merge_ranges"],
        )

    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


# =============================================================================
# SINGLE IMAGE MODE
# =============================================================================

if mode == "Single image":
    st.title("Table detection & extraction")
    st.caption(
        "Upload one image. The app detects table regions with Microsoft Table "
        "Transformer, then extracts cell text with OCR."
    )

    uploaded = st.file_uploader(
        "Upload table image",
        type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
        accept_multiple_files=False,
    )

    local_path = st.text_input(
        "Or local image path",
        value="",
        placeholder=r"C:\path\to\table.png",
    )

    image = None
    image_label = None

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        image_label = uploaded.name
    elif local_path.strip():
        path = Path(local_path.strip())
        if path.exists():
            image = Image.open(path).convert("RGB")
            image_label = str(path)
        else:
            st.error(f"File not found: {path}")

    if image is None:
        st.info("Upload an image or enter a local path to continue.")
        st.stop()

    st.session_state["table_image"] = image
    st.session_state["table_image_label"] = image_label

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Input image")
        st.image(image, caption=image_label, use_container_width=True)

    run = st.button("Detect & extract tables", type="primary")

    if not run and "last_extraction" not in st.session_state:
        st.info('Click "Detect & extract tables" to process this image.')
        st.stop()

    if run:
        with st.spinner("Detecting tables and running OCR..."):
            try:
                extraction = extract_tables_from_image(image)
                st.session_state["last_extraction"] = {
                    "label": image_label,
                    "detection_image": draw_detections(
                        image,
                        [
                            mtt.Detection(
                                label=item["label"],
                                score=item["score"],
                                box=item["box"],
                            )
                            for item in extraction
                            if not item["used_full_image_fallback"]
                        ],
                    ),
                    "tables": extraction,
                }
            except Exception as exc:
                st.exception(exc)
                st.stop()

    result = st.session_state.get("last_extraction")
    if not result or result.get("label") != image_label:
        st.info('Click "Detect & extract tables" to process this image.')
        st.stop()

    tables = result["tables"]
    with right:
        st.subheader("Detected table regions")
        if tables and tables[0]["used_full_image_fallback"]:
            st.warning(
                "No separate table region was detected. "
                "Processed the full image as one table."
            )
            st.image(image, caption="Full-image fallback", use_container_width=True)
        else:
            st.image(
                result["detection_image"],
                caption=f"{len(tables)} table region(s)",
                use_container_width=True,
            )

    st.success(f"Processed {len(tables)} table(s).")
    st.download_button(
        "Download Excel",
        data=tables_to_excel_bytes(tables),
        file_name=f"{Path(image_label).stem}_tables.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    for item in tables:
        st.markdown("---")
        st.subheader(
            f"Table {item['table_number']}  |  "
            f"{item['row_count']} rows × {item['column_count']} cols  |  "
            f"score={item['score']:.2f}"
        )

        c1, c2 = st.columns(2)
        with c1:
            st.caption("Cropped table")
            st.image(item["table_image"], use_container_width=True)
        with c2:
            st.caption("Structure (red=rows, blue=columns, green=spans)")
            st.image(item["structure_image"], use_container_width=True)

        if item["data"]:
            dataframe = pd.DataFrame(item["data"])
            st.dataframe(dataframe, use_container_width=True)
        else:
            st.error("No usable row/column structure was detected for this table.")

    st.stop()


# =============================================================================
# DATASET BROWSER MODE (original rd-tablebench viewer)
# =============================================================================

with st.spinner("Downloading dataset"):
    results = hf_hub_download(
        repo_id="reducto/rd-tablebench",
        filename="rd-tablebench.zip",
        repo_type="dataset",
    )


def unzip_dataset():
    if not os.path.exists("unzipped_dataset"):
        os.makedirs("unzipped_dataset")
        with st.spinner("Unzipping dataset"):
            with zipfile.ZipFile(results, "r") as zip_ref:
                zip_ref.extractall("unzipped_dataset")
    return "unzipped_dataset/rd-tablebench"


if st.button("Redo Unzip"):
    if os.path.exists("unzipped_dataset"):
        shutil.rmtree("unzipped_dataset")
        st.rerun()


dataset = unzip_dataset()

results = f"{dataset}/providers/scores.csv"

assert os.path.exists(results)

st.html("""
<style>
table {
  font-family: arial, sans-serif;
  border-collapse: collapse;
  white-space: pre;
}

td, th {
  border: 1px solid #dddddd;
  text-align: left;
  padding: 8px;
  font-weight: normal;
}

</style>
""")


df = pd.read_csv(results)

if "current_index" not in st.session_state:
    st.session_state.current_index = 0

col1, col2, col3 = st.columns([2, 5, 2])

with col1:
    st.html("<br/>")
    if st.button("⬅️ Previous", use_container_width=True):
        if st.session_state.current_index > 0:
            st.session_state.current_index -= 1
            st.rerun()

with col2:
    index_input = st.number_input(
        "Index",
        label_visibility="hidden",
        min_value=0,
        max_value=len(df) - 1,
        value=st.session_state.current_index,
        step=1,
    )

    if st.button("Go", use_container_width=True):
        st.session_state.current_index = int(index_input)
        st.rerun()

with col3:
    st.html("<br/>")
    if st.button("Next ➡️", use_container_width=True):
        if st.session_state.current_index < len(df) - 1:
            st.session_state.current_index += 1
            st.rerun()


col1, col2 = st.columns([1, 2])

providers = [
    "reducto",
    "azure",
    "textract",
    "gcloud",
    "unstructured",
    "gpt4o",
    "chunkr",
]

with col1:
    row = df.iloc[st.session_state.current_index]

    scores = [
        row[f"{p}_score"] if row[f"{p}_score"] is not None else 0 for p in providers
    ]

    fig, ax = plt.subplots(figsize=(6, 10))
    bars = ax.barh(providers[::-1], scores[::-1])

    ax.set_title("Provider Scores Comparison")
    ax.set_ylabel("Providers")
    ax.set_xlabel("Scores")
    ax.set_xlim(0, 1.1)

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2.0,
            f"{width:.3f}",
            ha="left",
            va="center",
        )

    plt.tight_layout()
    st.pyplot(fig)

with col2:
    image_path = f"{dataset}/_images/{row['pdf_path'].replace('.pdf', '.jpg')}"
    st.image(image_path, use_container_width=True)

st.write(row)
st.subheader("Groundtruth")
st.html(f"{dataset}/groundtruth/{row['pdf_path'].replace('.pdf', '.html')}")

st.subheader("Provider Outputs")
for p in providers:
    with st.expander(p):
        provider_html = (
            f"{dataset}/providers/{p}/{row['pdf_path'].replace('.pdf', '.html')}"
        )
        if os.path.exists(provider_html):
            st.html(provider_html)
        else:
            st.error(f"{p} failed to produce a table output for this image")
