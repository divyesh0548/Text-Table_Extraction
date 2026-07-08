import fitz  # PyMuPDF
import re

def insert_digital_header_fixed_font(input_pdf, output_pdf):
    """
    Extract header text and redraw it as real text on each page,
    using a standard font to avoid 'need font file or buffer' error.
    """
    try:
        pad = float(input("Enter padding above first row (pts): "))
    except:
        pad = 10.0

    doc = fitz.open(input_pdf)
    first = doc[0]
    w, h = first.rect.width, first.rect.height

    keys = ["DATE", "NARRATION", "CHQ.NO.", "WITHDRAWAL(DR)", "DEPOSIT(CR)", "BALANCE(INR)"]
    spans = []
    for block in first.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                txt = span["text"].strip()
                if txt.upper() in keys:
                    spans.append(span)
    if not spans:
        raise RuntimeError("Header not found")

    x0 = min(s["bbox"][0] for s in spans) - 2
    x1 = max(s["bbox"][2] for s in spans) + 2
    y0 = min(s["bbox"][1] for s in spans) - 2
    y1 = max(s["bbox"][3] for s in spans) + 2
    header_box = fitz.Rect(x0, y0, x1, y1)

    # Gather header text and relative x position
    header_texts = [(s["text"], s["bbox"][0]) for s in spans]

    for i in range(len(doc)):
        page = doc[i]

        date_y = None
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                line_text = "".join(s["text"] for s in line["spans"])
                if re.search(r"\d{2}/\d{2}/\d{4}", line_text):
                    date_y = line["spans"][0]["bbox"][1]
                    break
            if date_y:
                break
        if not date_y:
            continue

        target_rect = fitz.Rect(x0, date_y - pad - (y1 - y0), x1, date_y - pad)
        page.draw_rect(target_rect, fill=(1,1,1))

        total_width = x1 - x0
        original_width = header_box.width

        for text, sx in header_texts:
            rel_x = sx - x0
            scaled_x = x0 + (rel_x / original_width) * total_width
            y = target_rect.y0 + 2
            # Use standard font 'helv' to avoid font file error
            page.insert_text((scaled_x, y), text, fontname="helv", fontsize=11, color=(0,0,0))

    doc.save(output_pdf)
    doc.close()
    print(f"Saved PDF with digital header and fixed font: {output_pdf}")

if __name__ == "__main__":
    insert_digital_header_fixed_font("bs1.pdf", "bs1_with_digital_headers_fixed_font.pdf")
