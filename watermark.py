import fitz  # PyMuPDF


def remove_watermark(input_pdf_path: str, output_pdf_path: str, watermark_text: str):
    doc = fitz.open(input_pdf_path)

    for page in doc:
        # find all occurrences of the watermark text
        areas = page.search_for(watermark_text)
        for area in areas:
            # add a redaction annotation over exactly that text
            page.add_redact_annot(area, fill=(1, 1, 1))

        # apply all redactions on this page
        if areas:
            page.apply_redactions()

    # save cleaned PDF
    doc.save(output_pdf_path, garbage=4, deflate=True)
    doc.close()


if __name__ == "__main__":
    input_pdf = "sample7.pdf"
    output_pdf = "sample7_cleaned_exact.pdf"
    watermark_phrase = "LET EXPORT COPY"
    remove_watermark(input_pdf, output_pdf, watermark_phrase)
    print(f"Exact watermark removal complete. Saved as {output_pdf}")
