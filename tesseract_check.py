import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Divyesh Parmar\Downloads\Tesserect OCR\tesseract.exe'

print(pytesseract.get_tesseract_version())