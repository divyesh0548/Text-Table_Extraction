
import os
import logging
from pathlib import Path
from typing import Optional, Union

# Multiple library imports for different approaches
try:
    import pikepdf
    PIKEPDF_AVAILABLE = True
except ImportError:
    PIKEPDF_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from pypdf import PdfReader, PdfWriter
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

# Configure logging (optional - can be disabled)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def remove_pdf_restrictions(pdf_path: Union[str, Path], 
                          output_path: Optional[Union[str, Path]] = None,
                          password: Optional[str] = None,
                          remove_metadata: bool = True,
                          verbose: bool = True) -> bool:

    # Set up logging based on verbose flag
    if not verbose:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    # Convert paths to Path objects
    pdf_path = Path(pdf_path)

    # Validate input file
    if not pdf_path.exists():
        logger.error(f"Input file not found: {pdf_path}")
        return False

    if not pdf_path.suffix.lower() == '.pdf':
        logger.error(f"Input file is not a PDF: {pdf_path}")
        return False

    # Generate output path if not provided
    if output_path is None:
        output_path = pdf_path.parent / f"{pdf_path.stem}_unrestricted{pdf_path.suffix}"
    else:
        output_path = Path(output_path)

    logger.info(f"Processing: {pdf_path.name}")
    logger.info(f"Output will be saved to: {output_path}")

    # Try different methods in order of preference
    methods = [
        ("pikepdf", _remove_with_pikepdf),
        ("pymupdf", _remove_with_pymupdf), 
        ("pypdf", _remove_with_pypdf),
        ("pypdf2", _remove_with_pypdf2)
    ]

    for method_name, method_func in methods:
        if method_name == "pikepdf" and not PIKEPDF_AVAILABLE:
            continue
        elif method_name == "pymupdf" and not PYMUPDF_AVAILABLE:
            continue
        elif method_name == "pypdf" and not PYPDF_AVAILABLE:
            continue
        elif method_name == "pypdf2" and not PYPDF2_AVAILABLE:
            continue

        logger.info(f"Trying {method_name} method...")

        try:
            success = method_func(pdf_path, output_path, password, remove_metadata)
            if success:
                _verify_output(output_path, verbose)
                logger.info(f"✅ Successfully processed with {method_name}")
                return True
        except Exception as e:
            logger.warning(f"{method_name} method failed: {e}")
            continue

    logger.error("❌ All available methods failed")
    return False


def _remove_with_pikepdf(pdf_path: Path, output_path: Path, password: Optional[str], remove_metadata: bool) -> bool:
    """Remove restrictions using pikepdf."""
    try:
        # Open PDF
        if password:
            pdf = pikepdf.Pdf.open(pdf_path, password=password)
        else:
            pdf = pikepdf.Pdf.open(pdf_path)

        # Remove metadata if requested
        if remove_metadata and pdf.docinfo:
            for key in list(pdf.docinfo.keys()):
                del pdf.docinfo[key]

        # Remove XMP metadata
        if remove_metadata and '/Metadata' in pdf.Root:
            del pdf.Root.Metadata

        # Save without encryption (removes all restrictions)
        pdf.save(output_path, encryption=False, linearize=True)
        pdf.close()

        return True

    except Exception as e:
        logger.error(f"pikepdf error: {e}")
        return False


def _remove_with_pymupdf(pdf_path: Path, output_path: Path, password: Optional[str], remove_metadata: bool) -> bool:
    """Remove restrictions using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)

        # Handle password
        if doc.needs_pass:
            if password and doc.authenticate(password):
                pass
            else:
                logger.error("PDF requires password")
                return False

        # Remove metadata if requested
        if remove_metadata:
            metadata = {key: "" for key in ['title', 'author', 'subject', 'keywords', 'creator', 'producer', 'creationDate', 'modDate']}
            doc.set_metadata(metadata)
            try:
                doc.set_xml_metadata("")
            except:
                pass

        # Save without restrictions
        doc.save(output_path, 
                garbage=4,
                linear=True,
                clean=True,
                encryption=fitz.PDF_ENCRYPT_NONE,
                permissions=fitz.PDF_PERM_ACCESSIBILITY | fitz.PDF_PERM_PRINT | fitz.PDF_PERM_COPY | 
                           fitz.PDF_PERM_ANNOTATE | fitz.PDF_PERM_FORM | fitz.PDF_PERM_ROTATE | fitz.PDF_PERM_ASSEMBLE)

        doc.close()
        return True

    except Exception as e:
        logger.error(f"PyMuPDF error: {e}")
        return False


def _remove_with_pypdf(pdf_path: Path, output_path: Path, password: Optional[str], remove_metadata: bool) -> bool:
    """Remove restrictions using pypdf."""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)

            if reader.is_encrypted:
                if password and reader.decrypt(password):
                    pass
                else:
                    logger.error("PDF requires password")
                    return False

            writer = PdfWriter()

            for page in reader.pages:
                writer.add_page(page)

            if remove_metadata and hasattr(writer, 'add_metadata'):
                writer.add_metadata({})

            with open(output_path, 'wb') as output_file:
                writer.write(output_file)

        return True

    except Exception as e:
        logger.error(f"pypdf error: {e}")
        return False


def _remove_with_pypdf2(pdf_path: Path, output_path: Path, password: Optional[str], remove_metadata: bool) -> bool:
    """Remove restrictions using PyPDF2."""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfFileReader(file)

            if reader.isEncrypted:
                if password and reader.decrypt(password):
                    pass
                else:
                    logger.error("PDF requires password")
                    return False

            writer = PyPDF2.PdfFileWriter()

            for i in range(reader.numPages):
                page = reader.getPage(i)
                writer.addPage(page)

            if remove_metadata:
                writer.addMetadata({})

            with open(output_path, 'wb') as output_file:
                writer.write(output_file)

        return True

    except Exception as e:
        logger.error(f"PyPDF2 error: {e}")
        return False


def _verify_output(output_path: Path, verbose: bool):
    """Verify the output file was created successfully."""
    if output_path.exists():
        file_size = output_path.stat().st_size
        if verbose:
            logger.info(f"Output file created: {output_path}")
            logger.info(f"File size: {file_size:,} bytes")
    else:
        logger.error(f"Output file was not created: {output_path}")


def batch_remove_pdf_restrictions(input_directory: Union[str, Path], 
                                 output_directory: Optional[Union[str, Path]] = None,
                                 password: Optional[str] = None,
                                 remove_metadata: bool = True,
                                 verbose: bool = True) -> int:

    input_dir = Path(input_directory)
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_directory}")
        return 0

    output_dir = Path(output_directory) if output_directory else input_dir
    output_dir.mkdir(exist_ok=True)

    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {input_directory}")
        return 0

    if verbose:
        logger.info(f"Found {len(pdf_files)} PDF files to process")

    success_count = 0
    for pdf_file in pdf_files:
        output_file = output_dir / f"{pdf_file.stem}_unrestricted{pdf_file.suffix}"

        if verbose:
            logger.info(f"Processing: {pdf_file.name}")

        if remove_pdf_restrictions(pdf_file, output_file, password, remove_metadata, verbose=False):
            success_count += 1
            if verbose:
                logger.info(f"✅ Successfully processed: {pdf_file.name}")
        else:
            if verbose:
                logger.error(f"❌ Failed to process: {pdf_file.name}")

    if verbose:
        logger.info(f"Batch processing complete: {success_count}/{len(pdf_files)} files processed successfully")

    return success_count


# Convenience function with minimal parameters
def clean_pdf(pdf_file: str, output_file: str = None, password: str = None) -> bool:
    return remove_pdf_restrictions(pdf_file, output_file, password, remove_metadata=True, verbose=True)


if __name__ == "__main__":
    # Example usage when run as a script
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_cleaner_module.py <pdf_file> [output_file] [password]")
        print("Example: python pdf_cleaner_module.py document.pdf")
        print("Example: python pdf_cleaner_module.py document.pdf cleaned.pdf")
        print("Example: python pdf_cleaner_module.py document.pdf cleaned.pdf mypassword")
        sys.exit(1)

    pdf_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    password = sys.argv[3] if len(sys.argv) > 3 else None

    success = remove_pdf_restrictions(pdf_file, output_file, password)
    sys.exit(0 if success else 1)
