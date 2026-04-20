"""
parser.py
=========
Handles all file reading and raw text extraction.

Responsibilities:
  - Accept a Streamlit UploadedFile object
  - Detect file type (PDF or TXT)
  - Extract and return raw text
  - Handle errors gracefully without crashing the app

This module has NO dependency on LangChain or OpenAI —
it is purely about getting text out of files.
"""

import io
import tempfile
import os


def extract_text_from_file(uploaded_file) -> str:
    """
    Extract raw text from a Streamlit UploadedFile (PDF or TXT).

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        Extracted text as a plain string.
        Returns empty string if extraction fails.
    """
    file_name = uploaded_file.name
    extension = file_name.split(".")[-1].lower()

    if extension == "pdf":
        return _extract_from_pdf(uploaded_file)
    elif extension == "txt":
        return _extract_from_txt(uploaded_file)
    else:
        # Unsupported type — return empty and let the caller handle it
        return ""


def _extract_from_pdf(uploaded_file) -> str:
    """
    Extract text from a PDF using pypdf.

    pypdf reads page-by-page and returns text per page.
    We join all pages with newlines to get a single string.

    Note: pypdf works well for text-based PDFs.
    Scanned/image PDFs would need OCR (out of scope for this project).
    """
    try:
        # Import here so the rest of the app works even if pypdf is missing
        from pypdf import PdfReader

        # pypdf needs a file-like object — wrap the uploaded bytes
        pdf_bytes = io.BytesIO(uploaded_file.read())
        reader = PdfReader(pdf_bytes)

        pages_text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:  # Some pages may be blank or image-only
                pages_text.append(page_text)

        full_text = "\n".join(pages_text)
        print(f"[INFO] Extracted {len(full_text)} characters from PDF ({len(reader.pages)} pages)")
        return full_text

    except Exception as e:
        print(f"[ERROR] PDF extraction failed: {e}")
        return ""


def _extract_from_txt(uploaded_file) -> str:
    """
    Read a plain text file.
    Handles common encodings gracefully.
    """
    try:
        raw_bytes = uploaded_file.read()
        # Try UTF-8 first, fall back to latin-1 (handles most résumé files)
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = raw_bytes.decode("latin-1")

        print(f"[INFO] Extracted {len(text)} characters from TXT file")
        return text

    except Exception as e:
        print(f"[ERROR] TXT extraction failed: {e}")
        return ""


def clean_text(text: str) -> str:
    """
    Light text cleaning to remove excessive whitespace.
    Keeps the content intact — we don't want to remove keywords.

    Args:
        text: Raw extracted text

    Returns:
        Cleaned text with normalised whitespace
    """
    # Collapse multiple blank lines into one
    import re
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove trailing spaces on each line
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()
