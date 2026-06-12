"""
Document loaders — PDF, TXT, and plain text support.
No heavy dependencies: uses only PyPDF2 (optional) and stdlib.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple


def load_txt(file_path: str) -> str:
    """Load plain text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_pdf(file_path: str) -> str:
    """
    Load PDF using PyPDF2.
    Falls back to a helpful error if not installed.
    """
    try:
        import PyPDF2
        text_parts = []
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    text_parts.append(f"[صفحة {page_num + 1}]\n{text}")
        return '\n\n'.join(text_parts)
    except ImportError:
        raise ImportError(
            "PyPDF2 is required for PDF support: pip install PyPDF2"
        )
    except Exception as e:
        raise ValueError(f"Failed to read PDF {file_path}: {e}")


def clean_text(text: str) -> str:
    """
    Clean extracted text:
    - Remove excessive whitespace
    - Fix common PDF artifacts
    - Preserve Arabic text integrity
    """
    # Normalize newlines
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove non-printable characters (except Arabic/common Unicode)
    text = re.sub(r'[^\S\n]+', ' ', text)
    # Remove very short lines (likely artifacts)
    lines = text.split('\n')
    cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 3]
    return '\n'.join(cleaned_lines)


def load_documents_from_folder(folder_path: str) -> List[Tuple[str, str]]:
    """
    Load all .txt and .pdf files from a folder.
    Returns: list of (content, filename) tuples
    """
    folder = Path(folder_path)
    documents = []
    supported = {'.txt', '.pdf', '.md'}

    for file_path in sorted(folder.iterdir()):
        if file_path.suffix.lower() not in supported:
            continue

        try:
            print(f"📄 Loading: {file_path.name}")

            if file_path.suffix.lower() == '.pdf':
                raw_text = load_pdf(str(file_path))
            else:
                raw_text = load_txt(str(file_path))

            cleaned = clean_text(raw_text)

            if len(cleaned) < 50:
                print(f"  ⚠️  Skipped (too short): {file_path.name}")
                continue

            documents.append((cleaned, file_path.name))
            print(f"  ✅ {len(cleaned):,} characters loaded")

        except Exception as e:
            print(f"  ❌ Error loading {file_path.name}: {e}")

    print(f"\n📚 Total documents loaded: {len(documents)}")
    return documents


def load_document_from_bytes(file_bytes: bytes, filename: str) -> Tuple[str, str]:
    """
    Load document from raw bytes (for web upload).
    Returns: (content, filename)
    """
    import tempfile

    suffix = Path(filename).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        if suffix == '.pdf':
            raw_text = load_pdf(tmp_path)
        else:
            raw_text = file_bytes.decode('utf-8', errors='ignore')

        cleaned = clean_text(raw_text)
        return (cleaned, filename)
    finally:
        os.unlink(tmp_path)
