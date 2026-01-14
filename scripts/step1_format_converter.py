#!/usr/bin/env python3
"""
Step 1: File Format Conversion
==============================
Converts various document formats (.doc, .docx, .pdf, .html, .rtf) 
to UTF-8 encoded plain text files.

Usage:
    python step1_format_converter.py --input /path/to/raw --output /path/to/converted --workers 8
"""

import os
import re
import sys
import logging
import argparse
import chardet
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
from typing import Tuple, Optional

# Document conversion libraries
try:
    import docx
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False
    
try:
    import pdfplumber
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

try:
    from striprtf.striprtf import rtf_to_text
    HAS_RTF = True
except ImportError:
    HAS_RTF = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/step1_conversion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def detect_encoding(file_path: str) -> str:
    """Detect file encoding using chardet."""
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)  # Read first 10KB
    result = chardet.detect(raw_data)
    return result.get('encoding', 'utf-8') or 'utf-8'


def convert_txt(file_path: str) -> Optional[str]:
    """Convert plain text file to UTF-8."""
    try:
        encoding = detect_encoding(file_path)
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            return f.read()
    except Exception as e:
        logger.error(f"TXT conversion failed for {file_path}: {e}")
        return None


def convert_docx(file_path: str) -> Optional[str]:
    """Extract text from .docx files."""
    if not HAS_DOCX:
        logger.warning("python-docx not installed. Skipping .docx files.")
        return None
    try:
        doc = docx.Document(file_path)
        paragraphs = [para.text for para in doc.paragraphs]
        return '\n'.join(paragraphs)
    except Exception as e:
        logger.error(f"DOCX conversion failed for {file_path}: {e}")
        return None


def convert_pdf(file_path: str) -> Optional[str]:
    """Extract text from PDF files using pdfplumber."""
    if not HAS_PDF:
        logger.warning("pdfplumber not installed. Skipping .pdf files.")
        return None
    try:
        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return '\n'.join(text_parts)
    except Exception as e:
        logger.error(f"PDF conversion failed for {file_path}: {e}")
        return None


def convert_html(file_path: str) -> Optional[str]:
    """Extract text from HTML files."""
    if not HAS_BS4:
        logger.warning("BeautifulSoup not installed. Skipping .html files.")
        return None
    try:
        encoding = detect_encoding(file_path)
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
        # Remove script and style elements
        for script in soup(['script', 'style', 'meta', 'link']):
            script.decompose()
        return soup.get_text(separator='\n', strip=True)
    except Exception as e:
        logger.error(f"HTML conversion failed for {file_path}: {e}")
        return None


def convert_rtf(file_path: str) -> Optional[str]:
    """Extract text from RTF files."""
    if not HAS_RTF:
        logger.warning("striprtf not installed. Skipping .rtf files.")
        return None
    try:
        encoding = detect_encoding(file_path)
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            rtf_content = f.read()
        return rtf_to_text(rtf_content)
    except Exception as e:
        logger.error(f"RTF conversion failed for {file_path}: {e}")
        return None


def validate_utf8(text: str) -> bool:
    """Validate that text is proper UTF-8."""
    try:
        text.encode('utf-8').decode('utf-8')
        return True
    except (UnicodeEncodeError, UnicodeDecodeError):
        return False


def process_file(args: Tuple[str, str, str]) -> Tuple[str, bool, str]:
    """
    Process a single file: convert to UTF-8 text.
    
    Returns:
        Tuple of (filename, success, error_message)
    """
    input_path, output_dir, filename = args
    file_path = os.path.join(input_path, filename)
    ext = Path(filename).suffix.lower()
    
    # Select converter based on extension
    converters = {
        '.txt': convert_txt,
        '.docx': convert_docx,
        '.doc': convert_docx,  # May need additional handling
        '.pdf': convert_pdf,
        '.html': convert_html,
        '.htm': convert_html,
        '.rtf': convert_rtf,
    }
    
    converter = converters.get(ext)
    if not converter:
        return (filename, False, f"Unsupported format: {ext}")
    
    # Convert file
    text = converter(file_path)
    if text is None:
        return (filename, False, "Conversion returned None")
    
    # Validate UTF-8
    if not validate_utf8(text):
        return (filename, False, "UTF-8 validation failed")
    
    # Skip empty files
    if not text.strip():
        return (filename, False, "Empty content after conversion")
    
    # Write output
    output_filename = Path(filename).stem + '.txt'
    output_path = os.path.join(output_dir, output_filename)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        return (filename, True, "")
    except Exception as e:
        return (filename, False, str(e))


def get_supported_files(input_dir: str) -> list:
    """Get list of supported files in input directory."""
    supported_extensions = {'.txt', '.docx', '.doc', '.pdf', '.html', '.htm', '.rtf'}
    files = []
    
    for root, _, filenames in os.walk(input_dir):
        for filename in filenames:
            if Path(filename).suffix.lower() in supported_extensions:
                rel_path = os.path.relpath(os.path.join(root, filename), input_dir)
                files.append(rel_path)
    
    return files


def main():
    parser = argparse.ArgumentParser(
        description='Step 1: Convert document formats to UTF-8 text'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input directory containing raw documents'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output directory for converted text files'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=cpu_count(),
        help=f'Number of parallel workers (default: {cpu_count()})'
    )
    args = parser.parse_args()
    
    # Validate directories
    if not os.path.exists(args.input):
        logger.error(f"Input directory does not exist: {args.input}")
        sys.exit(1)
    
    os.makedirs(args.output, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Get files to process
    files = get_supported_files(args.input)
    logger.info(f"Found {len(files)} files to convert")
    
    if not files:
        logger.warning("No supported files found in input directory")
        return
    
    # Prepare arguments for multiprocessing
    process_args = [(args.input, args.output, f) for f in files]
    
    # Process files with multiprocessing
    success_count = 0
    failed_files = []
    
    with Pool(processes=args.workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_file, process_args),
            total=len(files),
            desc="Converting files",
            unit="file"
        ))
    
    # Collect results
    for filename, success, error in results:
        if success:
            success_count += 1
        else:
            failed_files.append((filename, error))
    
    # Log summary
    logger.info(f"\n{'='*50}")
    logger.info(f"Conversion Complete")
    logger.info(f"{'='*50}")
    logger.info(f"Total files: {len(files)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {len(failed_files)}")
    
    # Write failed files log
    if failed_files:
        with open('logs/step1_failed_files.log', 'w') as f:
            for filename, error in failed_files:
                f.write(f"{filename}\t{error}\n")
        logger.info(f"Failed files logged to: logs/step1_failed_files.log")


if __name__ == '__main__':
    main()
