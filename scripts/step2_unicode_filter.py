#!/usr/bin/env python3
"""
Step 2: Unicode Filtering
=========================
Filters files to maintain content within the Tibetan Unicode block (U+0F00–U+0FFF).
Files with <5% Tibetan Unicode are discarded; others are stripped of non-Unicode strings.

Usage:
    python step2_unicode_filter.py --input /path/to/converted --output /path/to/filtered --workers 8
"""

import os
import re
import sys
import logging
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/step2_unicode_filter.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Tibetan Unicode block: U+0F00 to U+0FFF
TIBETAN_PATTERN = re.compile(r'[\u0f00-\u0fff]')

# Extended Tibetan patterns (including common punctuation)
TIBETAN_EXTENDED_PATTERN = re.compile(r'[\u0f00-\u0fff\u0020\u000a\u000d]+')

# Non-Tibetan character pattern (for stripping)
NON_TIBETAN_PATTERN = re.compile(r'[^\u0f00-\u0fff\u0020\u000a\u000d།་༌༎༏༐༑༒]+')


class UnicodeStats:
    """Container for Unicode statistics."""
    def __init__(self):
        self.total_chars = 0
        self.tibetan_chars = 0
        self.ascii_chars = 0
        self.other_chars = 0
    
    @property
    def tibetan_ratio(self) -> float:
        if self.total_chars == 0:
            return 0.0
        return self.tibetan_chars / self.total_chars
    
    def __str__(self):
        return (
            f"Total: {self.total_chars}, "
            f"Tibetan: {self.tibetan_chars} ({self.tibetan_ratio:.2%}), "
            f"ASCII: {self.ascii_chars}, "
            f"Other: {self.other_chars}"
        )


def analyze_unicode(text: str) -> UnicodeStats:
    """Analyze Unicode character distribution in text."""
    stats = UnicodeStats()
    
    for char in text:
        code_point = ord(char)
        stats.total_chars += 1
        
        if 0x0F00 <= code_point <= 0x0FFF:
            stats.tibetan_chars += 1
        elif code_point < 128:
            stats.ascii_chars += 1
        else:
            stats.other_chars += 1
    
    return stats


def filter_tibetan_content(text: str, min_tibetan_ratio: float = 0.05) -> Optional[str]:
    """
    Filter text to keep only Tibetan content.
    
    Args:
        text: Input text
        min_tibetan_ratio: Minimum ratio of Tibetan characters required (default: 5%)
    
    Returns:
        Filtered text or None if below threshold
    """
    stats = analyze_unicode(text)
    
    # Discard if less than threshold Tibetan content
    if stats.tibetan_ratio < min_tibetan_ratio:
        return None
    
    # Process line by line
    filtered_lines = []
    
    for line in text.split('\n'):
        # Calculate Tibetan ratio for this line
        line_stats = analyze_unicode(line)
        
        if line_stats.total_chars == 0:
            continue
        
        # Keep lines with significant Tibetan content
        if line_stats.tibetan_ratio >= 0.3:  # At least 30% Tibetan per line
            # Strip non-Tibetan characters while preserving structure
            clean_line = clean_tibetan_line(line)
            if clean_line.strip():
                filtered_lines.append(clean_line)
    
    if not filtered_lines:
        return None
    
    return '\n'.join(filtered_lines)


def clean_tibetan_line(line: str) -> str:
    """
    Clean a single line, preserving Tibetan text and essential whitespace.
    
    Strategy:
    1. Keep all Tibetan characters
    2. Keep spaces and newlines
    3. Remove other scripts but preserve word boundaries
    """
    result = []
    prev_was_tibetan = False
    
    for char in line:
        code_point = ord(char)
        
        # Tibetan Unicode block
        if 0x0F00 <= code_point <= 0x0FFF:
            result.append(char)
            prev_was_tibetan = True
        # Whitespace (preserve)
        elif char in ' \t':
            if prev_was_tibetan:  # Only add space after Tibetan text
                result.append(' ')
            prev_was_tibetan = False
        # Skip other characters
        else:
            if prev_was_tibetan and result and result[-1] != ' ':
                result.append(' ')
            prev_was_tibetan = False
    
    # Clean up multiple spaces
    cleaned = ''.join(result)
    cleaned = re.sub(r' +', ' ', cleaned)
    return cleaned.strip()


def validate_tibetan_bytes(content: bytes) -> bool:
    """
    Byte-level validation for Tibetan UTF-8 sequences.
    Tibetan characters in UTF-8 are typically 3 bytes: E0 BC 80 to E0 BF BF
    """
    try:
        text = content.decode('utf-8')
        tibetan_chars = len(TIBETAN_PATTERN.findall(text))
        return tibetan_chars > 0
    except UnicodeDecodeError:
        return False


def process_file(args: Tuple[str, str, str, float]) -> Tuple[str, bool, str, dict]:
    """
    Process a single file for Unicode filtering.
    
    Returns:
        Tuple of (filename, success, error_message, stats_dict)
    """
    input_dir, output_dir, filename, min_ratio = args
    input_path = os.path.join(input_dir, filename)
    
    stats = {
        'original_chars': 0,
        'filtered_chars': 0,
        'tibetan_ratio': 0.0,
        'status': 'unknown'
    }
    
    try:
        # Read file
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        stats['original_chars'] = len(content)
        
        # Analyze original content
        original_stats = analyze_unicode(content)
        stats['tibetan_ratio'] = original_stats.tibetan_ratio
        
        # Filter content
        filtered_content = filter_tibetan_content(content, min_ratio)
        
        if filtered_content is None:
            stats['status'] = 'discarded_low_tibetan'
            return (filename, False, f"Below {min_ratio:.0%} Tibetan threshold", stats)
        
        stats['filtered_chars'] = len(filtered_content)
        
        # Verify filtered content has meaningful Tibetan
        if len(filtered_content.strip()) < 10:
            stats['status'] = 'discarded_too_short'
            return (filename, False, "Filtered content too short", stats)
        
        # Write output
        output_path = os.path.join(output_dir, filename)
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else output_dir, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(filtered_content)
        
        stats['status'] = 'success'
        return (filename, True, "", stats)
        
    except Exception as e:
        stats['status'] = 'error'
        return (filename, False, str(e), stats)


def get_txt_files(input_dir: str) -> list:
    """Get all .txt files in input directory."""
    files = []
    for root, _, filenames in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith('.txt'):
                rel_path = os.path.relpath(os.path.join(root, filename), input_dir)
                files.append(rel_path)
    return files


def main():
    parser = argparse.ArgumentParser(
        description='Step 2: Filter files by Tibetan Unicode content'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input directory containing converted text files'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output directory for filtered files'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=cpu_count(),
        help=f'Number of parallel workers (default: {cpu_count()})'
    )
    parser.add_argument(
        '--min-ratio', '-r',
        type=float,
        default=0.05,
        help='Minimum Tibetan character ratio (default: 0.05 = 5%%)'
    )
    args = parser.parse_args()
    
    # Validate directories
    if not os.path.exists(args.input):
        logger.error(f"Input directory does not exist: {args.input}")
        sys.exit(1)
    
    os.makedirs(args.output, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Get files to process
    files = get_txt_files(args.input)
    logger.info(f"Found {len(files)} files to filter")
    
    if not files:
        logger.warning("No .txt files found in input directory")
        return
    
    # Prepare arguments for multiprocessing
    process_args = [(args.input, args.output, f, args.min_ratio) for f in files]
    
    # Process files with multiprocessing
    success_count = 0
    discarded_count = 0
    failed_count = 0
    total_original_chars = 0
    total_filtered_chars = 0
    
    with Pool(processes=args.workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_file, process_args),
            total=len(files),
            desc="Filtering Unicode",
            unit="file"
        ))
    
    # Collect results
    discarded_files = []
    for filename, success, error, stats in results:
        total_original_chars += stats.get('original_chars', 0)
        
        if success:
            success_count += 1
            total_filtered_chars += stats.get('filtered_chars', 0)
        elif 'discarded' in stats.get('status', ''):
            discarded_count += 1
            discarded_files.append((filename, stats['status'], stats.get('tibetan_ratio', 0)))
        else:
            failed_count += 1
            logger.error(f"Failed: {filename} - {error}")
    
    # Log summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Unicode Filtering Complete")
    logger.info(f"{'='*60}")
    logger.info(f"Total files processed: {len(files)}")
    logger.info(f"Successfully filtered: {success_count}")
    logger.info(f"Discarded (low Tibetan): {discarded_count}")
    logger.info(f"Failed (errors): {failed_count}")
    logger.info(f"{'='*60}")
    logger.info(f"Original characters: {total_original_chars:,}")
    logger.info(f"Filtered characters: {total_filtered_chars:,}")
    if total_original_chars > 0:
        retention = total_filtered_chars / total_original_chars
        logger.info(f"Character retention: {retention:.2%}")
    
    # Write discarded files log
    if discarded_files:
        with open('logs/step2_discarded_files.log', 'w') as f:
            f.write("filename\tstatus\ttibetan_ratio\n")
            for filename, status, ratio in discarded_files:
                f.write(f"{filename}\t{status}\t{ratio:.4f}\n")
        logger.info(f"Discarded files logged to: logs/step2_discarded_files.log")


if __name__ == '__main__':
    main()
