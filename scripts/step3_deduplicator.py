#!/usr/bin/env python3
"""
Step 3: Document De-duplication
===============================
Uses MinHash Locality Sensitive Hashing (LSH) for efficient near-duplicate detection.
Shingling is performed at the syllable level (splitting by Tibetan tsek ་).

Parameters:
- Threshold: τ = 0.85 (Jaccard similarity)
- Permutations: n = 128

Usage:
    python step3_deduplicator.py --input /path/to/filtered --output /path/to/deduplicated --workers 8
"""

import os
import sys
import logging
import argparse
import pickle
import hashlib
from pathlib import Path
from multiprocessing import Pool, cpu_count, Manager
from tqdm import tqdm
from typing import Tuple, List, Set, Optional
from datasketch import MinHash, MinHashLSH
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/step3_deduplication.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Tibetan syllable separator (tsek)
TSEK = '་'
SHAD = '།'


def get_tibetan_shingles(text: str, n: int = 3) -> List[str]:
    """
    Generate n-gram shingles based on Tibetan syllables.
    
    Args:
        text: Input Tibetan text
        n: Size of shingle (number of syllables)
    
    Returns:
        List of shingle strings
    """
    # Split by tsek (syllable separator)
    # Also handle shad (sentence separator) as boundary
    text = text.replace(SHAD, f' {SHAD} ')
    syllables = [s.strip() for s in text.split(TSEK) if s.strip()]
    
    if len(syllables) < n:
        return [text]  # Return whole text as single shingle
    
    shingles = []
    for i in range(len(syllables) - n + 1):
        shingle = TSEK.join(syllables[i:i+n])
        shingles.append(shingle)
    
    return shingles


def compute_minhash(text: str, num_perm: int = 128) -> MinHash:
    """
    Compute MinHash signature for a document.
    
    Args:
        text: Input text
        num_perm: Number of permutations for MinHash
    
    Returns:
        MinHash object
    """
    m = MinHash(num_perm=num_perm)
    
    # Get syllable-based shingles
    shingles = get_tibetan_shingles(text, n=3)
    
    for shingle in shingles:
        m.update(shingle.encode('utf-8'))
    
    return m


def compute_file_hash(filepath: str) -> str:
    """Compute MD5 hash of file for exact duplicate detection."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_file_content(filepath: str) -> Tuple[str, int]:
    """Read file content and return with size."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    return content, len(content)


def process_file_minhash(args: Tuple[str, str, int]) -> Tuple[str, Optional[MinHash], str, int]:
    """
    Compute MinHash for a single file.
    
    Returns:
        Tuple of (filename, minhash, content_hash, file_size)
    """
    input_dir, filename, num_perm = args
    filepath = os.path.join(input_dir, filename)
    
    try:
        content, size = read_file_content(filepath)
        
        if not content.strip():
            return (filename, None, "", 0)
        
        # Compute content hash for exact duplicate detection
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        
        # Compute MinHash
        minhash = compute_minhash(content, num_perm)
        
        return (filename, minhash, content_hash, size)
        
    except Exception as e:
        logger.error(f"Error processing {filename}: {e}")
        return (filename, None, "", 0)


def copy_file(src: str, dst: str):
    """Copy file from source to destination."""
    os.makedirs(os.path.dirname(dst) if os.path.dirname(dst) else os.path.dirname(dst) or '.', exist_ok=True)
    with open(src, 'r', encoding='utf-8') as f:
        content = f.read()
    with open(dst, 'w', encoding='utf-8') as f:
        f.write(content)


def get_txt_files(input_dir: str) -> List[str]:
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
        description='Step 3: Deduplicate documents using MinHash LSH'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input directory containing filtered text files'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output directory for deduplicated files'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=cpu_count(),
        help=f'Number of parallel workers (default: {cpu_count()})'
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.85,
        help='Jaccard similarity threshold for deduplication (default: 0.85)'
    )
    parser.add_argument(
        '--num-perm', '-n',
        type=int,
        default=128,
        help='Number of MinHash permutations (default: 128)'
    )
    parser.add_argument(
        '--save-index',
        type=str,
        default=None,
        help='Path to save LSH index for future use'
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
    logger.info(f"Found {len(files)} files to deduplicate")
    
    if not files:
        logger.warning("No .txt files found in input directory")
        return
    
    # Phase 1: Compute MinHash signatures in parallel
    logger.info("Phase 1: Computing MinHash signatures...")
    
    process_args = [(args.input, f, args.num_perm) for f in files]
    
    file_signatures = {}
    content_hashes = {}
    
    with Pool(processes=args.workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_file_minhash, process_args),
            total=len(files),
            desc="Computing MinHash",
            unit="file"
        ))
    
    # Collect results
    for filename, minhash, content_hash, size in results:
        if minhash is not None:
            file_signatures[filename] = (minhash, content_hash, size)
    
    logger.info(f"Computed signatures for {len(file_signatures)} files")
    
    # Phase 2: Exact duplicate detection
    logger.info("Phase 2: Detecting exact duplicates...")
    
    hash_to_files = {}
    for filename, (minhash, content_hash, size) in file_signatures.items():
        if content_hash not in hash_to_files:
            hash_to_files[content_hash] = []
        hash_to_files[content_hash].append((filename, size))
    
    exact_duplicates = 0
    exact_duplicate_files = set()
    
    for content_hash, file_list in hash_to_files.items():
        if len(file_list) > 1:
            # Sort by size (keep largest) then by name (deterministic)
            file_list.sort(key=lambda x: (-x[1], x[0]))
            # Mark all but first as duplicates
            for filename, _ in file_list[1:]:
                exact_duplicate_files.add(filename)
                exact_duplicates += 1
    
    logger.info(f"Found {exact_duplicates} exact duplicates")
    
    # Phase 3: Near-duplicate detection using LSH
    logger.info("Phase 3: Building LSH index and detecting near-duplicates...")
    
    lsh = MinHashLSH(threshold=args.threshold, num_perm=args.num_perm)
    
    # Sort files by size (process larger files first to keep them as "originals")
    sorted_files = sorted(
        [(f, data) for f, data in file_signatures.items() if f not in exact_duplicate_files],
        key=lambda x: -x[1][2]  # Sort by size descending
    )
    
    near_duplicates = 0
    near_duplicate_files = set()
    duplicate_clusters = []
    
    for filename, (minhash, content_hash, size) in tqdm(sorted_files, desc="LSH Indexing", unit="file"):
        # Check for near-duplicates
        similar = lsh.query(minhash)
        
        if similar:
            near_duplicate_files.add(filename)
            near_duplicates += 1
            duplicate_clusters.append((filename, list(similar)))
        else:
            # Add to index
            lsh.insert(filename, minhash)
    
    logger.info(f"Found {near_duplicates} near-duplicates")
    
    # Phase 4: Copy unique files to output
    logger.info("Phase 4: Copying unique files to output...")
    
    all_duplicates = exact_duplicate_files | near_duplicate_files
    unique_files = [f for f in files if f not in all_duplicates and f in file_signatures]
    
    for filename in tqdm(unique_files, desc="Copying files", unit="file"):
        src = os.path.join(args.input, filename)
        dst = os.path.join(args.output, filename)
        os.makedirs(os.path.dirname(dst) if os.path.dirname(dst) else args.output, exist_ok=True)
        copy_file(src, dst)
    
    # Save LSH index if requested
    if args.save_index:
        with open(args.save_index, 'wb') as f:
            pickle.dump(lsh, f)
        logger.info(f"Saved LSH index to: {args.save_index}")
    
    # Log summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Deduplication Complete")
    logger.info(f"{'='*60}")
    logger.info(f"Total files processed: {len(files)}")
    logger.info(f"Files with valid signatures: {len(file_signatures)}")
    logger.info(f"Exact duplicates removed: {exact_duplicates}")
    logger.info(f"Near-duplicates removed: {near_duplicates}")
    logger.info(f"Unique files retained: {len(unique_files)}")
    logger.info(f"Deduplication rate: {(exact_duplicates + near_duplicates) / len(files) * 100:.2f}%")
    
    # Write duplicate logs
    with open('logs/step3_duplicates.log', 'w') as f:
        f.write("=== Exact Duplicates ===\n")
        for content_hash, file_list in hash_to_files.items():
            if len(file_list) > 1:
                f.write(f"\nHash: {content_hash}\n")
                f.write(f"  Kept: {file_list[0][0]}\n")
                for filename, _ in file_list[1:]:
                    f.write(f"  Duplicate: {filename}\n")
        
        f.write("\n=== Near Duplicates ===\n")
        for duplicate, originals in duplicate_clusters:
            f.write(f"\nDuplicate: {duplicate}\n")
            f.write(f"  Similar to: {originals}\n")
    
    logger.info(f"Duplicate details logged to: logs/step3_duplicates.log")


if __name__ == '__main__':
    main()
