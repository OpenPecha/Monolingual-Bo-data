#!/usr/bin/env python3
"""
Step 4: Linguistic Filtering & Segmentation
============================================
Uses the Botok library for Tibetan text tokenization and sentence segmentation.
Applies filters:
- Syllable Filter: Sentences with <4 syllables are removed
- Language Filter: Sentences must contain >80% Tibetan script characters

Usage:
    python step4_segmenter.py --input /path/to/deduplicated --output /path/to/segmented --workers 8
"""

import os
import re
import sys
import logging
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from typing import Tuple, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/step4_segmentation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Tibetan Unicode patterns
TIBETAN_CHAR_PATTERN = re.compile(r'[\u0f00-\u0fff]')
TSEK = '་'
SHAD = '།'

# Try to import Botok
try:
    from botok import WordTokenizer
    from botok.tokenizers.sentencetokenizer import SentenceTokenizer
    HAS_BOTOK = True
except ImportError:
    HAS_BOTOK = False
    logger.warning("Botok not installed. Using fallback segmentation.")


class TibetanSegmenter:
    """
    Tibetan text segmenter using Botok or fallback methods.
    """
    
    def __init__(self, use_botok: bool = True):
        self.use_botok = use_botok and HAS_BOTOK
        
        if self.use_botok:
            try:
                self.word_tokenizer = WordTokenizer()
                self.sentence_tokenizer = SentenceTokenizer()
                logger.info("Initialized Botok tokenizers")
            except Exception as e:
                logger.warning(f"Failed to initialize Botok: {e}. Using fallback.")
                self.use_botok = False
    
    def segment_sentences(self, text: str) -> List[str]:
        """
        Segment text into sentences.
        
        Args:
            text: Input Tibetan text
        
        Returns:
            List of sentences
        """
        if self.use_botok:
            return self._segment_with_botok(text)
        else:
            return self._segment_fallback(text)
    
    def _segment_with_botok(self, text: str) -> List[str]:
        """Segment using Botok library."""
        try:
            # Tokenize words
            tokens = self.word_tokenizer.tokenize(text, split_affixes=False)
            
            # Reconstruct sentences
            sentences = self.sentence_tokenizer.reconstruct_sents(tokens)
            
            return [sent.strip() for sent in sentences if sent.strip()]
        except Exception as e:
            logger.debug(f"Botok segmentation failed, using fallback: {e}")
            return self._segment_fallback(text)
    
    def _segment_fallback(self, text: str) -> List[str]:
        """
        Fallback sentence segmentation based on Tibetan punctuation.
        Tibetan sentences typically end with shad (།) or double shad (།།).
        """
        # Split by sentence-ending punctuation
        # Single shad (།), double shad (།།), or other sentence markers
        sentences = re.split(r'།།?', text)
        
        # Clean and filter
        result = []
        for sent in sentences:
            sent = sent.strip()
            if sent:
                # Add shad back for completeness
                result.append(sent + '།')
        
        return result
    
    def count_syllables(self, text: str) -> int:
        """
        Count syllables in Tibetan text.
        Syllables are separated by tsek (་).
        """
        # Split by tsek
        syllables = text.split(TSEK)
        # Filter empty strings and count
        return len([s for s in syllables if s.strip()])
    
    def tibetan_ratio(self, text: str) -> float:
        """Calculate ratio of Tibetan characters in text."""
        if not text:
            return 0.0
        
        tibetan_chars = len(TIBETAN_CHAR_PATTERN.findall(text))
        total_chars = len(text.replace(' ', '').replace('\n', ''))
        
        if total_chars == 0:
            return 0.0
        
        return tibetan_chars / total_chars


def filter_sentence(
    sentence: str,
    segmenter: TibetanSegmenter,
    min_syllables: int = 4,
    min_tibetan_ratio: float = 0.8
) -> Tuple[bool, str]:
    """
    Apply filters to a sentence.
    
    Args:
        sentence: Input sentence
        segmenter: TibetanSegmenter instance
        min_syllables: Minimum syllable count
        min_tibetan_ratio: Minimum Tibetan character ratio
    
    Returns:
        Tuple of (passes_filter, reason)
    """
    # Check syllable count
    syllable_count = segmenter.count_syllables(sentence)
    if syllable_count < min_syllables:
        return False, f"too_few_syllables ({syllable_count})"
    
    # Check Tibetan ratio
    ratio = segmenter.tibetan_ratio(sentence)
    if ratio < min_tibetan_ratio:
        return False, f"low_tibetan_ratio ({ratio:.2f})"
    
    return True, "passed"


# Global segmenter for multiprocessing
_segmenter = None


def init_worker():
    """Initialize worker process with segmenter."""
    global _segmenter
    _segmenter = TibetanSegmenter(use_botok=HAS_BOTOK)


def process_file(args: Tuple[str, str, str, int, float]) -> Tuple[str, bool, str, dict]:
    """
    Process a single file: segment and filter sentences.
    
    Returns:
        Tuple of (filename, success, error_message, stats_dict)
    """
    global _segmenter
    
    input_dir, output_dir, filename, min_syllables, min_tibetan_ratio = args
    input_path = os.path.join(input_dir, filename)
    
    stats = {
        'total_sentences': 0,
        'passed_sentences': 0,
        'filtered_syllables': 0,
        'filtered_ratio': 0,
        'original_chars': 0,
        'output_chars': 0
    }
    
    try:
        # Read file
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        stats['original_chars'] = len(content)
        
        if not content.strip():
            return (filename, False, "Empty file", stats)
        
        # Segment into sentences
        sentences = _segmenter.segment_sentences(content)
        stats['total_sentences'] = len(sentences)
        
        # Filter sentences
        passed_sentences = []
        
        for sent in sentences:
            passes, reason = filter_sentence(
                sent, _segmenter, min_syllables, min_tibetan_ratio
            )
            
            if passes:
                passed_sentences.append(sent)
            elif 'syllables' in reason:
                stats['filtered_syllables'] += 1
            elif 'ratio' in reason:
                stats['filtered_ratio'] += 1
        
        stats['passed_sentences'] = len(passed_sentences)
        
        if not passed_sentences:
            return (filename, False, "No sentences passed filters", stats)
        
        # Write output (one sentence per line)
        output_content = '\n'.join(passed_sentences)
        stats['output_chars'] = len(output_content)
        
        output_path = os.path.join(output_dir, filename)
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else output_dir, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_content)
        
        return (filename, True, "", stats)
        
    except Exception as e:
        return (filename, False, str(e), stats)


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
        description='Step 4: Segment and filter Tibetan text'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input directory containing deduplicated text files'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output directory for segmented files'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=cpu_count(),
        help=f'Number of parallel workers (default: {cpu_count()})'
    )
    parser.add_argument(
        '--min-syllables', '-s',
        type=int,
        default=4,
        help='Minimum syllables per sentence (default: 4)'
    )
    parser.add_argument(
        '--min-tibetan-ratio', '-r',
        type=float,
        default=0.8,
        help='Minimum Tibetan character ratio (default: 0.8 = 80%%)'
    )
    args = parser.parse_args()
    
    # Validate directories
    if not os.path.exists(args.input):
        logger.error(f"Input directory does not exist: {args.input}")
        sys.exit(1)
    
    os.makedirs(args.output, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Check Botok availability
    if HAS_BOTOK:
        logger.info("Using Botok for tokenization")
    else:
        logger.warning("Botok not available. Using fallback segmentation.")
        logger.warning("Install Botok for better results: pip install botok")
    
    # Get files to process
    files = get_txt_files(args.input)
    logger.info(f"Found {len(files)} files to segment")
    
    if not files:
        logger.warning("No .txt files found in input directory")
        return
    
    # Prepare arguments for multiprocessing
    process_args = [
        (args.input, args.output, f, args.min_syllables, args.min_tibetan_ratio)
        for f in files
    ]
    
    # Process files with multiprocessing
    total_stats = {
        'total_sentences': 0,
        'passed_sentences': 0,
        'filtered_syllables': 0,
        'filtered_ratio': 0,
        'original_chars': 0,
        'output_chars': 0
    }
    
    success_count = 0
    failed_files = []
    
    with Pool(processes=args.workers, initializer=init_worker) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_file, process_args),
            total=len(files),
            desc="Segmenting",
            unit="file"
        ))
    
    # Collect results
    for filename, success, error, stats in results:
        for key in total_stats:
            total_stats[key] += stats.get(key, 0)
        
        if success:
            success_count += 1
        else:
            failed_files.append((filename, error))
    
    # Log summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Segmentation Complete")
    logger.info(f"{'='*60}")
    logger.info(f"Total files processed: {len(files)}")
    logger.info(f"Successfully segmented: {success_count}")
    logger.info(f"Failed: {len(failed_files)}")
    logger.info(f"{'='*60}")
    logger.info(f"Total sentences found: {total_stats['total_sentences']:,}")
    logger.info(f"Sentences passed: {total_stats['passed_sentences']:,}")
    logger.info(f"Filtered (syllables): {total_stats['filtered_syllables']:,}")
    logger.info(f"Filtered (ratio): {total_stats['filtered_ratio']:,}")
    
    if total_stats['total_sentences'] > 0:
        pass_rate = total_stats['passed_sentences'] / total_stats['total_sentences']
        logger.info(f"Sentence pass rate: {pass_rate:.2%}")
    
    logger.info(f"{'='*60}")
    logger.info(f"Original characters: {total_stats['original_chars']:,}")
    logger.info(f"Output characters: {total_stats['output_chars']:,}")
    
    if total_stats['original_chars'] > 0:
        retention = total_stats['output_chars'] / total_stats['original_chars']
        logger.info(f"Character retention: {retention:.2%}")
    
    # Write failed files log
    if failed_files:
        with open('logs/step4_failed_files.log', 'w') as f:
            for filename, error in failed_files:
                f.write(f"{filename}\t{error}\n")
        logger.info(f"Failed files logged to: logs/step4_failed_files.log")


if __name__ == '__main__':
    main()
