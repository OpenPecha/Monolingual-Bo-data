#!/usr/bin/env python3
"""
Step 5: OCR Quality Classification
===================================
Uses a KenLM 5-gram language model to classify sentences by perplexity.

Quality Classes:
- Class A (Highest): PPL ≤ 100
- Class B (Medium): 100 < PPL ≤ 500
- Class C (Noise): PPL > 500

KenLM Training: See docs/KENLM_TRAINING.md
Repository: https://github.com/kpu/kenlm

Usage:
    python step5_classifier.py --input /path/to/segmented --output /path/to/classified --model model.bin --workers 8
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from typing import Tuple, List, Dict, Optional
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/step5_classification.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Try to import KenLM
try:
    import kenlm
    HAS_KENLM = True
except ImportError:
    HAS_KENLM = False
    logger.warning("KenLM not installed. Install with: pip install kenlm")


class QualityClassifier:
    """
    Quality classifier using KenLM perplexity scoring.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold_a: float = 100,
        threshold_b: float = 500
    ):
        """
        Initialize the classifier.
        
        Args:
            model_path: Path to KenLM binary model file
            threshold_a: Upper perplexity bound for Class A
            threshold_b: Upper perplexity bound for Class B
        """
        self.threshold_a = threshold_a
        self.threshold_b = threshold_b
        self.model = None
        
        if model_path and HAS_KENLM:
            try:
                self.model = kenlm.Model(model_path)
                logger.info(f"Loaded KenLM model from: {model_path}")
            except Exception as e:
                logger.error(f"Failed to load KenLM model: {e}")
    
    def calculate_perplexity(self, sentence: str) -> float:
        """
        Calculate perplexity of a sentence using KenLM.
        
        Args:
            sentence: Input sentence
        
        Returns:
            Perplexity score (lower = more likely/cleaner)
        """
        if self.model is None:
            # Fallback: return default middle value
            return 250.0
        
        try:
            return self.model.perplexity(sentence)
        except Exception as e:
            logger.debug(f"Perplexity calculation failed: {e}")
            return float('inf')
    
    def classify(self, sentence: str) -> Tuple[str, float]:
        """
        Classify a sentence by quality.
        
        Args:
            sentence: Input sentence
        
        Returns:
            Tuple of (class_label, perplexity)
        """
        ppl = self.calculate_perplexity(sentence)
        
        if ppl <= self.threshold_a:
            return 'A', ppl
        elif ppl <= self.threshold_b:
            return 'B', ppl
        else:
            return 'C', ppl


# Global classifier for multiprocessing
_classifier = None
_model_path = None
_threshold_a = 100  # Default value
_threshold_b = 500  # Default value


def init_worker():
    """Initialize worker process with classifier."""
    global _classifier, _model_path, _threshold_a, _threshold_b
    _classifier = QualityClassifier(
        model_path=_model_path,
        threshold_a=_threshold_a if _threshold_a is not None else 100,
        threshold_b=_threshold_b if _threshold_b is not None else 500
    )


def process_file(args: Tuple[str, str, str]) -> Tuple[str, bool, str, dict]:
    """
    Process a single file: classify each sentence by quality.
    
    Returns:
        Tuple of (filename, success, error_message, stats_dict)
    """
    global _classifier
    
    input_dir, output_dir, filename = args
    input_path = os.path.join(input_dir, filename)
    
    stats = {
        'total_sentences': 0,
        'class_A': 0,
        'class_B': 0,
        'class_C': 0,
        'avg_perplexity': 0.0
    }
    
    try:
        # Read file (one sentence per line from Step 4)
        with open(input_path, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
        
        stats['total_sentences'] = len(sentences)
        
        if not sentences:
            return (filename, False, "No sentences in file", stats)
        
        # Classify each sentence
        classified = {'A': [], 'B': [], 'C': []}
        total_ppl = 0.0
        
        for sent in sentences:
            class_label, ppl = _classifier.classify(sent)
            classified[class_label].append((sent, ppl))
            total_ppl += ppl if ppl != float('inf') else 1000
            stats[f'class_{class_label}'] += 1
        
        stats['avg_perplexity'] = total_ppl / len(sentences) if sentences else 0
        
        # Write to output directories
        base_filename = Path(filename).stem
        
        for class_label, sent_list in classified.items():
            if not sent_list:
                continue
            
            # Create output path
            output_path = os.path.join(output_dir, class_label, f"{base_filename}.txt")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Write sentences (without perplexity scores)
            with open(output_path, 'w', encoding='utf-8') as f:
                for sent, _ in sent_list:
                    f.write(sent + '\n')
        
        return (filename, True, "", stats)
        
    except Exception as e:
        return (filename, False, str(e), stats)


def process_file_with_scores(args: Tuple[str, str, str]) -> Tuple[str, bool, str, dict]:
    """
    Process a single file and include perplexity scores in output.
    
    Returns:
        Tuple of (filename, success, error_message, stats_dict)
    """
    global _classifier
    
    input_dir, output_dir, filename = args
    input_path = os.path.join(input_dir, filename)
    
    stats = {
        'total_sentences': 0,
        'class_A': 0,
        'class_B': 0,
        'class_C': 0,
        'avg_perplexity': 0.0
    }
    
    try:
        # Read file
        with open(input_path, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
        
        stats['total_sentences'] = len(sentences)
        
        if not sentences:
            return (filename, False, "No sentences in file", stats)
        
        # Classify each sentence
        classified = {'A': [], 'B': [], 'C': []}
        total_ppl = 0.0
        
        for sent in sentences:
            class_label, ppl = _classifier.classify(sent)
            classified[class_label].append((sent, ppl))
            total_ppl += ppl if ppl != float('inf') else 1000
            stats[f'class_{class_label}'] += 1
        
        stats['avg_perplexity'] = total_ppl / len(sentences) if sentences else 0
        
        # Write to output directories with scores
        base_filename = Path(filename).stem
        
        for class_label, sent_list in classified.items():
            if not sent_list:
                continue
            
            output_path = os.path.join(output_dir, class_label, f"{base_filename}.tsv")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Write sentences with perplexity scores
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("sentence\tperplexity\n")
                for sent, ppl in sent_list:
                    f.write(f"{sent}\t{ppl:.2f}\n")
        
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
    global _model_path, _threshold_a, _threshold_b
    
    parser = argparse.ArgumentParser(
        description='Step 5: Classify sentences by quality using KenLM perplexity'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input directory containing segmented text files'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output directory for classified files'
    )
    parser.add_argument(
        '--model', '-m',
        required=False,
        help='Path to KenLM binary model file (.bin)'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=cpu_count(),
        help=f'Number of parallel workers (default: {cpu_count()})'
    )
    parser.add_argument(
        '--threshold-a', '-ta',
        type=float,
        default=100,
        help='Perplexity threshold for Class A (default: 100)'
    )
    parser.add_argument(
        '--threshold-b', '-tb',
        type=float,
        default=500,
        help='Perplexity threshold for Class B (default: 500)'
    )
    parser.add_argument(
        '--include-scores',
        action='store_true',
        help='Include perplexity scores in output (writes .tsv files)'
    )
    args = parser.parse_args()
    
    # Set global variables for worker initialization
    _model_path = args.model
    _threshold_a = args.threshold_a
    _threshold_b = args.threshold_b
    
    # Validate directories
    if not os.path.exists(args.input):
        logger.error(f"Input directory does not exist: {args.input}")
        sys.exit(1)
    
    # Create output directories for each class
    for class_label in ['A', 'B', 'C']:
        os.makedirs(os.path.join(args.output, class_label), exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Check KenLM availability
    if not HAS_KENLM:
        logger.warning("KenLM not installed. Using fallback classification (all sentences -> Class B).")
        logger.warning("Install KenLM for actual perplexity-based classification: pip install kenlm")
        logger.warning("See docs/KENLM_TRAINING.md for model training instructions.")
    
    if not args.model:
        if HAS_KENLM:
            logger.warning("No model specified. Using default perplexity (250) for all sentences.")
            logger.warning("Specify a model with --model for actual classification.")
    elif args.model and not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        sys.exit(1)
    
    # Get files to process
    files = get_txt_files(args.input)
    logger.info(f"Found {len(files)} files to classify")
    
    if not files:
        logger.warning("No .txt files found in input directory")
        return
    
    # Prepare arguments for multiprocessing
    process_args = [(args.input, args.output, f) for f in files]
    
    # Select processor function
    processor = process_file_with_scores if args.include_scores else process_file
    
    # Process files with multiprocessing
    total_stats = {
        'total_sentences': 0,
        'class_A': 0,
        'class_B': 0,
        'class_C': 0,
        'total_perplexity': 0.0
    }
    
    success_count = 0
    failed_files = []
    
    with Pool(processes=args.workers, initializer=init_worker) as pool:
        results = list(tqdm(
            pool.imap_unordered(processor, process_args),
            total=len(files),
            desc="Classifying",
            unit="file"
        ))
    
    # Collect results
    for filename, success, error, stats in results:
        if success:
            success_count += 1
            total_stats['total_sentences'] += stats['total_sentences']
            total_stats['class_A'] += stats['class_A']
            total_stats['class_B'] += stats['class_B']
            total_stats['class_C'] += stats['class_C']
            total_stats['total_perplexity'] += stats['avg_perplexity'] * stats['total_sentences']
        else:
            failed_files.append((filename, error))
    
    # Calculate averages
    avg_ppl = (
        total_stats['total_perplexity'] / total_stats['total_sentences']
        if total_stats['total_sentences'] > 0 else 0
    )
    
    # Log summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Quality Classification Complete")
    logger.info(f"{'='*60}")
    logger.info(f"Total files processed: {len(files)}")
    logger.info(f"Successfully classified: {success_count}")
    logger.info(f"Failed: {len(failed_files)}")
    logger.info(f"{'='*60}")
    logger.info(f"Total sentences: {total_stats['total_sentences']:,}")
    logger.info(f"{'='*60}")
    logger.info(f"Classification Distribution:")
    
    if total_stats['total_sentences'] > 0:
        pct_a = total_stats['class_A'] / total_stats['total_sentences'] * 100
        pct_b = total_stats['class_B'] / total_stats['total_sentences'] * 100
        pct_c = total_stats['class_C'] / total_stats['total_sentences'] * 100
        
        logger.info(f"  Class A (PPL ≤ {args.threshold_a}): {total_stats['class_A']:,} ({pct_a:.1f}%)")
        logger.info(f"  Class B ({args.threshold_a} < PPL ≤ {args.threshold_b}): {total_stats['class_B']:,} ({pct_b:.1f}%)")
        logger.info(f"  Class C (PPL > {args.threshold_b}): {total_stats['class_C']:,} ({pct_c:.1f}%)")
    
    logger.info(f"{'='*60}")
    logger.info(f"Average perplexity: {avg_ppl:.2f}")
    
    # Write summary to file
    summary_path = os.path.join(args.output, 'classification_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Tibetan Text Quality Classification Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {args.model or 'None (default scoring)'}\n")
        f.write(f"Threshold A: PPL ≤ {args.threshold_a}\n")
        f.write(f"Threshold B: {args.threshold_a} < PPL ≤ {args.threshold_b}\n\n")
        f.write(f"Total sentences: {total_stats['total_sentences']:,}\n")
        f.write(f"Class A: {total_stats['class_A']:,}\n")
        f.write(f"Class B: {total_stats['class_B']:,}\n")
        f.write(f"Class C: {total_stats['class_C']:,}\n")
        f.write(f"Average perplexity: {avg_ppl:.2f}\n")
    
    logger.info(f"Summary written to: {summary_path}")
    
    # Write failed files log
    if failed_files:
        with open('logs/step5_failed_files.log', 'w') as f:
            for filename, error in failed_files:
                f.write(f"{filename}\t{error}\n")
        logger.info(f"Failed files logged to: logs/step5_failed_files.log")


if __name__ == '__main__':
    main()
