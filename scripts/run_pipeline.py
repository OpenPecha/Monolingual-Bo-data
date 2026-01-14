#!/usr/bin/env python3
"""
Tibetan Data Cleaning Pipeline - Main Orchestrator
===================================================
Runs all five pipeline steps in sequence with progress tracking.

Pipeline Steps:
1. Format Conversion → UTF-8 text
2. Unicode Filtering → Tibetan content only
3. Deduplication → Remove duplicates
4. Segmentation → Sentence-level filtering
5. Classification → Quality scoring

Usage:
    python run_pipeline.py --input /path/to/raw --output /path/to/output --model model.bin
    
    # Run specific steps only
    python run_pipeline.py --input /path/to/raw --output /path/to/output --steps 1,2,3
"""

import os
import sys
import time
import logging
import argparse
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PipelineStep:
    """Represents a single pipeline step."""
    
    def __init__(
        self,
        step_num: int,
        name: str,
        script: str,
        input_dir: str,
        output_dir: str,
        extra_args: Optional[List[str]] = None
    ):
        self.step_num = step_num
        self.name = name
        self.script = script
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.extra_args = extra_args or []
        self.duration = None
        self.success = False
    
    def __str__(self):
        return f"Step {self.step_num}: {self.name}"


def count_files(directory: str) -> int:
    """Count .txt files in a directory."""
    count = 0
    if os.path.exists(directory):
        for root, _, files in os.walk(directory):
            count += sum(1 for f in files if f.endswith('.txt'))
    return count


def get_dir_size(directory: str) -> int:
    """Get total size of directory in bytes."""
    total = 0
    if os.path.exists(directory):
        for root, _, files in os.walk(directory):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
    return total


def format_size(size_bytes: int) -> str:
    """Format size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    return str(timedelta(seconds=int(seconds)))


def run_step(step: PipelineStep, workers: int, dry_run: bool = False) -> bool:
    """
    Run a single pipeline step.
    
    Args:
        step: PipelineStep to run
        workers: Number of parallel workers
        dry_run: If True, only print command without executing
    
    Returns:
        True if successful, False otherwise
    """
    script_path = os.path.join(os.path.dirname(__file__), step.script)
    
    cmd = [
        sys.executable,
        script_path,
        '--input', step.input_dir,
        '--output', step.output_dir,
        '--workers', str(workers)
    ] + step.extra_args
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Running {step}")
    logger.info(f"{'='*60}")
    logger.info(f"Input: {step.input_dir}")
    logger.info(f"Output: {step.output_dir}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    if dry_run:
        logger.info("[DRY RUN] Skipping execution")
        return True
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True
        )
        
        step.duration = time.time() - start_time
        step.success = result.returncode == 0
        
        if step.success:
            logger.info(f"✓ {step} completed in {format_duration(step.duration)}")
        else:
            logger.error(f"✗ {step} failed with exit code {result.returncode}")
        
        return step.success
        
    except Exception as e:
        step.duration = time.time() - start_time
        step.success = False
        logger.error(f"✗ {step} failed with error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Tibetan Data Cleaning Pipeline Orchestrator'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input directory containing raw documents'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Base output directory for all pipeline stages'
    )
    parser.add_argument(
        '--model', '-m',
        default=None,
        help='Path to KenLM model for Step 5 classification'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=os.cpu_count(),
        help=f'Number of parallel workers (default: {os.cpu_count()})'
    )
    parser.add_argument(
        '--steps',
        type=str,
        default='1,2,3,4,5',
        help='Comma-separated list of steps to run (default: 1,2,3,4,5)'
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.85,
        help='Deduplication threshold (default: 0.85)'
    )
    parser.add_argument(
        '--threshold-a',
        type=float,
        default=100,
        help='Perplexity threshold for Class A (default: 100)'
    )
    parser.add_argument(
        '--threshold-b',
        type=float,
        default=500,
        help='Perplexity threshold for Class B (default: 500)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print commands without executing'
    )
    args = parser.parse_args()
    
    # Parse steps to run
    steps_to_run = set(int(s.strip()) for s in args.steps.split(','))
    
    # Create output directories
    base_output = args.output
    dirs = {
        'converted': os.path.join(base_output, 'converted'),
        'filtered': os.path.join(base_output, 'filtered'),
        'deduplicated': os.path.join(base_output, 'deduplicated'),
        'segmented': os.path.join(base_output, 'segmented'),
        'classified': os.path.join(base_output, 'classified'),
    }
    
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Define pipeline steps
    pipeline_steps = [
        PipelineStep(
            step_num=1,
            name="Format Conversion",
            script="step1_format_converter.py",
            input_dir=args.input,
            output_dir=dirs['converted']
        ),
        PipelineStep(
            step_num=2,
            name="Unicode Filtering",
            script="step2_unicode_filter.py",
            input_dir=dirs['converted'],
            output_dir=dirs['filtered'],
            extra_args=['--min-ratio', '0.05']
        ),
        PipelineStep(
            step_num=3,
            name="Deduplication",
            script="step3_deduplicator.py",
            input_dir=dirs['filtered'],
            output_dir=dirs['deduplicated'],
            extra_args=['--threshold', str(args.threshold)]
        ),
        PipelineStep(
            step_num=4,
            name="Segmentation",
            script="step4_segmenter.py",
            input_dir=dirs['deduplicated'],
            output_dir=dirs['segmented'],
            extra_args=['--min-syllables', '4', '--min-tibetan-ratio', '0.8']
        ),
        PipelineStep(
            step_num=5,
            name="Classification",
            script="step5_classifier.py",
            input_dir=dirs['segmented'],
            output_dir=dirs['classified'],
            extra_args=[
                '--threshold-a', str(args.threshold_a),
                '--threshold-b', str(args.threshold_b)
            ] + (['--model', args.model] if args.model else [])
        ),
    ]
    
    # Print pipeline overview
    logger.info("\n" + "="*60)
    logger.info("TIBETAN DATA CLEANING PIPELINE")
    logger.info("="*60)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Input directory: {args.input}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Steps to run: {sorted(steps_to_run)}")
    if args.model:
        logger.info(f"KenLM model: {args.model}")
    
    # Input statistics
    input_files = count_files(args.input)
    input_size = get_dir_size(args.input)
    logger.info(f"\nInput statistics:")
    logger.info(f"  Files: {input_files:,}")
    logger.info(f"  Size: {format_size(input_size)}")
    
    # Run pipeline
    total_start = time.time()
    successful_steps = []
    failed_steps = []
    
    for step in pipeline_steps:
        if step.step_num not in steps_to_run:
            logger.info(f"\nSkipping {step}")
            continue
        
        success = run_step(step, args.workers, args.dry_run)
        
        if success:
            successful_steps.append(step)
        else:
            failed_steps.append(step)
            if not args.dry_run:
                logger.error(f"Pipeline stopped due to failure at {step}")
                break
    
    total_duration = time.time() - total_start
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("PIPELINE SUMMARY")
    logger.info("="*60)
    logger.info(f"Total duration: {format_duration(total_duration)}")
    logger.info(f"Successful steps: {len(successful_steps)}")
    logger.info(f"Failed steps: {len(failed_steps)}")
    
    if not args.dry_run:
        # Output statistics
        logger.info("\nOutput statistics by stage:")
        for name, path in dirs.items():
            files = count_files(path)
            size = get_dir_size(path)
            if files > 0:
                logger.info(f"  {name}: {files:,} files, {format_size(size)}")
        
        # Classification breakdown if Step 5 ran
        if 5 in steps_to_run and os.path.exists(dirs['classified']):
            logger.info("\nClassification breakdown:")
            for cls in ['A', 'B', 'C']:
                cls_path = os.path.join(dirs['classified'], cls)
                files = count_files(cls_path)
                size = get_dir_size(cls_path)
                logger.info(f"  Class {cls}: {files:,} files, {format_size(size)}")
    
    # Step durations
    logger.info("\nStep durations:")
    for step in pipeline_steps:
        if step.duration is not None:
            status = "✓" if step.success else "✗"
            logger.info(f"  {status} {step}: {format_duration(step.duration)}")
    
    # Exit with appropriate code
    if failed_steps:
        sys.exit(1)
    else:
        logger.info("\n✓ Pipeline completed successfully!")
        sys.exit(0)


if __name__ == '__main__':
    main()
