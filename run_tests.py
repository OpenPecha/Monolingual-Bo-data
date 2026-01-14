#!/usr/bin/env python3
"""
Test Runner for Tibetan Data Pipeline
=====================================
Runs all tests with coverage reporting.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --step 1           # Run Step 1 tests only
    python run_tests.py --step 2           # Run Step 2 tests only
    python run_tests.py -v                 # Verbose output
    python run_tests.py --coverage         # With coverage report
"""

import os
import sys
import argparse
import subprocess


def run_tests(step=None, verbose=False, coverage=False):
    """Run pytest with specified options."""
    
    # Build pytest command
    cmd = [sys.executable, '-m', 'pytest']
    
    # Add test path
    test_dir = os.path.join(os.path.dirname(__file__), 'tests')
    
    if step:
        # Run specific step tests
        test_file = os.path.join(test_dir, f'test_step{step}_*.py')
        cmd.append(test_file)
    else:
        cmd.append(test_dir)
    
    # Add options
    if verbose:
        cmd.append('-v')
    
    if coverage:
        cmd.extend(['--cov=scripts', '--cov-report=html', '--cov-report=term'])
    
    # Add common options
    cmd.extend([
        '-x',  # Stop on first failure
        '--tb=short',  # Short traceback
        '-W', 'ignore::DeprecationWarning',
    ])
    
    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)
    
    # Run tests
    result = subprocess.run(cmd)
    
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description='Run pipeline tests')
    parser.add_argument(
        '--step', '-s',
        type=int,
        choices=[1, 2, 3, 4, 5],
        help='Run tests for specific step only'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--coverage', '-c',
        action='store_true',
        help='Run with coverage report'
    )
    parser.add_argument(
        '--integration', '-i',
        action='store_true',
        help='Run integration tests only'
    )
    
    args = parser.parse_args()
    
    # Change to project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    
    if args.integration:
        # Run integration tests
        cmd = [
            sys.executable, '-m', 'pytest',
            'tests/test_integration.py',
            '-v' if args.verbose else '',
            '-x',
            '--tb=short',
        ]
        cmd = [c for c in cmd if c]  # Remove empty strings
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    
    exit_code = run_tests(
        step=args.step,
        verbose=args.verbose,
        coverage=args.coverage
    )
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
