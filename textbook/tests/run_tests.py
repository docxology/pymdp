#!/usr/bin/env python3
"""
PyMDP Textbook Test Runner
=========================

Comprehensive test runner for all PyMDP textbook tests.

Usage:
    python run_tests.py                 # Run all tests
    python run_tests.py --fast          # Run only fast tests
    python run_tests.py --coverage      # Run with coverage report
    python run_tests.py --verbose       # Verbose output
    python run_tests.py test_core.py    # Run specific test file
"""

import sys
import subprocess
import argparse
from pathlib import Path


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description='Run PyMDP textbook tests')
    parser.add_argument('tests', nargs='*', help='Specific test files to run')
    parser.add_argument('--fast', action='store_true', help='Run only fast tests')
    parser.add_argument('--coverage', action='store_true', help='Generate coverage report')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--html-cov', action='store_true', help='Generate HTML coverage report')
    parser.add_argument('--parallel', '-j', type=int, help='Number of parallel processes')
    
    args = parser.parse_args()
    
    # Build pytest command
    cmd = [sys.executable, '-m', 'pytest']
    
    # Add test directory or specific tests
    test_dir = Path(__file__).parent
    if args.tests:
        # Run specific test files
        cmd.extend(args.tests)
    else:
        # Run all tests in the directory
        cmd.append(str(test_dir))
    
    # Add options
    if args.verbose:
        cmd.append('-v')
    
    if args.fast:
        cmd.extend(['-m', 'not slow'])  # Skip slow tests
    
    if args.coverage:
        cmd.extend(['--cov=pymdp', '--cov-report=term-missing'])
        
    if args.html_cov:
        cmd.extend(['--cov=pymdp', '--cov-report=html'])
        
    if args.parallel:
        cmd.extend(['-n', str(args.parallel)])
    
    # Add standard options for better output
    cmd.extend([
        '--tb=short',  # Shorter traceback format
        '--strict-markers',  # Be strict about test markers
        '--disable-warnings'  # Reduce noise from warnings
    ])
    
    print("Running command:", ' '.join(cmd))
    print("=" * 60)
    
    # Run the tests
    try:
        result = subprocess.run(cmd, cwd=test_dir.parent)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
