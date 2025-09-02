#!/usr/bin/env python3
"""
Comprehensive Test Runner for PyMDP Textbook
============================================

This script runs all tests for the PyMDP textbook examples and utilities,
ensuring comprehensive validation of real PyMDP method usage.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path
import time

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from validation import validate_all_examples, create_validation_report


def run_pytest_tests(test_dir="textbook/tests", verbose=False, coverage=False):
    """
    Run pytest tests for all modules.
    
    Parameters
    ----------
    test_dir : str
        Directory containing test files
    verbose : bool
        Whether to run in verbose mode
    coverage : bool
        Whether to generate coverage report
        
    Returns
    -------
    success : bool
        Whether all tests passed
    """
    print("🧪 Running PyMDP Core Tests...")
    
    # Build pytest command
    cmd = ["python", "-m", "pytest", test_dir]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=../src", "--cov-report=html", "--cov-report=term"])
    
    # Add test files
    test_files = [
        "test_pymdp_core.py",
        "test_example_utils.py", 
        "test_validation.py"
    ]
    
    for test_file in test_files:
        test_path = os.path.join(test_dir, test_file)
        if os.path.exists(test_path):
            cmd.append(test_path)
    
    # Run tests
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False


def run_example_validation(examples_dir="textbook/examples", output_dir="test_outputs"):
    """
    Run validation on all examples.
    
    Parameters
    ----------
    examples_dir : str
        Directory containing example files
    output_dir : str
        Directory to save validation results
        
    Returns
    -------
    validation_results : dict
        Validation results
    """
    print("🔍 Validating PyMDP Examples...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run validation
    validation_results = validate_all_examples(examples_dir)
    
    # Save validation report
    report_path = os.path.join(output_dir, "validation_report.json")
    create_validation_report(validation_results, report_path)
    
    return validation_results


def run_example_tests(examples_dir="textbook/examples", timeout=300):
    """
    Run all example files to ensure they work correctly.
    
    Parameters
    ----------
    examples_dir : str
        Directory containing example files
    timeout : int
        Timeout in seconds for each example
        
    Returns
    -------
    results : dict
        Results of running examples
    """
    print("🚀 Running PyMDP Examples...")
    
    examples_path = Path(examples_dir)
    if not examples_path.exists():
        print(f"❌ Examples directory not found: {examples_dir}")
        return {}
    
    # Find all Python example files
    example_files = list(examples_path.glob("*.py"))
    example_files = [f for f in example_files if not f.name.startswith('__')]
    
    results = {}
    successful = 0
    failed = 0
    
    for example_file in example_files:
        print(f"  Running {example_file.name}...")
        
        try:
            # Run example with timeout
            result = subprocess.run(
                ["python", str(example_file)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=examples_path
            )
            
            if result.returncode == 0:
                print(f"    ✅ SUCCESS")
                successful += 1
                results[example_file.name] = {
                    'status': 'SUCCESS',
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
            else:
                print(f"    ❌ FAILED (exit code: {result.returncode})")
                failed += 1
                results[example_file.name] = {
                    'status': 'FAILED',
                    'exit_code': result.returncode,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
                
        except subprocess.TimeoutExpired:
            print(f"    ⏰ TIMEOUT")
            failed += 1
            results[example_file.name] = {
                'status': 'TIMEOUT',
                'error': 'Execution timed out'
            }
        except Exception as e:
            print(f"    ❌ ERROR: {e}")
            failed += 1
            results[example_file.name] = {
                'status': 'ERROR',
                'error': str(e)
            }
    
    print(f"\n📊 Example Results: {successful} successful, {failed} failed")
    
    return {
        'total': len(example_files),
        'successful': successful,
        'failed': failed,
        'results': results
    }


def generate_comprehensive_report(test_results, validation_results, example_results, output_dir="test_outputs"):
    """
    Generate comprehensive test report.
    
    Parameters
    ----------
    test_results : bool
        Whether pytest tests passed
    validation_results : dict
        Validation results
    example_results : dict
        Example execution results
    output_dir : str
        Output directory for reports
    """
    print("📋 Generating Comprehensive Report...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate report
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'test_results': {
            'pytest_passed': test_results,
            'status': 'PASSED' if test_results else 'FAILED'
        },
        'validation_results': validation_results,
        'example_results': example_results,
        'summary': {
            'overall_status': 'PASSED' if (test_results and 
                                          validation_results.get('summary', {}).get('validation_rate', 0) > 0.8 and
                                          example_results.get('successful', 0) > 0) else 'FAILED',
            'pytest_status': 'PASSED' if test_results else 'FAILED',
            'validation_rate': validation_results.get('summary', {}).get('validation_rate', 0),
            'example_success_rate': example_results.get('successful', 0) / max(1, example_results.get('total', 1))
        }
    }
    
    # Save report
    import json
    report_path = os.path.join(output_dir, "comprehensive_test_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"📋 Comprehensive report saved: {report_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print("="*60)
    print(f"Overall Status: {report['summary']['overall_status']}")
    print(f"Pytest Tests: {report['summary']['pytest_status']}")
    print(f"Validation Rate: {report['summary']['validation_rate']:.2%}")
    print(f"Example Success Rate: {report['summary']['example_success_rate']:.2%}")
    
    if validation_results.get('summary'):
        summary = validation_results['summary']
        print(f"\nValidation Details:")
        print(f"  Total Files: {summary.get('total_files', 0)}")
        print(f"  Valid Files: {summary.get('valid_files', 0)}")
        print(f"  Invalid Files: {summary.get('invalid_files', 0)}")
        print(f"  Average Coverage: {summary.get('average_coverage', 0):.2%}")
    
    if example_results.get('total'):
        print(f"\nExample Execution Details:")
        print(f"  Total Examples: {example_results.get('total', 0)}")
        print(f"  Successful: {example_results.get('successful', 0)}")
        print(f"  Failed: {example_results.get('failed', 0)}")
    
    print("="*60)


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run comprehensive PyMDP textbook tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", "-c", action="store_true", help="Generate coverage report")
    parser.add_argument("--examples-only", action="store_true", help="Run only example tests")
    parser.add_argument("--validation-only", action="store_true", help="Run only validation")
    parser.add_argument("--pytest-only", action="store_true", help="Run only pytest tests")
    parser.add_argument("--output-dir", default="test_outputs", help="Output directory for reports")
    parser.add_argument("--examples-dir", default="textbook/examples", help="Examples directory")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout for example execution")
    
    args = parser.parse_args()
    
    print("🧪 PyMDP Textbook Comprehensive Test Suite")
    print("=" * 50)
    
    # Initialize results
    test_results = True
    validation_results = {}
    example_results = {}
    
    # Run pytest tests
    if not args.examples_only and not args.validation_only:
        test_results = run_pytest_tests(verbose=args.verbose, coverage=args.coverage)
        print(f"Pytest Tests: {'✅ PASSED' if test_results else '❌ FAILED'}")
    
    # Run validation
    if not args.pytest_only and not args.examples_only:
        validation_results = run_example_validation(args.examples_dir, args.output_dir)
        validation_rate = validation_results.get('summary', {}).get('validation_rate', 0)
        print(f"Validation: {'✅ PASSED' if validation_rate > 0.8 else '❌ FAILED'} ({validation_rate:.2%})")
    
    # Run examples
    if not args.pytest_only and not args.validation_only:
        example_results = run_example_tests(args.examples_dir, args.timeout)
        success_rate = example_results.get('successful', 0) / max(1, example_results.get('total', 1))
        print(f"Examples: {'✅ PASSED' if success_rate > 0.8 else '❌ FAILED'} ({success_rate:.2%})")
    
    # Generate comprehensive report
    if not args.pytest_only:
        generate_comprehensive_report(test_results, validation_results, example_results, args.output_dir)
    
    # Determine overall success
    overall_success = (
        test_results and
        validation_results.get('summary', {}).get('validation_rate', 0) > 0.8 and
        example_results.get('successful', 0) > 0
    )
    
    if overall_success:
        print("\n🎉 All tests passed! PyMDP textbook is ready.")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Check the reports for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
