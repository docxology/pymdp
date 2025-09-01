"""
PyMDP Textbook Testing Framework
===============================

This package contains comprehensive tests for all PyMDP methods and components.

Test Structure:
- test_core.py: Core PyMDP functionality tests
- test_inference.py: Inference method tests
- test_control.py: Control method tests
- test_learning.py: Learning method tests
- test_rgm.py: RGM (Relational Generative Model) tests
- test_utils.py: Utility function tests
- test_integration.py: Integration and end-to-end tests

Usage:
    # Run all tests
    python -m pytest tests/
    
    # Run specific test module
    python -m pytest tests/test_core.py
    
    # Run with coverage
    python -m pytest tests/ --cov=pymdp --cov-report=html
"""

__version__ = "1.0.0"
__author__ = "PyMDP Textbook Contributors"
