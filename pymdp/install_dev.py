#!/usr/bin/env python3
"""
Development Installation Script
=============================

This script installs PyMDP in development mode, which allows you to modify
the source code and have the changes immediately reflected without reinstalling.

Usage:
    python3 install_dev.py

Requirements:
    - Python 3.7+
    - pip
"""

import subprocess
import sys
from pathlib import Path

def install_dev():
    """Install package in development mode."""
    try:
        # Get the project root directory
        project_root = Path(__file__).parent.absolute()
        
        # Install in development mode
        subprocess.check_call([
            sys.executable, 
            "-m", 
            "pip", 
            "install", 
            "-e", 
            "."
        ])
        
        print("\nSuccessfully installed PyMDP in development mode!")
        print("\nYou can now import the package using:")
        print("    import pymdp")
        print("    from pymdp import rgm")
        
    except subprocess.CalledProcessError as e:
        print(f"\nError installing package: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    install_dev() 