"""
Check PyMDP Installation
"""

import sys
import importlib
from pathlib import Path

def check_import(module_name: str) -> bool:
    """Check if module can be imported"""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def main():
    """Check installation"""
    # Required modules
    modules = [
        'pymdp',
        'pymdp.rgm',
        'pymdp.gnn',
        'torch',
        'torchvision',
        'numpy',
        'matplotlib',
        'seaborn',
        'psutil',
        'yaml',
        'tqdm'
    ]
    
    print("\nChecking PyMDP installation...\n")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check each module
    all_passed = True
    for module in modules:
        result = check_import(module)
        status = "✓" if result else "✗"
        print(f"{status} {module}")
        all_passed &= result
        
    if all_passed:
        print("\nAll checks passed!")
    else:
        print("\nSome checks failed. Please check installation.")
        sys.exit(1)

if __name__ == "__main__":
    main() 