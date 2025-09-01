#!/usr/bin/env python3
"""
Quick installation test for PyMDP textbook setup
"""

import pymdp
import numpy as np

def test_basic_functionality():
    """Test basic PyMDP functionality"""
    print("Testing basic PyMDP functionality...")
    
    # Test utilities
    zeros = pymdp.obj_array_zeros([[2, 3]])
    print(f"obj_array_zeros created: {type(zeros)}")
    
    # Test math functions
    values = np.array([1.0, 2.0, 3.0])
    soft_values = pymdp.softmax(values)
    print(f"Softmax result: {soft_values}")
    
    print("✓ Basic functionality test passed")

if __name__ == "__main__":
    test_basic_functionality()
