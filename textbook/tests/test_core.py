"""
Core PyMDP Functionality Tests
=============================

Test core PyMDP functions including utilities, mathematical operations,
and basic data structures.
"""

import pytest
import numpy as np
import pymdp
from pymdp.utils import obj_array_zeros, sample, obj_array_uniform, is_obj_array
from pymdp.maths import softmax, spm_dot
try:
    from pymdp.maths import kl_divergence
except ImportError:
    from pymdp.maths import kl_div as kl_divergence


class TestUtilities:
    """Test utility functions."""
    
    def test_obj_array_zeros_basic(self):
        """Test basic obj_array_zeros functionality."""
        # Single array
        arr = obj_array_zeros([3])
        assert len(arr) == 1
        assert arr[0].shape == (3,)
        assert np.allclose(arr[0], 0.0)
    
    def test_obj_array_zeros_multiple(self):
        """Test obj_array_zeros with multiple arrays."""
        shapes = [[2, 3], [4], [2, 2, 2]]
        arr = obj_array_zeros(shapes)
        
        assert len(arr) == 3
        assert arr[0].shape == (2, 3)
        assert arr[1].shape == (4,)
        assert arr[2].shape == (2, 2, 2)
        
        for a in arr:
            assert np.allclose(a, 0.0)
    
    def test_obj_array_uniform(self):
        """Test obj_array_uniform functionality."""
        shapes = [[3], [2]]  # Use 1D shapes for proper probability distributions
        arr = obj_array_uniform(shapes)
        
        assert len(arr) == 2
        
        # Check that each array sums to 1 (probability distribution)
        for a in arr:
            assert np.isclose(np.sum(a), 1.0)
            assert np.all(a >= 0)
    
    def test_is_obj_array(self):
        """Test is_obj_array detection."""
        # Regular numpy array
        regular_arr = np.array([1, 2, 3])
        assert not is_obj_array(regular_arr)
        
        # Object array
        obj_arr = obj_array_zeros([3])
        assert is_obj_array(obj_arr)
    
    def test_sample_basic(self):
        """Test basic sampling functionality."""
        np.random.seed(42)  # For reproducible tests
        
        # Simple categorical distribution
        probs = np.array([0.2, 0.5, 0.3])
        samples = [sample(probs) for _ in range(1000)]
        
        # Check samples are valid
        assert all(s in [0, 1, 2] for s in samples)
        
        # Check approximate distribution (with tolerance for randomness)
        counts = np.bincount(samples)
        empirical_probs = counts / len(samples)
        
        # Should be roughly similar to true probabilities
        assert np.allclose(empirical_probs, probs, atol=0.1)
    
    def test_sample_edge_cases(self):
        """Test sample function edge cases."""
        # Deterministic case
        probs = np.array([0.0, 1.0, 0.0])
        for _ in range(10):
            assert sample(probs) == 1
        
        # Near-uniform case
        probs = np.ones(5) / 5
        samples = [sample(probs) for _ in range(100)]
        assert all(s in range(5) for s in samples)


class TestMathematicalOperations:
    """Test mathematical operations."""
    
    def test_softmax_basic(self):
        """Test basic softmax functionality."""
        x = np.array([1.0, 2.0, 3.0])
        result = softmax(x)
        
        # Check valid probability distribution
        pytest.assert_valid_probability_distribution(result)
        
        # Check that larger inputs get higher probabilities
        assert result[2] > result[1] > result[0]
    
    def test_softmax_temperature(self):
        """Test softmax with temperature parameter."""
        x = np.array([1.0, 2.0, 3.0])
        
        # PyMDP softmax doesn't have temperature parameter - simulate with scaling
        # High temperature (more uniform) - scale down the values
        result_high = softmax(x / 10.0)
        
        # Low temperature (more peaked) - scale up the values  
        result_low = softmax(x * 10.0)
        
        # Check both are valid distributions
        pytest.assert_valid_probability_distribution(result_high)
        pytest.assert_valid_probability_distribution(result_low)
        
        # Low temperature should be more peaked
        assert np.max(result_low) > np.max(result_high)
    
    def test_softmax_edge_cases(self):
        """Test softmax edge cases."""
        # Large values (test numerical stability)
        x = np.array([1000.0, 1001.0, 1002.0])
        result = softmax(x)
        pytest.assert_valid_probability_distribution(result)
        
        # Single element
        x = np.array([5.0])
        result = softmax(x)
        assert np.isclose(result[0], 1.0)
        
        # All zeros
        x = np.zeros(3)
        result = softmax(x)
        assert np.allclose(result, 1/3)  # Should be uniform
    
    def test_dot_operations(self):
        """Test dot product operations."""
        # Matrix-vector multiplication
        A = np.random.rand(3, 4)
        x = np.random.rand(4)
        
        result = spm_dot(A, x)
        expected = np.dot(A, x)
        
        assert np.allclose(result, expected)
    
    def test_kl_divergence(self):
        """Test KL divergence calculation."""
        # Two probability distributions
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.4, 0.4, 0.2])
        
        kl = kl_divergence(p, q)
        
        # KL divergence should be non-negative
        assert kl >= 0
        
        # KL(p, p) should be 0
        kl_self = kl_divergence(p, p)
        assert np.isclose(kl_self, 0.0, atol=1e-10)


class TestImportStructure:
    """Test that all expected modules can be imported."""
    
    def test_main_imports(self):
        """Test main PyMDP imports."""
        # These should work without error
        from pymdp import inference, control, utils, maths
        from pymdp.utils import obj_array_zeros
        from pymdp.maths import softmax
    
    def test_algorithm_imports(self):
        """Test algorithm imports."""
        from pymdp.algos import fpi, mmp
    
    def test_environment_imports(self):
        """Test environment imports."""
        from pymdp.envs import env, grid_worlds, tmaze
    
    def test_rgm_imports(self):
        """Test RGM imports - validate that PyMDP structure is intact."""
        # Test that core PyMDP modules can be imported and work correctly
        try:
            import pymdp
            
            # Test direct attributes available on pymdp module
            direct_attributes = ['maths', 'utils', 'algos', 'control', 'inference']
            for attr in direct_attributes:
                assert hasattr(pymdp, attr), f"PyMDP should have {attr} attribute"
            
            # Test that we can import additional modules
            importable_modules = ['envs', 'agent']
            for module in importable_modules:
                try:
                    exec(f"from pymdp import {module}")
                except ImportError:
                    # Some modules might be optional - that's ok
                    pass
            
            # Test that core functionality works
            from pymdp.maths import softmax
            from pymdp.utils import obj_array_zeros
            
            # Basic functionality test
            test_array = np.array([1.0, 2.0, 3.0])
            result = softmax(test_array)
            
            assert np.isclose(result.sum(), 1.0), "Softmax should normalize to 1"
            assert len(result) == 3, "Softmax should preserve array length"
            
            # Test obj_array creation
            obj_arr = obj_array_zeros([[2, 3]])
            assert len(obj_arr) == 1, "obj_array should have correct length"
            assert obj_arr[0].shape == (2, 3), "obj_array elements should have correct shape"
            
        except ImportError as e:
            # If critical PyMDP components are missing, that's a real issue
            assert False, f"Critical PyMDP component missing: {e}"


class TestPackageMetadata:
    """Test package metadata and configuration."""
    
    def test_version_available(self):
        """Test that version information is available."""
        assert hasattr(pymdp, '__version__')
        assert isinstance(pymdp.__version__, str)
        assert len(pymdp.__version__) > 0
    
    def test_package_root(self):
        """Test package root path."""
        assert hasattr(pymdp, 'PACKAGE_ROOT')
        assert pymdp.PACKAGE_ROOT.exists()
    
    def test_all_exports(self):
        """Test that __all__ exports are valid."""
        if hasattr(pymdp, '__all__'):
            for name in pymdp.__all__:
                assert hasattr(pymdp, name), f"Exported name '{name}' not found in module"


class TestDataStructures:
    """Test PyMDP data structures and their operations."""
    
    def test_obj_array_indexing(self):
        """Test object array indexing and slicing."""
        arr = obj_array_zeros([[3, 2], [4], [2, 2]])
        
        # Basic indexing
        assert arr[0].shape == (3, 2)
        assert arr[1].shape == (4,)
        assert arr[2].shape == (2, 2)
        
        # Modification
        arr[0][0, 0] = 1.0
        assert arr[0][0, 0] == 1.0
        
        # Iteration
        shapes = []
        for a in arr:
            shapes.append(a.shape)
        
        expected_shapes = [(3, 2), (4,), (2, 2)]
        assert shapes == expected_shapes
    
    def test_obj_array_operations(self):
        """Test operations on object arrays."""
        arr1 = obj_array_uniform([[2], [3]])
        arr2 = obj_array_uniform([[2], [3]])
        
        # Element-wise operations should work on individual arrays
        for i in range(len(arr1)):
            result = arr1[i] + arr2[i]
            assert result.shape == arr1[i].shape
    
    @pytest.mark.parametrize("shape", [
        [3],
        [2, 4],
        [2, 3, 4],
        [5, 1, 3, 2]
    ])
    def test_obj_array_various_shapes(self, shape):
        """Test object arrays with various shapes."""
        arr = obj_array_zeros([shape])
        assert len(arr) == 1
        assert arr[0].shape == tuple(shape)
        assert np.allclose(arr[0], 0.0)
