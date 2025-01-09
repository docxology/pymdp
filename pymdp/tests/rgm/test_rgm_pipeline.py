"""
Unit Tests for Renormalization Generative Model
============================================

Tests for the Renormalization Generative Model implementation.
"""

import unittest
import torch
from pathlib import Path
import warnings
import pytest

class TestRenormalizationGenerativeModel(unittest.TestCase):
    """Test suite for the Renormalization Generative Model.""" 

def test_gnn_loader():
    """Test GNN specification loading."""
    loader = RGMGNNLoader()
    specs = loader.load_specifications(TEST_GNN_DIR)
    assert len(specs) > 0
    
    # Test deprecated method warning
    with warnings.catch_warnings(record=True) as w:
        specs_old = loader.load_gnn_specs(TEST_GNN_DIR)
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "deprecated" in str(w[-1].message) 

def test_renderer():
    """Test matrix rendering."""
    # Setup test directory
    test_dir = Path("test_experiment")
    test_dir.mkdir(exist_ok=True)
    gnn_dir = test_dir / "gnn_specs"
    gnn_dir.mkdir(exist_ok=True)
    
    # Create test GNN spec
    test_spec = """
    matrices:
      recognition:
        A0: [784, 256]
      generative:
        B0: [256, 784]
      lateral:
        D0: [256, 256]
    initialization:
      method: "orthogonal"
      gain: 1.0
    constraints:
      recognition:
        normalize: "row"
      generative:
        normalize: "column"
      lateral:
        symmetric: true
        positive: true
    """
    
    with open(gnn_dir / "test.gnn", "w") as f:
        f.write(test_spec)
    
    # Test renderer
    renderer = RGMRenderer(test_dir)
    matrices = renderer.render_matrices()
    
    # Verify matrices
    assert "A0" in matrices
    assert "B0" in matrices
    assert "D0" in matrices
    assert matrices["A0"].shape == (784, 256)
    assert matrices["B0"].shape == (256, 784)
    assert matrices["D0"].shape == (256, 256)
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir) 

def test_matrix_initialization():
    """Test matrix initialization methods."""
    # Test orthogonal initialization
    shape = (784, 256)
    matrix = initialize_orthogonal(shape)
    assert matrix.shape == shape
    
    # Verify orthogonality
    product = torch.mm(matrix.t(), matrix)
    assert torch.allclose(product, torch.eye(256), atol=1e-6)
    
    # Test normal initialization
    matrix = initialize_normal(shape)
    assert matrix.shape == shape
    assert -4 < matrix.mean() < 4  # Roughly zero mean
    assert 0 < matrix.std() < 0.1  # Small standard deviation
    
    # Test identity plus noise
    shape = (256, 256)  # Must be square
    matrix = initialize_identity_plus_noise(shape)
    assert matrix.shape == shape
    # Should be close to identity
    assert torch.allclose(matrix, torch.eye(256), atol=0.1) 

def test_gnn_directory_validation():
    """Test GNN directory validation."""
    # Setup test directory
    test_dir = Path("test_experiment")
    test_dir.mkdir(exist_ok=True)
    gnn_dir = test_dir / "gnn_specs"
    gnn_dir.mkdir(exist_ok=True)
    
    # Create test files
    required_files = [
        'rgm_base.gnn',
        'rgm_mnist.gnn',
        'rgm_message_passing.gnn',
        'rgm_hierarchical_level.gnn'
    ]
    
    test_content = """
    matrices:
      recognition:
        A0: [784, 256]
      generative:
        B0: [256, 784]
      lateral:
        D0: [256, 256]
    initialization:
      method: "orthogonal"
      gain: 1.0
    """
    
    for file in required_files:
        with open(gnn_dir / file, 'w') as f:
            f.write(test_content)
            
    # Test validation
    utils = RGMExperimentUtils()
    assert utils.validate_gnn_directory(gnn_dir)
    
    # Test missing file
    (gnn_dir / required_files[0]).unlink()
    with pytest.raises(FileNotFoundError):
        utils.validate_gnn_directory(gnn_dir)
        
    # Test empty file
    with open(gnn_dir / required_files[1], 'w') as f:
        f.write("")
    with pytest.raises(ValueError):
        utils.validate_gnn_directory(gnn_dir)
        
    # Cleanup
    import shutil
    shutil.rmtree(test_dir) 