"""
Tests for Example Utilities
===========================

Comprehensive tests for the example utilities to ensure they work correctly
with real PyMDP methods.
"""

import pytest
import numpy as np
import sys
import os
from pathlib import Path
import tempfile
import shutil

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from example_utils import (
    ExampleRunner, MatrixBuilder, AnalysisUtils,
    create_standard_example_setup, run_standard_analysis
)
from pymdp_core import PyMDPCore
from pymdp import utils
from pymdp.utils import obj_array_zeros


class TestExampleRunner:
    """Test ExampleRunner class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.runner = ExampleRunner("test_example", self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test ExampleRunner initialization."""
        assert self.runner.example_name == "test_example"
        assert self.runner.output_dir.exists()
        assert len(self.runner.results) == 0
        assert len(self.runner.visualizations) == 0
    
    def test_save_results(self):
        """Test saving results to JSON."""
        test_results = {
            'test_key': 'test_value',
            'numbers': [1, 2, 3],
            'numpy_array': np.array([1, 2, 3])
        }
        
        self.runner.save_results(test_results, "test_results.json")
        
        # Check file was created
        output_file = self.runner.output_dir / "test_results.json"
        assert output_file.exists()
        
        # Check content
        import json
        with open(output_file, 'r') as f:
            loaded_results = json.load(f)
        
        assert loaded_results['test_key'] == 'test_value'
        assert loaded_results['numbers'] == [1, 2, 3]
    
    def test_save_visualization(self):
        """Test saving visualizations."""
        import matplotlib.pyplot as plt
        
        # Create a simple figure
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        ax.set_title("Test Plot")
        
        self.runner.save_visualization(
            fig, "test_plot.png", 
            title="Test Plot", 
            description="A test visualization"
        )
        
        # Check file was created
        output_file = self.runner.output_dir / "test_plot.png"
        assert output_file.exists()
        
        # Check visualization metadata
        assert len(self.runner.visualizations) == 1
        viz_info = self.runner.visualizations[0]
        assert viz_info['filename'] == "test_plot.png"
        assert viz_info['title'] == "Test Plot"
        assert viz_info['description'] == "A test visualization"
    
    def test_create_summary(self):
        """Test creating example summary."""
        # Add some test data
        self.runner.results = {'test': 'data'}
        self.runner.visualizations = [{'filename': 'test.png'}]
        
        summary = self.runner.create_summary()
        
        assert summary['example_name'] == "test_example"
        assert summary['results'] == {'test': 'data'}
        assert summary['visualizations'] == [{'filename': 'test.png'}]
        assert 'timestamp' in summary
        
        # Check summary file was created
        summary_file = self.runner.output_dir / "test_example_summary.json"
        assert summary_file.exists()


class TestMatrixBuilder:
    """Test MatrixBuilder class."""
    
    def test_create_observation_model_identity(self):
        """Test creating identity observation model."""
        A = MatrixBuilder.create_observation_model(3, 3, "identity")
        
        assert A is not None
        assert len(A) == 1  # Single modality
        assert A[0].shape == (3, 3)
        assert np.allclose(A[0], np.eye(3))
    
    def test_create_observation_model_noisy(self):
        """Test creating noisy observation model."""
        A = MatrixBuilder.create_observation_model(3, 3, "noisy", noise=0.1)
        
        assert A is not None
        assert len(A) == 1
        assert A[0].shape == (3, 3)
        
        # Check normalization
        assert np.allclose(np.sum(A[0], axis=0), 1.0)
    
    def test_create_observation_model_random(self):
        """Test creating random observation model."""
        A = MatrixBuilder.create_observation_model(3, 3, "random")
        
        assert A is not None
        assert len(A) == 1
        assert A[0].shape == (3, 3)
        
        # Check normalization
        assert np.allclose(np.sum(A[0], axis=0), 1.0)
    
    def test_create_transition_model_random(self):
        """Test creating random transition model."""
        B = MatrixBuilder.create_transition_model(3, 2, "random")
        
        assert B is not None
        assert len(B) == 1  # Single factor
        assert B[0].shape == (3, 3, 2)
        
        # Check normalization for each action
        for a in range(2):
            assert np.allclose(np.sum(B[0][:, :, a], axis=0), 1.0)
    
    def test_create_transition_model_deterministic(self):
        """Test creating deterministic transition model."""
        B = MatrixBuilder.create_transition_model(3, 2, "deterministic")
        
        assert B is not None
        assert len(B) == 1
        assert B[0].shape == (3, 3, 2)
        
        # Check that each column has exactly one 1.0
        for a in range(2):
            for s in range(3):
                col = B[0][:, s, a]
                assert np.sum(col) == 1.0
                assert np.sum(col == 1.0) == 1
    
    def test_create_preferences_uniform(self):
        """Test creating uniform preferences."""
        C = MatrixBuilder.create_preferences(3, "uniform")
        
        assert C is not None
        assert len(C) == 1  # Single modality
        assert C[0].shape == (3,)
        assert np.allclose(C[0], 0.0)
    
    def test_create_preferences_linear(self):
        """Test creating linear preferences."""
        C = MatrixBuilder.create_preferences(3, "linear")
        
        assert C is not None
        assert len(C) == 1
        assert C[0].shape == (3,)
        assert np.allclose(C[0], np.linspace(-1, 1, 3))
    
    def test_create_preferences_custom(self):
        """Test creating custom preferences."""
        custom_values = [0.5, -0.3, 1.0]
        C = MatrixBuilder.create_preferences(3, "custom", values=custom_values)
        
        assert C is not None
        assert len(C) == 1
        assert C[0].shape == (3,)
        assert np.allclose(C[0], custom_values)
    
    def test_create_prior_uniform(self):
        """Test creating uniform prior."""
        D = MatrixBuilder.create_prior(3, "uniform")
        
        assert D is not None
        assert len(D) == 1  # Single factor
        assert D[0].shape == (3,)
        assert np.allclose(D[0], np.ones(3) / 3)
        assert np.allclose(np.sum(D[0]), 1.0)
    
    def test_create_prior_custom(self):
        """Test creating custom prior."""
        custom_values = [0.6, 0.3, 0.1]
        D = MatrixBuilder.create_prior(3, "custom", values=custom_values)
        
        assert D is not None
        assert len(D) == 1
        assert D[0].shape == (3,)
        assert np.allclose(D[0], custom_values)
        assert np.allclose(np.sum(D[0]), 1.0)


class TestAnalysisUtils:
    """Test AnalysisUtils class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create test matrices
        self.A = obj_array_zeros([[3, 3]])
        self.A[0] = np.eye(3)
        
        self.B = obj_array_zeros([[3, 3, 2]])
        for a in range(2):
            for s in range(3):
                next_s = (s + a) % 3
                self.B[0][next_s, s, a] = 1.0
        
        self.C = obj_array_zeros([[3]])
        self.C[0] = np.array([-1, 0, 1])
        
        self.D = obj_array_zeros([[3]])
        self.D[0] = np.ones(3) / 3
    
    def test_analyze_vfe_components(self):
        """Test VFE component analysis."""
        observations = [0, 1, 2]
        priors = [self.D, self.D, self.D]
        
        analysis = AnalysisUtils.analyze_vfe_components(
            self.A, observations, priors
        )
        
        assert 'vfe_results' in analysis
        assert 'mean_vfe' in analysis
        assert 'vfe_std' in analysis
        
        assert len(analysis['vfe_results']) == 3
        for result in analysis['vfe_results']:
            assert 'observation' in result
            assert 'vfe' in result
            assert 'complexity' in result
            assert 'accuracy' in result
            assert isinstance(result['vfe'], float)
    
    def test_analyze_efe_components(self):
        """Test EFE component analysis."""
        beliefs = np.array([0.5, 0.3, 0.2])
        policies = [[0], [1]]
        
        analysis = AnalysisUtils.analyze_efe_components(
            self.A, self.B, self.C, beliefs, policies
        )
        
        assert 'efe_results' in analysis
        assert 'best_policy' in analysis
        assert 'mean_efe' in analysis
        
        assert len(analysis['efe_results']) == 2
        for result in analysis['efe_results']:
            assert 'policy' in result
            assert 'efe' in result
            assert 'pragmatic_value' in result
            assert 'epistemic_value' in result
            assert isinstance(result['efe'], float)
    
    def test_compare_inference_methods(self):
        """Test inference method comparison."""
        observations = [0, 1]
        priors = [self.D, self.D]
        
        comparison = AnalysisUtils.compare_inference_methods(
            self.A, observations, priors
        )
        
        assert 'comparison_results' in comparison
        assert 'mean_difference' in comparison
        
        assert len(comparison['comparison_results']) == 2
        for result in comparison['comparison_results']:
            assert 'observation' in result
            assert 'pymdp_posterior' in result
            assert 'manual_posterior' in result
            assert 'difference' in result
            assert isinstance(result['difference'], float)


class TestStandardSetup:
    """Test standard example setup functions."""
    
    def test_create_standard_example_setup(self):
        """Test creating standard example setup."""
        setup = create_standard_example_setup("test_example", 3, 2, 3)
        
        assert 'A' in setup
        assert 'B' in setup
        assert 'C' in setup
        assert 'D' in setup
        assert 'agent' in setup
        assert 'runner' in setup
        assert 'num_states' in setup
        assert 'num_actions' in setup
        assert 'num_obs' in setup
        
        assert setup['num_states'] == 3
        assert setup['num_actions'] == 2
        assert setup['num_obs'] == 3
        
        # Check matrices
        assert setup['A'][0].shape == (3, 3)
        assert setup['B'][0].shape == (3, 3, 2)
        assert setup['C'][0].shape == (3,)
        assert setup['D'][0].shape == (3,)
        
        # Check agent
        assert setup['agent'] is not None
        assert hasattr(setup['agent'], 'A')
        
        # Check runner
        assert setup['runner'] is not None
        assert setup['runner'].example_name == "test_example"
    
    def test_run_standard_analysis(self):
        """Test running standard analysis."""
        setup = create_standard_example_setup("test_example", 3, 2, 3)
        observations = [0, 1, 2]
        
        analysis = run_standard_analysis(setup, observations)
        
        assert 'vfe_analysis' in analysis
        assert 'efe_analysis' in analysis
        assert 'inference_comparison' in analysis
        
        # Check VFE analysis
        vfe_analysis = analysis['vfe_analysis']
        assert 'vfe_results' in vfe_analysis
        assert 'mean_vfe' in vfe_analysis
        
        # Check EFE analysis
        efe_analysis = analysis['efe_analysis']
        assert 'efe_results' in efe_analysis
        assert 'best_policy' in efe_analysis
        
        # Check inference comparison
        inference_comparison = analysis['inference_comparison']
        assert 'comparison_results' in inference_comparison
        assert 'mean_difference' in inference_comparison


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_matrix_types(self):
        """Test handling of invalid matrix types."""
        with pytest.raises(ValueError):
            MatrixBuilder.create_observation_model(3, 3, "invalid_type")
        
        with pytest.raises(ValueError):
            MatrixBuilder.create_transition_model(3, 2, "invalid_type")
        
        with pytest.raises(ValueError):
            MatrixBuilder.create_preferences(3, "invalid_type")
        
        with pytest.raises(ValueError):
            MatrixBuilder.create_prior(3, "invalid_type")
    
    def test_empty_observations(self):
        """Test handling of empty observations."""
        setup = create_standard_example_setup("test_example", 3, 2, 3)
        
        # Should handle empty observations gracefully
        analysis = run_standard_analysis(setup, [])
        
        assert 'vfe_analysis' in analysis
        assert len(analysis['vfe_analysis']['vfe_results']) == 0
    
    def test_mismatched_dimensions(self):
        """Test handling of dimension mismatches."""
        # Create setup with mismatched dimensions
        setup = create_standard_example_setup("test_example", 3, 2, 2)
        
        # Should handle gracefully
        observations = [0, 1]
        analysis = run_standard_analysis(setup, observations)
        
        assert 'vfe_analysis' in analysis
        assert 'efe_analysis' in analysis


if __name__ == "__main__":
    pytest.main([__file__])
