import unittest
import numpy as np
import os
import shutil
from pathlib import Path
from pymdp.gnn.gnn_matrix_factory import GNNMatrixFactory
from pymdp.gnn.gnn_runner import GNNRunner
from pymdp.agent import Agent
from pymdp import utils

class TestGNN(unittest.TestCase):
    
    def setUp(self):
        """Setup test environment and basic GNN model"""
        self.test_dir = Path("test_outputs/gnn")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Load step-by-step example model
        self.model_path = Path("pymdp/gnn/models/step_by_step.gnn")
        self.runner = GNNRunner(self.model_path, sandbox_root=self.test_dir)
        
    def tearDown(self):
        """Clean up test outputs"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
            
    def test_model_loading(self):
        """Test GNN model loading and validation"""
        factory = GNNMatrixFactory(self.model_path)
        self.assertEqual(factory.gnn['modelName'], "Step_by_Step_POMDP")
        self.assertIn("Dynamic", factory.gnn['modelType'])
        
    def test_matrix_generation(self):
        """Test matrix generation from GNN definition"""
        matrices = self.runner.generate_matrices()
        
        # Check matrix shapes
        self.assertEqual(matrices['A'][0].shape, (3, 3))  # 3x3 observation matrix
        self.assertEqual(matrices['B'][0].shape, (3, 3, 2))  # 3x3x2 transition matrix
        self.assertEqual(matrices['C'][0].shape, (3,))  # Length 3 preference vector
        
        # Check matrix properties
        self.assertTrue(np.allclose(matrices['A'][0].sum(axis=0), 1.0))  # A is normalized
        self.assertTrue(np.allclose(matrices['B'][0].sum(axis=0), 1.0))  # B is normalized
        
    def test_sandbox_creation(self):
        """Test sandbox directory structure creation"""
        self.runner.generate_matrices()
        
        # Check directory structure
        self.assertTrue((self.test_dir / "matrices").exists())
        self.assertTrue((self.test_dir / "visualizations").exists())
        self.assertTrue((self.test_dir / "exports").exists())
        
    def test_markdown_conversion(self):
        """Test GNN to markdown conversion"""
        factory = GNNMatrixFactory(self.model_path)
        md_content = factory.to_markdown()
        
        # Check markdown content
        self.assertIn("Step-by-Step Active Inference POMDP", md_content)
        self.assertIn("Key Equations", md_content)
        self.assertIn("```yaml", md_content)
        
    def test_agent_creation(self):
        """Test creating PyMDP agent from GNN definition"""
        matrices = self.runner.generate_matrices()
        agent = Agent(**matrices)
        
        # Test basic agent functionality
        obs = [0]  # Initial observation
        qs = agent.infer_states(obs)
        
        self.assertEqual(len(qs), 1)  # One hidden state factor
        self.assertEqual(len(qs[0]), 3)  # Three possible states
        
    def test_visualization_generation(self):
        """Test visualization generation"""
        self.runner.generate_matrices()
        self.runner.generate_visualizations()
        
        # Check visualization outputs
        viz_dir = self.test_dir / "visualizations"
        self.assertTrue((viz_dir / "model_graph.png").exists())
        self.assertTrue((viz_dir / "A_matrices").exists())
        self.assertTrue((viz_dir / "B_matrices").exists())
        
    def test_full_pipeline(self):
        """Test complete GNN pipeline with AgentMaker integration"""
        from pymdp.environments import GridWorldEnv
        
        # Run experiment
        self.runner.run_experiment(GridWorldEnv, T=5)
        
        # Check experiment outputs
        self.assertTrue((self.test_dir / "traces").exists())
        self.assertTrue((self.test_dir / "exports" / "model_summary.json").exists())
        
    def test_error_handling(self):
        """Test error handling for invalid models"""
        invalid_model = {
            "modelName": "Invalid",
            "modelType": ["Dynamic"]
            # Missing required sections
        }
        
        invalid_path = self.test_dir / "invalid.gnn"
        with open(invalid_path, 'w') as f:
            json.dump(invalid_model, f)
            
        with self.assertRaises(ValueError):
            GNNMatrixFactory(invalid_path)

if __name__ == '__main__':
    unittest.main() 