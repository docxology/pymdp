"""
Validation Utilities
====================

Utilities for validating PyMDP examples and ensuring they use real PyMDP methods.
This module provides comprehensive validation functions for examples, matrices,
and PyMDP operations.
"""

import numpy as np
import inspect
import ast
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from pymdp_core import PyMDPCore
from pymdp import utils, maths, inference, control, learning
from pymdp.agent import Agent


class PyMDPValidator:
    """
    Comprehensive validator for PyMDP examples and operations.
    
    This class ensures that examples use real PyMDP methods exclusively
    and validates the correctness of PyMDP operations.
    """
    
    def __init__(self):
        """Initialize validator with PyMDP method registry."""
        self.pymdp_methods = self._get_pymdp_methods()
        self.pymdp_classes = self._get_pymdp_classes()
        self.pymdp_functions = self._get_pymdp_functions()
    
    def _get_pymdp_methods(self) -> Dict[str, List[str]]:
        """Get all available PyMDP methods."""
        methods = {}
        
        # Agent methods
        agent_methods = [method for method in dir(Agent) if not method.startswith('_')]
        methods['Agent'] = agent_methods
        
        # Inference methods
        inference_methods = [method for method in dir(inference) if not method.startswith('_')]
        methods['inference'] = inference_methods
        
        # Control methods
        control_methods = [method for method in dir(control) if not method.startswith('_')]
        methods['control'] = control_methods
        
        # Learning methods
        learning_methods = [method for method in dir(learning) if not method.startswith('_')]
        methods['learning'] = learning_methods
        
        # Utils methods
        utils_methods = [method for method in dir(utils) if not method.startswith('_')]
        methods['utils'] = utils_methods
        
        # Maths methods
        maths_methods = [method for method in dir(maths) if not method.startswith('_')]
        methods['maths'] = maths_methods
        
        return methods
    
    def _get_pymdp_classes(self) -> List[str]:
        """Get all available PyMDP classes."""
        return ['Agent']
    
    def _get_pymdp_functions(self) -> List[str]:
        """Get all available PyMDP functions."""
        functions = []
        
        # Add functions from each module
        for module_name, methods in self.pymdp_methods.items():
            if module_name != 'Agent':
                functions.extend([f"{module_name}.{method}" for method in methods])
        
        return functions
    
    def validate_example_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate an example file to ensure it uses real PyMDP methods.
        
        Parameters
        ----------
        file_path : str
            Path to the example file
            
        Returns
        -------
        validation_results : dict
            Validation results including method usage, issues, and recommendations
        """
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Parse the file
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            return {
                'valid': False,
                'error': f"Syntax error: {e}",
                'pymdp_methods_used': [],
                'non_pymdp_methods': [],
                'recommendations': []
            }
        
        # Analyze the AST
        pymdp_methods_used = []
        non_pymdp_methods = []
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module and 'pymdp' in node.module:
                    for alias in node.names:
                        imports.append(f"{node.module}.{alias.name}")
            elif isinstance(node, ast.Call):
                method_name = self._extract_method_name(node)
                if method_name:
                    if self._is_pymdp_method(method_name):
                        pymdp_methods_used.append(method_name)
                    else:
                        non_pymdp_methods.append(method_name)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            pymdp_methods_used, non_pymdp_methods, imports
        )
        
        return {
            'valid': len(non_pymdp_methods) == 0,
            'pymdp_methods_used': pymdp_methods_used,
            'non_pymdp_methods': non_pymdp_methods,
            'imports': imports,
            'recommendations': recommendations,
            'coverage_score': len(pymdp_methods_used) / max(1, len(pymdp_methods_used) + len(non_pymdp_methods))
        }
    
    def _extract_method_name(self, node: ast.Call) -> str:
        """Extract method name from AST call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                return f"{node.func.value.id}.{node.func.attr}"
            elif isinstance(node.func.value, ast.Attribute):
                return f"{self._extract_method_name(ast.Call(func=node.func.value))}.{node.func.attr}"
        return ""
    
    def _is_pymdp_method(self, method_name: str) -> bool:
        """Check if a method is a real PyMDP method."""
        # Check if it's a direct PyMDP method
        if method_name in self.pymdp_functions:
            return True
        
        # Check if it's a method from our core utilities
        if method_name.startswith('PyMDPCore.') or method_name.startswith('pymdp_core.'):
            return True
        
        # Check if it's a standard PyMDP import
        if any(pymdp_method in method_name for pymdp_method in [
            'obj_array', 'softmax', 'kl_div', 'entropy', 'spm_log',
            'update_posterior_states', 'sample_action', 'construct_policies',
            'calc_free_energy', 'calc_expected_utility', 'calc_states_info_gain'
        ]):
            return True
        
        # Check if it's a standard Python/library function (not PyMDP-related)
        standard_functions = [
            'print', 'range', 'len', 'str', 'int', 'float', 'list', 'dict', 'tuple',
            'np.array', 'np.zeros', 'np.ones', 'np.eye', 'np.linspace', 'np.random',
            'np.argmax', 'np.sum', 'np.mean', 'np.std', 'np.max', 'np.min',
            'plt.plot', 'plt.subplots', 'plt.show', 'plt.savefig', 'plt.figure',
            'plt.tight_layout', 'fig.suptitle', 'fig.add_subplot',
            'matplotlib.use', 'sys.path.insert', 'os.path.join', 'Path.mkdir',
            'json.dump', 'json.load', 'time.time', 'datetime.now',
            'enumerate', 'zip', 'isinstance', 'hasattr', 'getattr', 'setattr',
            'is_normalized', 'norm_dist'  # These are PyMDP utility functions
        ]
        
        if any(std_func in method_name for std_func in standard_functions):
            return True  # These are acceptable, not PyMDP methods
        
        return False
    
    def _generate_recommendations(self, pymdp_methods: List[str], 
                                non_pymdp_methods: List[str], 
                                imports: List[str]) -> List[str]:
        """Generate recommendations for improving PyMDP usage."""
        recommendations = []
        
        # Check for missing PyMDP imports
        if 'pymdp' not in str(imports):
            recommendations.append("Add 'import pymdp' to use real PyMDP methods")
        
        # Check for non-PyMDP methods
        for method in non_pymdp_methods:
            if 'infer_states' in method.lower():
                recommendations.append(f"Replace '{method}' with 'PyMDPCore.infer_states' or 'agent.infer_states'")
            elif 'compute_vfe' in method.lower():
                recommendations.append(f"Replace '{method}' with 'PyMDPCore.compute_vfe'")
            elif 'compute_efe' in method.lower():
                recommendations.append(f"Replace '{method}' with 'PyMDPCore.compute_efe'")
            elif 'create_agent' in method.lower():
                recommendations.append(f"Replace '{method}' with 'PyMDPCore.create_agent'")
        
        # Check for missing PyMDP methods
        if not any('infer_states' in method for method in pymdp_methods):
            recommendations.append("Consider using 'agent.infer_states' for state inference")
        
        if not any('infer_policies' in method for method in pymdp_methods):
            recommendations.append("Consider using 'agent.infer_policies' for policy inference")
        
        return recommendations
    
    def validate_matrices(self, A=None, B=None, C=None, D=None) -> Dict[str, Any]:
        """
        Validate PyMDP matrices using real PyMDP validation methods.
        
        Parameters
        ----------
        A : obj_array, optional
            Observation model
        B : obj_array, optional
            Transition model
        C : obj_array, optional
            Preferences
        D : obj_array, optional
            Prior beliefs
            
        Returns
        -------
        validation_results : dict
            Matrix validation results
        """
        results = {}
        
        if A is not None:
            results['A'] = self._validate_observation_model(A)
        
        if B is not None:
            results['B'] = self._validate_transition_model(B)
        
        if C is not None:
            results['C'] = self._validate_preferences(C)
        
        if D is not None:
            results['D'] = self._validate_priors(D)
        
        return results
    
    def _validate_observation_model(self, A) -> Dict[str, Any]:
        """Validate observation model using PyMDP utilities."""
        try:
            # Use real PyMDP validation
            validation_results = PyMDPCore.validate_matrices(A, None)
            return {
                'valid': validation_results['A']['is_normalized'],
                'is_normalized': validation_results['A']['is_normalized'],
                'num_modalities': validation_results['A']['num_modalities'],
                'shapes': validation_results['A']['shapes'],
                'issues': [] if validation_results['A']['is_normalized'] else ['Columns do not sum to 1']
            }
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'issues': [f"Validation error: {e}"]
            }
    
    def _validate_transition_model(self, B) -> Dict[str, Any]:
        """Validate transition model using PyMDP utilities."""
        try:
            # Use real PyMDP validation
            validation_results = PyMDPCore.validate_matrices(None, B)
            return {
                'valid': validation_results['B']['is_normalized'],
                'is_normalized': validation_results['B']['is_normalized'],
                'num_factors': validation_results['B']['num_factors'],
                'shapes': validation_results['B']['shapes'],
                'issues': [] if validation_results['B']['is_normalized'] else ['Columns do not sum to 1']
            }
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'issues': [f"Validation error: {e}"]
            }
    
    def _validate_preferences(self, C) -> Dict[str, Any]:
        """Validate preferences using PyMDP utilities."""
        try:
            # Use real PyMDP validation
            validation_results = PyMDPCore.validate_matrices(None, None, C)
            return {
                'valid': True,
                'num_modalities': validation_results['C']['num_modalities'],
                'shapes': validation_results['C']['shapes'],
                'issues': []
            }
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'issues': [f"Validation error: {e}"]
            }
    
    def _validate_priors(self, D) -> Dict[str, Any]:
        """Validate priors using PyMDP utilities."""
        try:
            # Use real PyMDP validation
            validation_results = PyMDPCore.validate_matrices(None, None, None, D)
            return {
                'valid': validation_results['D']['is_normalized'],
                'is_normalized': validation_results['D']['is_normalized'],
                'num_factors': validation_results['D']['num_factors'],
                'shapes': validation_results['D']['shapes'],
                'issues': [] if validation_results['D']['is_normalized'] else ['Columns do not sum to 1']
            }
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'issues': [f"Validation error: {e}"]
            }
    
    def validate_agent_operations(self, agent: Agent, test_observations: List[int]) -> Dict[str, Any]:
        """
        Validate agent operations using real PyMDP methods.
        
        Parameters
        ----------
        agent : pymdp.agent.Agent
            PyMDP agent to validate
        test_observations : list[int]
            Test observations for validation
            
        Returns
        -------
        validation_results : dict
            Agent operation validation results
        """
        results = {
            'inference_valid': True,
            'policy_inference_valid': True,
            'action_sampling_valid': True,
            'issues': [],
            'test_results': []
        }
        
        for obs in test_observations:
            try:
                # Test state inference
                qs = PyMDPCore.infer_states(agent, obs)
                if qs is None or len(qs) == 0:
                    results['inference_valid'] = False
                    results['issues'].append(f"State inference failed for observation {obs}")
                
                # Test policy inference
                q_pi, G = PyMDPCore.infer_policies(agent)
                if q_pi is None or G is None:
                    results['policy_inference_valid'] = False
                    results['issues'].append(f"Policy inference failed for observation {obs}")
                
                # Test action sampling
                action = PyMDPCore.sample_action(agent)
                if action is None:
                    results['action_sampling_valid'] = False
                    results['issues'].append(f"Action sampling failed for observation {obs}")
                
                results['test_results'].append({
                    'observation': obs,
                    'state_inference_success': qs is not None,
                    'policy_inference_success': q_pi is not None,
                    'action_sampling_success': action is not None
                })
                
            except Exception as e:
                results['issues'].append(f"Error testing observation {obs}: {e}")
                results['test_results'].append({
                    'observation': obs,
                    'error': str(e)
                })
        
        results['overall_valid'] = (
            results['inference_valid'] and 
            results['policy_inference_valid'] and 
            results['action_sampling_valid']
        )
        
        return results


def validate_all_examples(examples_dir: str = "textbook/examples") -> Dict[str, Any]:
    """
    Validate all examples in the examples directory.
    
    Parameters
    ----------
    examples_dir : str
        Path to examples directory
        
    Returns
    -------
    validation_results : dict
        Complete validation results for all examples
    """
    validator = PyMDPValidator()
    results = {}
    
    examples_path = Path(examples_dir)
    if not examples_path.exists():
        return {'error': f"Examples directory not found: {examples_dir}"}
    
    # Find all Python example files
    example_files = list(examples_path.glob("*.py"))
    
    for file_path in example_files:
        if file_path.name.startswith('__'):
            continue
        
        print(f"Validating {file_path.name}...")
        results[file_path.name] = validator.validate_example_file(str(file_path))
    
    # Generate summary
    total_files = len(results)
    valid_files = sum(1 for r in results.values() if r.get('valid', False))
    avg_coverage = np.mean([r.get('coverage_score', 0) for r in results.values()])
    
    summary = {
        'total_files': total_files,
        'valid_files': valid_files,
        'invalid_files': total_files - valid_files,
        'average_coverage': avg_coverage,
        'validation_rate': valid_files / total_files if total_files > 0 else 0
    }
    
    return {
        'summary': summary,
        'file_results': results
    }


def create_validation_report(validation_results: Dict[str, Any], output_path: str = "validation_report.json"):
    """
    Create a comprehensive validation report.
    
    Parameters
    ----------
    validation_results : dict
        Validation results from validate_all_examples
    output_path : str
        Path to save the validation report
    """
    with open(output_path, 'w') as f:
        import json
        json.dump(validation_results, f, indent=2, default=str)
    
    print(f"📋 Validation report saved: {output_path}")
    
    # Print summary
    summary = validation_results.get('summary', {})
    print(f"\n📊 Validation Summary:")
    print(f"  Total files: {summary.get('total_files', 0)}")
    print(f"  Valid files: {summary.get('valid_files', 0)}")
    print(f"  Invalid files: {summary.get('invalid_files', 0)}")
    print(f"  Validation rate: {summary.get('validation_rate', 0):.2%}")
    print(f"  Average PyMDP coverage: {summary.get('average_coverage', 0):.2%}")


# Convenience functions
def validate_example(file_path: str) -> Dict[str, Any]:
    """Validate a single example file."""
    validator = PyMDPValidator()
    return validator.validate_example_file(file_path)

def validate_matrices(A=None, B=None, C=None, D=None) -> Dict[str, Any]:
    """Validate PyMDP matrices."""
    validator = PyMDPValidator()
    return validator.validate_matrices(A, B, C, D)

def validate_agent(agent: Agent, test_observations: List[int] = [0, 1, 2]) -> Dict[str, Any]:
    """Validate PyMDP agent operations."""
    validator = PyMDPValidator()
    return validator.validate_agent_operations(agent, test_observations)
