"""
RGM Pipeline Manager
===================

Manages the complete RGM pipeline execution including:
1. Configuration loading and validation
2. Matrix rendering
3. Model execution
4. Results analysis
5. Error handling and recovery
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

from rgm_experiment_utils import RGMExperimentUtils
from rgm_config_loader import RGMConfigLoader
from rgm_model_state import RGMModelState
from rgm_data_manager import RGMDataManager
from rgm_matrix_normalizer import RGMMatrixNormalizer
from rgm_matrix_initializer import RGMMatrixInitializer

class RGMPipelineManager:
    """Manages RGM pipeline execution"""
    
    def __init__(self):
        """Initialize pipeline manager"""
        try:
            # Get experiment state
            self.experiment = RGMExperimentUtils.get_experiment()
            self.logger = RGMExperimentUtils.get_logger('pipeline')
            
            # Initialize components
            self.config_loader = RGMConfigLoader()
            self.matrix_normalizer = RGMMatrixNormalizer()
            self.matrix_initializer = RGMMatrixInitializer(self.config)
            
            # Load configuration
            self.config = self.config_loader.load_config()
            
            # Create checkpoint directory
            self.checkpoint_dir = self.experiment['dirs']['simulation'] / "checkpoints"
            self.checkpoint_dir.mkdir(exist_ok=True)
            
            self.logger.info("Pipeline manager initialization complete")
            
        except Exception as e:
            logging.error(f"Failed to initialize pipeline: {str(e)}")
            raise
            
    def execute_pipeline(self) -> Path:
        """
        Execute complete RGM pipeline.
        
        Returns:
            Path to results directory
        """
        try:
            self.logger.info("\n" + "="*80)
            self.logger.info(" Starting RGM Pipeline")
            self.logger.info("="*80 + "\n")
            
            start_time = time.time()
            
            # Stage 1: Matrix Rendering
            self.logger.info("\n" + "="*80)
            self.logger.info("🎨 Stage 1: Matrix Rendering")
            self.logger.info("="*80 + "\n")
            
            matrices_dir = self._run_matrix_rendering()
            
            # Normalize matrices
            self.logger.info("Normalizing matrices...")
            matrices = self._load_and_normalize_matrices(matrices_dir)
            
            # Stage 2: Model Execution
            self.logger.info("\n" + "="*80)
            self.logger.info("🧠 Stage 2: Model Execution")
            self.logger.info("="*80 + "\n")
            
            results_dir = self._run_model_execution(matrices)
            
            # Stage 3: Results Analysis
            self.logger.info("\n" + "="*80)
            self.logger.info("📊 Stage 3: Results Analysis")
            self.logger.info("="*80 + "\n")
            
            analysis_dir = self._run_results_analysis(results_dir)
            
            # Log completion
            duration = time.time() - start_time
            peak_memory = self._get_peak_memory_usage()
            
            self.logger.info(f"Pipeline completed in {duration:.2f}s")
            self.logger.info(f"Peak memory usage: {peak_memory}")
            
            # Log directory structure
            self._log_directory_structure()
            
            return analysis_dir
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            self._handle_pipeline_failure(e)
            raise
            
    def _run_matrix_rendering(self) -> Path:
        """Run matrix rendering stage"""
        try:
            # Initialize matrices
            matrices = self.matrix_initializer.initialize_matrices(
                self.config['model']['hierarchy']
            )
            
            # Normalize matrices
            matrices = self.matrix_normalizer.normalize_matrices(matrices)
            
            # Save matrices
            matrices_dir = self.experiment['dirs']['matrices']
            for name, matrix in matrices.items():
                np.save(matrices_dir / f"{name}.npy", matrix)
                
            return matrices_dir
            
        except Exception as e:
            self.logger.error(f"Matrix rendering failed: {str(e)}")
            raise
            
    def _load_and_normalize_matrices(self, matrices_dir: Path) -> Dict[str, np.ndarray]:
        """Load and normalize matrices"""
        try:
            # Load matrices
            matrices = {}
            for matrix_file in matrices_dir.glob("*.npy"):
                name = matrix_file.stem
                matrix = np.load(matrix_file)
                matrices[name] = matrix
                
            # Normalize matrices
            normalized = self.matrix_normalizer.normalize_matrices(matrices)
            
            # Validate normalization
            is_valid, messages = self.matrix_normalizer.check_normalization(normalized)
            if not is_valid:
                raise ValueError(
                    f"Matrix normalization failed: {'; '.join(messages)}"
                )
                
            # Save normalized matrices
            for name, matrix in normalized.items():
                np.save(matrices_dir / f"{name}_normalized.npy", matrix)
                
            return normalized
            
        except Exception as e:
            self.logger.error(f"Error loading/normalizing matrices: {str(e)}")
            raise
            
    def _run_model_execution(self, matrices: Dict[str, np.ndarray]) -> Path:
        """Run model execution stage"""
        try:
            from rgm_execute import RGMExecutor
            
            executor = RGMExecutor(matrices)
            results_dir = executor.execute_inference()
            
            return results_dir
            
        except Exception as e:
            self.logger.error(f"Model execution failed: {str(e)}")
            raise
            
    def _run_results_analysis(self, results_dir: Path) -> Path:
        """Run results analysis stage"""
        try:
            from rgm_analyze import RGMAnalyzer
            
            analyzer = RGMAnalyzer(results_dir)
            analysis_dir = analyzer.analyze_results()
            
            return analysis_dir
            
        except Exception as e:
            self.logger.error(f"Results analysis failed: {str(e)}")
            raise
            
    def _handle_pipeline_failure(self, error: Exception):
        """Handle pipeline failure and save error state"""
        try:
            error_info = {
                'timestamp': datetime.now().isoformat(),
                'error_type': type(error).__name__,
                'error_message': str(error),
                'stage': getattr(self, '_current_stage', None),
                'memory_usage': self._get_peak_memory_usage()
            }
            
            # Save error state
            error_path = self.experiment['dirs']['logs'] / "pipeline_error.json"
            with open(error_path, 'w') as f:
                json.dump(error_info, f, indent=2)
                
            self.logger.info(f"Error state saved to: {error_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving error state: {str(e)}")
            
    def _get_peak_memory_usage(self) -> str:
        """Get peak memory usage"""
        try:
            import psutil
            process = psutil.Process()
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            total_memory = psutil.virtual_memory().total / 1024 / 1024  # MB
            percent = peak_memory / total_memory * 100
            
            return f"{peak_memory:.1f} MB ({percent:.1f}%)"
            
        except ImportError:
            return "Memory usage unavailable"
            
    def _log_directory_structure(self):
        """Log experiment directory structure"""
        self.logger.info("\nResults Directory Structure:")
        self.logger.info(f"📁 Root: {self.experiment['dirs']['root']}")
        self.logger.info(f"📊 ├─ render/")
        self.logger.info(f"🧠 ├─ simulation/")
        self.logger.info(f"📈 ├─ analysis/")
        self.logger.info(f"📝 └─ logs/")