"""
RGM Model Evaluator
=================

Handles evaluation and analysis of trained RGM models.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from sklearn.metrics import confusion_matrix
import seaborn as sns

from ..utils.rgm_logging import RGMLogging
from ..utils.visualization_utils import RGMVisualizationUtils

class RGMEvaluator:
    """Evaluates trained RGM models."""
    
    def __init__(
        self,
        model,
        test_loader: DataLoader,
        exp_dir: Path,
        device: Optional[torch.device] = None
    ):
        """Initialize evaluator."""
        self.logger = RGMLogging.get_logger("rgm.evaluator")
        self.model = model
        self.test_loader = test_loader
        self.exp_dir = exp_dir
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create evaluation directory
        self.eval_dir = exp_dir / "evaluation"
        self.eval_dir.mkdir(exist_ok=True, parents=True)
        
    def evaluate(self):
        """Run complete evaluation."""
        self.logger.info("Starting model evaluation...")
        self.model.eval()
        
        # Collect predictions and targets
        all_preds = []
        all_targets = []
        reconstruction_error = 0.0
        
        with torch.no_grad():
            for data, targets in self.test_loader:
                data = data.to(self.device)
                outputs = self.model(data)
                
                # Store predictions and targets
                preds = outputs['output'].argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.numpy())
                
                # Calculate reconstruction error
                reconstruction_error += F.mse_loss(
                    outputs['predictions']['level0'],
                    data
                ).item()
                
                # Visualize sample reconstructions
                if len(all_preds) <= 100:  # Only first batch
                    self._visualize_reconstructions(
                        data,
                        outputs['predictions']['level0'],
                        targets,
                        batch_idx=len(all_preds)//len(data)
                    )
                    
        # Calculate metrics
        reconstruction_error /= len(self.test_loader)
        accuracy = np.mean(np.array(all_preds) == np.array(all_targets))
        
        # Log results
        self.logger.info("\n📊 Evaluation Results:")
        self.logger.info(f"   • Accuracy: {accuracy:.4f}")
        self.logger.info(f"   • Reconstruction Error: {reconstruction_error:.4f}")
        
        # Generate confusion matrix
        self._plot_confusion_matrix(all_targets, all_preds)
        
        self.logger.info(f"\n💾 Evaluation results saved to: {self.eval_dir}")
        
    def _visualize_reconstructions(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        labels: torch.Tensor,
        batch_idx: int
    ):
        """Visualize input-reconstruction pairs."""
        save_path = self.eval_dir / f"reconstructions_batch_{batch_idx:03d}.png"
        RGMVisualizationUtils.plot_mnist_grid(
            images=inputs,
            reconstructions=reconstructions,
            save_path=save_path,
            title=f"Test Set Reconstructions (Batch {batch_idx})"
        )
        
    def _plot_confusion_matrix(self, targets: list, predictions: list):
        """Generate and save confusion matrix plot."""
        cm = confusion_matrix(targets, predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=range(10),
            yticklabels=range(10)
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(self.eval_dir / "confusion_matrix.png")
        plt.close() 