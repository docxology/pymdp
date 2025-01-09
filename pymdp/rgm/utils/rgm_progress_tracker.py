"""
RGM Progress Tracker
=================

Tracks and visualizes training progress for the Renormalization Generative Model.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import json
from datetime import datetime

class RGMProgressTracker:
    """Tracks training progress and generates visualizations."""
    
    def __init__(self, output_dir: Path):
        """Initialize progress tracker."""
        self.output_dir = output_dir
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_times': []
        }
        
    def update(self, epoch: int, metrics: Dict):
        """Update metrics history."""
        for key in self.metrics_history:
            if key in metrics:
                self.metrics_history[key].append(metrics[key])
                
        # Save metrics to file
        self._save_metrics(epoch, metrics)
        
        # Generate visualizations
        if epoch % 5 == 0:  # Update plots every 5 epochs
            self._generate_plots()
    
    def _save_metrics(self, epoch: int, metrics: Dict):
        """Save metrics to JSON file."""
        metrics_file = self.output_dir / 'metrics.json'
        
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                history = json.load(f)
        else:
            history = []
            
        metrics['epoch'] = epoch
        metrics['timestamp'] = datetime.now().isoformat()
        history.append(metrics)
        
        with open(metrics_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def _generate_plots(self):
        """Generate training progress visualizations."""
        plots_dir = self.output_dir / 'progress_plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Loss plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics_history['train_loss'], label='Train Loss')
        plt.plot(self.metrics_history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig(plots_dir / 'loss_history.png')
        plt.close()
        
        # Learning rate plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics_history['learning_rate'])
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig(plots_dir / 'lr_history.png')
        plt.close() 