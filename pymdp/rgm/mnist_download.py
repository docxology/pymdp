"""
MNIST Data Downloader and Preprocessor
====================================

Downloads and preprocesses MNIST data for RGM pipeline.

This module:
1. Downloads MNIST dataset
2. Processes images to 32x32 RGB format
3. Creates balanced training subset
4. Generates exemplars for each digit
5. Saves processed data and visualizations
"""

import os
import sys
import torch
import logging
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
matplotlib.rcParams['font.family'] = ['DejaVu Sans']  # Use a font that supports emojis
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset

# Add parent directory to Python path for imports
file = Path(__file__).resolve()
parent, top = file.parent, file.parents[1]
if str(top) not in sys.path:
    sys.path.insert(0, str(top))

from rgm.utils import RGMLogging, RGMExperimentState

class MNISTPreprocessor:
    """Preprocesses MNIST data for RGM pipeline."""
    
    def __init__(self, data_dir: Path, exp_state: RGMExperimentState):
        """
        Initialize preprocessor.
        
        Args:
            data_dir: Directory to store MNIST data
            exp_state: Experiment state object
        """
        self.logger = RGMLogging.get_logger('mnist_preproc')
        self.data_dir = data_dir
        self.exp_state = exp_state
        
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup transform pipeline with better normalization
        self.transform = transforms.Compose([
            transforms.Resize((32, 32), antialiasing=True),  # Use antialiasing
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert to RGB
            transforms.Normalize(mean=[0.5], std=[0.5])  # Better normalization
        ])
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")

    def prepare_datasets(self, n_train: int = 10000, n_exemplars: int = 13) -> Dict:
        """
        Prepare MNIST datasets for RGM pipeline.
        
        Args:
            n_train: Number of training samples
            n_exemplars: Number of exemplars per class
            
        Returns:
            Dictionary containing processed datasets
        """
        try:
            self.logger.info("Downloading and processing MNIST data...")
            
            # Download MNIST dataset
            train_data = datasets.MNIST(self.data_dir, train=True, download=True, transform=self.transform)
            test_data = datasets.MNIST(self.data_dir, train=False, download=True, transform=self.transform)
            
            # Create balanced training subset
            train_subset = self._create_balanced_subset(train_data, n_train)
            
            # Generate exemplars
            exemplars = self._generate_exemplars(train_subset, n_exemplars)
            
            # Log dataset statistics
            self.logger.info(f"Training set size: {len(train_subset)}")
            self.logger.info(f"Test set size: {len(test_data)}")
            self.logger.info(f"Exemplars per class: {n_exemplars}")
            
            # Save dataset information
            self._save_dataset_info(train_subset, test_data, exemplars)
            
            # Generate visualizations
            self._generate_visualizations(train_subset, exemplars)
            
            return {
                'train': train_subset,
                'test': test_data,
                'exemplars': exemplars
            }
            
        except Exception as e:
            self.logger.error(f"Error preparing MNIST datasets: {str(e)}")
            raise

    def _create_balanced_subset(self, dataset: Dataset, n_samples: int) -> Dataset:
        """Create balanced subset of the dataset."""
        try:
            labels = dataset.targets.numpy()
            n_per_class = n_samples // 10
            
            indices = []
            for digit in range(10):
                digit_indices = np.where(labels == digit)[0]
                if len(digit_indices) < n_per_class:
                    raise ValueError(f"Not enough samples for digit {digit}")
                selected = np.random.choice(digit_indices, size=n_per_class, replace=False)
                indices.extend(selected)
            
            return Subset(dataset, indices)
            
        except Exception as e:
            self.logger.error(f"❌ Error selecting indices: {str(e)}")
            raise

    def _generate_exemplars(self, dataset: Dataset, n_per_class: int) -> Dict[int, torch.Tensor]:
        """Generate exemplars for each class."""
        try:
            images, labels = dataset.data, dataset.targets
            exemplars = {}
            
            for digit in range(10):
                # Get images for this digit
                digit_mask = labels == digit
                digit_images = images[digit_mask]
                
                if len(digit_images) < n_per_class:
                    raise ValueError(f"Not enough samples for digit {digit} exemplars")
                
                # Select first n examples
                exemplars[digit] = digit_images[:n_per_class]
            
            return exemplars
            
        except Exception as e:
            self.logger.error(f"❌ Error selecting exemplars: {str(e)}")
            raise

    def _save_dataset_info(self, train_subset: Dataset, test_data: Dataset, exemplars: Dict[int, torch.Tensor]) -> None:
        """Save dataset information."""
        try:
            # Save train subset
            torch.save(train_subset.data, self.data_dir / "train_data.pt")
            torch.save(train_subset.targets, self.data_dir / "train_labels.pt")
            
            # Save test data
            torch.save(test_data.data, self.data_dir / "test_data.pt")
            torch.save(test_data.targets, self.data_dir / "test_labels.pt")
            
            # Save exemplars
            torch.save(exemplars, self.data_dir / "exemplars.pt")
            
            self.logger.info("\n💾 Saved dataset information")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving dataset information: {str(e)}")
            raise

    def _generate_visualizations(self, train_subset: Dataset, exemplars: Dict[int, torch.Tensor]) -> None:
        """Generate visualizations of the data."""
        try:
            self.logger.info("\nGenerating visualizations...")
            
            # Use a clean style
            plt.style.use('seaborn-v0_8')
            
            # Example digits - show one from each class
            fig = plt.figure(figsize=(15, 6))
            fig.suptitle("MNIST Examples - One Per Class", fontsize=12, y=1.05)
            
            # Find one example of each digit
            for digit in range(10):
                ax = plt.subplot(2, 5, digit + 1)
                digit_mask = train_subset.targets == digit
                digit_image = train_subset.data[digit_mask][0][0]  # First image of this digit, first channel
                
                # Normalize image for display
                img_min, img_max = digit_image.min(), digit_image.max()
                digit_image = (digit_image - img_min) / (img_max - img_min)
                
                ax.imshow(digit_image.cpu(), cmap='gray', interpolation='nearest')
                ax.axis('off')
                ax.set_title(f'Digit {digit}', pad=5)
            
            plt.tight_layout()
            examples_path = self.data_dir / "examples.png"
            plt.savefig(examples_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            self.logger.info(f"Saved example digits to {examples_path}")
            
            # Class distribution
            plt.figure(figsize=(12, 6))
            values, counts = torch.unique(train_subset.targets, return_counts=True)
            values, counts = values.cpu().numpy(), counts.cpu().numpy()
            
            # Create bar plot with better colors
            colors = plt.cm.viridis(np.linspace(0, 1, len(values)))
            bars = plt.bar(values, counts, color=colors, edgecolor='black', alpha=0.7)
            plt.title("MNIST Class Distribution", pad=20, fontsize=12)
            plt.xlabel("Digit Class", fontsize=10)
            plt.ylabel("Number of Samples", fontsize=10)
            
            # Add count labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom')
            
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.xticks(values)
            
            dist_path = self.data_dir / "distribution.png"
            plt.savefig(dist_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            self.logger.info(f"Saved class distribution to {dist_path}")
            
            # Show exemplars grid with improved layout
            n_classes = len(exemplars)
            n_exemplars = len(exemplars[0])
            
            # Create figure with better proportions
            fig = plt.figure(figsize=(2*n_exemplars, 2*n_classes))
            fig.suptitle("MNIST Exemplars", fontsize=12, y=1.02)
            
            # Add exemplars with improved normalization
            for digit in range(n_classes):
                exemplars_digit = exemplars[digit]
                for j in range(n_exemplars):
                    ax = plt.subplot(n_classes, n_exemplars, digit*n_exemplars + j + 1)
                    
                    # Normalize each exemplar
                    img = exemplars_digit[j][0].cpu()
                    img = (img - img.min()) / (img.max() - img.min())
                    
                    ax.imshow(img, cmap='gray', interpolation='nearest')
                    ax.axis('off')
                    
                    # Add digit label only for first column
                    if j == 0:
                        ax.set_title(f'Digit {digit}', x=-0.5, y=0.5)
            
            plt.tight_layout()
            exemplars_path = self.data_dir / "exemplars.png"
            plt.savefig(exemplars_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            self.logger.info(f"Saved exemplars grid to {exemplars_path}")
            
            # Log visualization summary
            self.logger.info("\nGenerated visualizations:")
            self.logger.info(f"- Examples: {examples_path}")
            self.logger.info(f"- Distribution: {dist_path}")
            self.logger.info(f"- Exemplars: {exemplars_path}")
            
        except Exception as e:
            self.logger.error(f"Visualization error: {str(e)}")
            self.logger.debug("Traceback:", exc_info=True)
            self.logger.warning("Continuing without visualizations...")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Download and preprocess MNIST dataset')
    parser.add_argument('--data-dir', type=str, default='data/mnist', help='Data directory')
    parser.add_argument('--exp-dir', type=str, help='Experiment directory')
    parser.add_argument('--n-train', type=int, default=10000, help='Number of training samples')
    parser.add_argument('--n-exemplars', type=int, default=13, help='Number of exemplars per class')
    args = parser.parse_args()
    
    try:
        # Setup experiment state if directory provided
        exp_state = RGMExperimentState(Path(args.exp_dir)) if args.exp_dir else None
        
        # Initialize and run preprocessor
        preprocessor = MNISTPreprocessor(data_dir=Path(args.data_dir), exp_state=exp_state)
        preprocessor.prepare_datasets(n_train=args.n_train, n_exemplars=args.n_exemplars)
        
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()