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
    
    def __init__(self, 
                 data_dir: Optional[Path] = None,
                 exp_state: Optional[RGMExperimentState] = None):
        """Initialize preprocessor."""
        self.exp_state = exp_state
        self.logger = RGMLogging.get_logger("rgm.mnist")
        
        # Setup paths
        self.data_dir = Path(data_dir or "data/mnist")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup transform pipeline with better normalization
        self.transform = transforms.Compose([
            transforms.Resize((32, 32), antialias=True),  # Use antialiasing
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert to RGB
            transforms.Normalize(mean=[0.5], std=[0.5])  # Better normalization
        ])
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")

    def prepare_datasets(self, n_train: int = 10000, n_exemplars: int = 13) -> Dict[str, Dataset]:
        """Prepare MNIST datasets."""
        try:
            self.logger.info("\n" + "="*50)
            self.logger.info("🚀 Starting MNIST dataset preparation")
            self.logger.info("="*50)
            
            # Check if data already exists
            processed_dir = self.data_dir / "processed"
            if processed_dir.exists() and self._check_processed_data(processed_dir):
                self.logger.info("\n♻️  Found valid processed data, loading...")
                return self._load_processed_data(processed_dir)
            
            # Download datasets
            self.logger.info("\n📥 Downloading MNIST dataset...")
            train_full = datasets.MNIST(root=str(self.data_dir), train=True, download=True)
            test_full = datasets.MNIST(root=str(self.data_dir), train=False, download=True)
            
            self.logger.info(f"✅ Downloaded {len(train_full)} training and {len(test_full)} test samples")
            
            # Validate parameters
            n_train = min(n_train, len(train_full))
            n_train = (n_train // 10) * 10  # Ensure divisible by number of classes
            n_exemplars = min(n_exemplars, n_train // 10)
            
            # Create balanced training subset
            train_indices = self._select_balanced_indices(train_full, n_train)
            
            # Process datasets
            train_data = self._process_dataset(train_full, train_indices, "🔄 Processing training data")
            test_data = self._process_dataset(test_full, None, "🔄 Processing test data")
            
            # Validate processed data
            self._validate_data(train_data[0], "training")
            self._validate_data(test_data[0], "test")
            
            # Select exemplars
            exemplars = self._select_exemplars(train_data, n_exemplars)
            
            # Create processed datasets
            processed_data = {
                'train': {'images': train_data[0], 'labels': train_data[1]},
                'test': {'images': test_data[0], 'labels': test_data[1]},
                'exemplars': exemplars
            }
            
            # Save processed data
            self._save_processed_data(processed_data, processed_dir)
            
            # Generate visualizations
            vis_dir = self.data_dir / "visualizations"
            vis_dir.mkdir(exist_ok=True)
            self._visualize_data(processed_data, vis_dir)
            
            # Update experiment state
            if self.exp_state:
                self._update_experiment_state(processed_data, vis_dir)
            
            self.logger.info("\n" + "="*50)
            self.logger.info("✨ Dataset preparation complete!")
            self.logger.info(f"📊 Training samples: {len(train_data[0])}")
            self.logger.info(f"📊 Test samples: {len(test_data[0])}")
            self.logger.info(f"🎯 Exemplars per class: {n_exemplars}")
            self.logger.info("="*50)
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"\n❌ Error: {str(e)}")
            if self.exp_state:
                self.exp_state.update({
                    'mnist': {
                        'status': 'failed',
                        'error': str(e)
                    }
                })
            raise

    def _process_dataset(self, dataset: Dataset, indices: Optional[List[int]] = None, desc: str = "") -> Tuple[torch.Tensor, torch.Tensor]:
        """Process dataset with transforms."""
        try:
            # Create transformed dataset with better error handling
            transformed_dataset = datasets.MNIST(
                root=str(self.data_dir),
                train=dataset.train,
                transform=self.transform,
                download=False
            )
            
            # Use subset if indices provided
            if indices is not None:
                transformed_dataset = Subset(transformed_dataset, indices)
            
            # Create dataloader with better settings
            dataloader = DataLoader(
                transformed_dataset,
                batch_size=100,
                shuffle=False,
                num_workers=min(4, os.cpu_count() or 1),  # Adaptive number of workers
                pin_memory=True if torch.cuda.is_available() else False,
                prefetch_factor=2
            )
            
            # Process batches with better memory handling
            images, labels = [], []
            total_samples = len(transformed_dataset)
            
            with tqdm(total=total_samples, desc=desc) as pbar:
                for batch_images, batch_labels in dataloader:
                    images.append(batch_images)
                    labels.append(batch_labels)
                    pbar.update(batch_images.size(0))
            
            # Concatenate all batches
            all_images = torch.cat(images)
            all_labels = torch.cat(labels)
            
            self.logger.info(f"Processed {len(all_images)} images with shape {all_images.shape}")
            return all_images, all_labels
            
        except Exception as e:
            self.logger.error(f"Error processing dataset: {str(e)}")
            raise

    def _validate_data(self, images: torch.Tensor, name: str) -> None:
        """Validate processed data."""
        try:
            if images.dim() != 4:
                raise ValueError(f"{name} images must be 4D (N,C,H,W)")
            if images.shape[1] != 3:
                raise ValueError(f"{name} images must have 3 channels")
            if images.shape[2:] != (32, 32):
                raise ValueError(f"{name} images must be 32x32")
            if not torch.isfinite(images).all():
                raise ValueError(f"{name} images contain invalid values")
        except Exception as e:
            self.logger.error(f"❌ Data validation error: {str(e)}")
            raise

    def _select_balanced_indices(self, dataset: Dataset, n_samples: int) -> List[int]:
        """Select balanced subset of indices."""
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
            
            return indices
            
        except Exception as e:
            self.logger.error(f"❌ Error selecting indices: {str(e)}")
            raise

    def _select_exemplars(self, data: Tuple[torch.Tensor, torch.Tensor], n_per_class: int) -> Dict[int, torch.Tensor]:
        """Select exemplars for each class."""
        try:
            images, labels = data
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

    def _save_processed_data(self, data: Dict, save_dir: Path) -> None:
        """Save processed data."""
        try:
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save tensors
            torch.save(data['train'], save_dir / "train.pt")
            torch.save(data['test'], save_dir / "test.pt")
            torch.save(data['exemplars'], save_dir / "exemplars.pt")
            
            self.logger.info(f"\n💾 Saved processed data to {save_dir}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving data: {str(e)}")
            raise

    def _load_processed_data(self, data_dir: Path) -> Dict:
        """Load processed data."""
        try:
            data = {
                'train': torch.load(data_dir / "train.pt"),
                'test': torch.load(data_dir / "test.pt"),
                'exemplars': torch.load(data_dir / "exemplars.pt")
            }
            
            # Validate loaded data
            self._validate_data(data['train']['images'], "training")
            self._validate_data(data['test']['images'], "test")
            
            self.logger.info(f"✅ Loaded processed data from {data_dir}")
            return data
            
        except Exception as e:
            self.logger.error(f"❌ Error loading data: {str(e)}")
            raise

    def _check_processed_data(self, data_dir: Path) -> bool:
        """Check if processed data exists and is valid."""
        try:
            return all((data_dir / f"{name}.pt").exists() 
                      for name in ["train", "test", "exemplars"])
        except Exception:
            return False

    def _update_experiment_state(self, data: Dict, vis_dir: Path) -> None:
        """Update experiment state with dataset info."""
        try:
            train_images = data['train']['images']
            self.exp_state.update({
                'mnist': {
                    'status': 'complete',
                    'train_samples': len(train_images),
                    'test_samples': len(data['test']['images']),
                    'exemplars': sum(len(x) for x in data['exemplars'].values()),
                    'image_shape': list(train_images.shape[1:]),
                    'data_stats': {
                        'min': float(train_images.min()),
                        'max': float(train_images.max()),
                        'mean': float(train_images.mean()),
                        'std': float(train_images.std())
                    },
                    'visualizations': {
                        'examples': str(vis_dir / "examples.png"),
                        'distribution': str(vis_dir / "distribution.png"),
                        'exemplars': str(vis_dir / "exemplars.png")
                    }
                }
            })
            self.logger.info("\n📝 Updated experiment state")
        except Exception as e:
            self.logger.error(f"❌ Error updating experiment state: {str(e)}")
            raise

    def _visualize_data(self, data: Dict, save_dir: Path) -> None:
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
                digit_mask = data['train']['labels'] == digit
                digit_image = data['train']['images'][digit_mask][0][0]  # First image of this digit, first channel
                
                # Normalize image for display
                img_min, img_max = digit_image.min(), digit_image.max()
                digit_image = (digit_image - img_min) / (img_max - img_min)
                
                ax.imshow(digit_image.cpu(), cmap='gray', interpolation='nearest')
                ax.axis('off')
                ax.set_title(f'Digit {digit}', pad=5)
            
            plt.tight_layout()
            examples_path = save_dir / "examples.png"
            plt.savefig(examples_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            self.logger.info(f"Saved example digits to {examples_path}")
            
            # Class distribution
            plt.figure(figsize=(12, 6))
            values, counts = torch.unique(data['train']['labels'], return_counts=True)
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
            
            dist_path = save_dir / "distribution.png"
            plt.savefig(dist_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            self.logger.info(f"Saved class distribution to {dist_path}")
            
            # Show exemplars grid with improved layout
            n_classes = len(data['exemplars'])
            n_exemplars = len(data['exemplars'][0])
            
            # Create figure with better proportions
            fig = plt.figure(figsize=(2*n_exemplars, 2*n_classes))
            fig.suptitle("MNIST Exemplars", fontsize=12, y=1.02)
            
            # Add exemplars with improved normalization
            for digit in range(n_classes):
                exemplars = data['exemplars'][digit]
                for j in range(n_exemplars):
                    ax = plt.subplot(n_classes, n_exemplars, digit*n_exemplars + j + 1)
                    
                    # Normalize each exemplar
                    img = exemplars[j][0].cpu()
                    img = (img - img.min()) / (img.max() - img.min())
                    
                    ax.imshow(img, cmap='gray', interpolation='nearest')
                    ax.axis('off')
                    
                    # Add digit label only for first column
                    if j == 0:
                        ax.set_title(f'Digit {digit}', x=-0.5, y=0.5)
            
            plt.tight_layout()
            exemplars_path = save_dir / "exemplars.png"
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