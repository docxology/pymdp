"""
MNIST Data Downloader and Preprocessor
===================================

Downloads and preprocesses MNIST data for RGM pipeline.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

# Import local modules
from local_imports import check_dependencies, import_rgm_modules

# Check dependencies first
check_dependencies()

# Import RGM modules
rgm_modules = import_rgm_modules()
RGMDataUtils = rgm_modules['RGMDataUtils']
RGMConfigUtils = rgm_modules['RGMConfigUtils']
RGMExperimentUtils = rgm_modules['RGMExperimentUtils']

class MNISTPreprocessor:
    """Preprocesses MNIST data for RGM pipeline"""
    
    def __init__(self, 
                 data_dir: Optional[Path] = None,
                 config_path: Optional[Path] = None):
        """Initialize preprocessor"""
        # Setup experiment
        self.exp_dirs = RGMExperimentUtils.setup_experiment("mnist_preprocessing")
        self.logger = RGMExperimentUtils.setup_logging(
            self.exp_dirs['logs'],
            "mnist_preprocessor"
        )
        
        # Setup paths
        self.data_dir = data_dir or Path("data/mnist")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load config
        self.config = self._load_config(config_path)
        
        # Setup transform pipeline
        self.transform = self._create_transform()
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
    def _load_config(self, config_path: Optional[Path]) -> dict:
        """Load configuration"""
        if config_path and config_path.exists():
            return RGMConfigUtils.load_config(config_path)
        return {
            "data": {
                "mnist": {
                    "image_format": {
                        "input_size": [28, 28],
                        "output_size": [32, 32],
                        "channels": 3,
                        "dtype": "float32",
                        "range": [0, 1]
                    },
                    "training": {
                        "total_samples": 10000,
                        "exemplars_per_class": 13
                    }
                }
            },
            "preprocessing": {
                "histogram_equalization": True,
                "clip_limit": 0.03,
                "gaussian_smoothing": True,
                "kernel_size": [2, 2],
                "sigma": 1.0
            }
        }
        
    def _create_transform(self) -> transforms.Compose:
        """Create preprocessing transform pipeline"""
        def to_rgb(x: torch.Tensor) -> torch.Tensor:
            """Convert grayscale to RGB"""
            return x.repeat(3, 1, 1)
            
        def preprocess_batch(x: torch.Tensor) -> torch.Tensor:
            """Apply RGM preprocessing"""
            return RGMDataUtils.preprocess_batch(
                x.unsqueeze(0),
                self.config
            ).squeeze(0)
        
        return transforms.Compose([
            # Resize to target size
            transforms.Resize(self.config['data']['mnist']['image_format']['output_size']),
            transforms.ToTensor(),
            transforms.Lambda(to_rgb),
            transforms.Lambda(preprocess_batch)
        ])
        
    def prepare_datasets(self) -> Dict[str, Dataset]:
        """Prepare datasets for RGM pipeline"""
        try:
            self.logger.info("Preparing MNIST datasets...")
            
            # Download base datasets
            train_full = datasets.MNIST(
                root=str(self.data_dir),
                train=True,
                download=True
            )
            
            test_full = datasets.MNIST(
                root=str(self.data_dir),
                train=False,
                download=True
            )
            
            # Select training subset
            train_indices = self._select_balanced_indices(
                train_full,
                self.config['data']['mnist']['training']['total_samples']
            )
            
            # Process datasets
            train_data = self._process_dataset(
                datasets.MNIST(
                    root=str(self.data_dir),
                    train=True
                ),
                train_indices
            )
            
            test_data = self._process_dataset(
                datasets.MNIST(
                    root=str(self.data_dir),
                    train=False
                )
            )
            
            # Select exemplars
            exemplars = self._select_exemplars(
                train_data,
                self.config['data']['mnist']['training']['exemplars_per_class']
            )
            
            # Create processed datasets
            processed_data = {
                'train': {
                    'images': train_data[0],
                    'labels': train_data[1]
                },
                'test': {
                    'images': test_data[0],
                    'labels': test_data[1]
                },
                'exemplars': exemplars
            }
            
            # Save preprocessed data
            self._save_preprocessed_data(processed_data)
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error preparing datasets: {str(e)}")
            raise
            
    def _process_dataset(self,
                        dataset: Dataset,
                        indices: Optional[List[int]] = None,
                        desc: str = "Processing batches") -> Tuple[torch.Tensor, torch.Tensor]:
        """Process dataset with transforms"""
        try:
            # First apply transform to dataset
            transformed_dataset = datasets.MNIST(
                root=str(self.data_dir),
                train=dataset.train,
                transform=self.transform,
                download=False
            )
            
            if indices is not None:
                transformed_dataset = Subset(transformed_dataset, indices)
            
            dataloader = DataLoader(
                transformed_dataset,
                batch_size=100,
                shuffle=False,
                num_workers=4,
                pin_memory=True if torch.cuda.is_available() else False
            )
            
            processed_images = []
            labels = []
            
            # Use tqdm with dynamic description
            with tqdm(total=len(dataloader), desc=desc) as pbar:
                for batch_idx, (images, batch_labels) in enumerate(dataloader):
                    images = images.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    processed_images.append(images)
                    labels.append(batch_labels)
                    
                    # Update progress bar with current batch size
                    pbar.set_postfix({"batch_size": len(images)})
                    pbar.update(1)
                    
            all_images = torch.cat(processed_images)
            all_labels = torch.cat(labels)
            
            self.logger.info(f"Processed {len(all_images)} images with shape {all_images.shape}")
            return all_images, all_labels
            
        except Exception as e:
            self.logger.error(f"Error processing dataset: {str(e)}")
            self.logger.error("Traceback:", exc_info=True)
            raise
        
    def _select_balanced_indices(self,
                               dataset: Dataset,
                               n_samples: int) -> List[int]:
        """Select balanced subset of indices"""
        labels = dataset.targets.numpy()
        
        # Calculate samples per class
        n_classes = len(set(labels))
        samples_per_class = n_samples // n_classes
        
        selected_indices = []
        for class_idx in range(n_classes):
            # Get indices for this class
            class_indices = np.where(labels == class_idx)[0]
            
            # Randomly select indices
            selected = np.random.choice(
                class_indices,
                size=samples_per_class,
                replace=False
            )
            
            selected_indices.extend(selected)
            
        return selected_indices
        
    def _select_exemplars(self,
                         data: Tuple[torch.Tensor, torch.Tensor],
                         n_per_class: int) -> Dict[int, torch.Tensor]:
        """Select exemplars for each class"""
        try:
            images, labels = data
            exemplars = {}
            
            self.logger.info(f"Selecting {n_per_class} exemplars per class...")
            
            for digit in range(10):
                # Get indices for this digit
                digit_mask = labels == digit
                digit_images = images[digit_mask]
                
                # Select first n examples
                if len(digit_images) < n_per_class:
                    self.logger.warning(f"Only {len(digit_images)} examples available for digit {digit}")
                    exemplars[digit] = digit_images
                else:
                    exemplars[digit] = digit_images[:n_per_class]
                
                self.logger.info(f"Selected {len(exemplars[digit])} exemplars for digit {digit}")
            
            return exemplars
            
        except Exception as e:
            self.logger.error(f"Error selecting exemplars: {str(e)}")
            self.logger.error("Traceback:", exc_info=True)
            raise
        
    def _save_preprocessed_data(self, data: Dict):
        """Save preprocessed datasets"""
        try:
            save_path = self.data_dir / "preprocessed"
            save_path.mkdir(exist_ok=True)
            
            # Validate data shapes
            expected_shape = tuple(self.config['data']['mnist']['image_format']['output_size'])
            actual_shape = data['train']['images'].shape[2:]
            
            if actual_shape != expected_shape:
                self.logger.warning(
                    f"Image shape mismatch: expected {expected_shape}, got {actual_shape}"
                )
            
            # Save each component separately
            self.logger.info("Saving preprocessed data...")
            
            # Save training data
            train_path = save_path / "train.pt"
            torch.save(data['train'], train_path)
            self.logger.info(f"Saved training data: {train_path}")
            
            # Save test data
            test_path = save_path / "test.pt"
            torch.save(data['test'], test_path)
            self.logger.info(f"Saved test data: {test_path}")
            
            # Save exemplars
            exemplars_path = save_path / "exemplars.pt"
            torch.save(data['exemplars'], exemplars_path)
            self.logger.info(f"Saved exemplars: {exemplars_path}")
            
            # Save metadata with validation info
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'config': self.config,
                'sizes': {
                    'train': len(data['train']['images']),
                    'test': len(data['test']['images']),
                    'exemplars': {
                        str(k): len(v) for k, v in data['exemplars'].items()
                    }
                },
                'shapes': {
                    'train': list(data['train']['images'].shape),
                    'test': list(data['test']['images'].shape)
                },
                'device': str(self.device),
                'validation': {
                    'shape_match': actual_shape == expected_shape,
                    'value_range': {
                        'min': float(data['train']['images'].min()),
                        'max': float(data['train']['images'].max())
                    },
                    'class_balance': {
                        str(k): int((data['train']['labels'] == k).sum())
                        for k in range(10)
                    }
                }
            }
            
            metadata_path = save_path / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            self.logger.info(f"Saved metadata: {metadata_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving preprocessed data: {str(e)}")
            self.logger.error("Traceback:", exc_info=True)
            raise

def main():
    """Main execution function"""
    try:
        # Parse arguments
        parser = argparse.ArgumentParser(
            description='Download and preprocess MNIST dataset for RGM'
        )
        parser.add_argument(
            '--data-dir',
            type=str,
            default='data/mnist',
            help='Directory to store dataset'
        )
        parser.add_argument(
            '--config',
            type=str,
            default=None,
            help='Path to configuration file'
        )
        parser.add_argument(
            '--num-workers',
            type=int,
            default=4,
            help='Number of worker processes for data loading'
        )
        args = parser.parse_args()
        
        # Initialize preprocessor
        preprocessor = MNISTPreprocessor(
            data_dir=Path(args.data_dir),
            config_path=Path(args.config) if args.config else None
        )
        
        # Prepare datasets
        datasets = preprocessor.prepare_datasets()
        
        # Log dataset sizes
        logger.info("\nDataset preparation complete:")
        logger.info(f"Training samples: {len(datasets['train']['images'])}")
        logger.info(f"Test samples: {len(datasets['test']['images'])}")
        logger.info(f"Exemplars: {sum(len(x) for x in datasets['exemplars'].values())}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()