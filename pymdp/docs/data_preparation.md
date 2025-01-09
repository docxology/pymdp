# RGM Data Preparation
===================

## MNIST Dataset

The RGM pipeline uses the MNIST dataset for training and evaluation. The data preparation stage handles:

1. Dataset Download
   - Automatic download if not present
   - Verification of downloaded files
   - Storage in experiment directory

2. Data Preprocessing
   - Normalization (mean=0.1307, std=0.3081)
   - Train/validation split (90%/10%)
   - Tensor conversion
   - Batch preparation

3. Data Loading
   - DataLoader configuration
   - Multi-worker loading
   - GPU pinned memory (optional)
   - Shuffling for training

## Configuration

Data preparation is configured in `mnist_config.json`:

```json
{
    "data": {
        "batch_size": 128,
        "train_val_split": 0.9,
        "num_workers": 4,
        "pin_memory": true
    }
}
```

## Directory Structure

```
experiments/
└── rgm_mnist_pipeline_{timestamp}/
    └── data/
        └── MNIST/
            ├── raw/              # Downloaded files
            ├── processed/        # Preprocessed tensors
            └── splits/           # Train/val/test splits
```

## Logging

The data preparation stage logs:
1. Download progress
2. Dataset statistics
3. Data directory location
4. Preprocessing steps
5. DataLoader configuration

## Error Handling

The stage includes:
1. Download retry logic
2. Checksum verification
3. Disk space checks
4. Corrupted file detection
5. Informative error messages 