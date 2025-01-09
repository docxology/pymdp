# RGM Model Training
=================

## Training Loop

The RGM training process consists of:

1. Forward Pass
   - Bottom-up recognition (input → states)
   - Top-down generation (states → predictions)
   - Error computation at each level
   - Loss aggregation across hierarchy

2. Loss Components
   - Reconstruction loss (MSE)
   - Prediction error loss
   - Sparsity regularization
   - Hierarchical consistency loss

3. Optimization
   - Adam optimizer with configurable parameters
   - Learning rate scheduling with ReduceLROnPlateau
   - Early stopping with patience
   - Gradient clipping for stability

4. Validation
   - Regular validation checks
   - Best model checkpointing
   - Performance monitoring
   - Early stopping decisions

## Configuration

Training parameters in `mnist_config.json`:
```json
{
    "training": {
        "n_epochs": 50,
        "batch_size": 128,
        "learning_rate": 0.001,
        "log_interval": 10000,
        "checkpoint_interval": 5,
        "validation_interval": 5,
        "plot_frequency": 5,
        "early_stopping": {
            "patience": 10,
            "min_delta": 0.001
        },
        "optimizer": {
            "type": "adam",
            "betas": [0.9, 0.999],
            "weight_decay": 1e-5
        },
        "scheduler": {
            "type": "plateau",
            "factor": 0.5,
            "patience": 5,
            "min_lr": 1e-6
        }
    }
}
```

## Progress Monitoring

Training progress is tracked via:
1. Loss curves (train/val)
2. Reconstruction quality
3. Learning rate changes
4. Validation metrics
5. Visual reconstructions
6. State activations
7. Feature maps

## Checkpointing

The trainer saves:
1. Model state dict
2. Optimizer state
3. Scheduler state
4. Best validation metrics
5. Training configuration
6. Current epoch/step
7. Loss history

## Visualization

During training, the system generates:
1. Input-reconstruction pairs
2. Loss component plots
3. Learning rate curves
4. State activation patterns
5. Feature visualizations
6. Hierarchical representations
7. Error distributions

## Error Handling

The training loop includes:
1. Exception catching and logging
2. Gradient anomaly detection
3. NaN loss checking
4. Memory monitoring
5. Device synchronization
6. Checkpoint recovery 