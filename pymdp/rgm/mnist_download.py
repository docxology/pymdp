"""
MNIST Data Preparation
===================

Handles downloading and preprocessing of MNIST dataset.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class MNISTDataset:
    """Container for MNIST dataset splits and loaders."""
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_dataset: datasets.MNIST
    val_dataset: datasets.MNIST
    test_dataset: datasets.MNIST

class MNISTPreprocessor:
    """Handles MNIST dataset preparation."""
    
    def __init__(self, data_dir: Path, batch_size: int = 128):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
    def prepare_data(self) -> MNISTDataset:
        """
        Download and prepare MNIST dataset.
        
        Returns:
            MNISTDataset object containing data loaders and datasets
        """
        # Download training data
        train_full = datasets.MNIST(
            self.data_dir, train=True, download=True,
            transform=self.transform
        )
        
        # Split into train/val
        train_size = int(0.9 * len(train_full))
        val_size = len(train_full) - train_size
        train_dataset, val_dataset = random_split(
            train_full, [train_size, val_size]
        )
        
        # Download test data
        test_dataset = datasets.MNIST(
            self.data_dir, train=False,
            download=True, transform=self.transform
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        return MNISTDataset(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset
        )