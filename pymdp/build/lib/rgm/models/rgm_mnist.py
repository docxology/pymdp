"""RGM MNIST Model

Implementation of the RGM model for MNIST digit recognition.
"""

from typing import Tuple

import torch
import torch.nn as nn

from ..utils.rgm_logging import RGMLogging

class RGMMNISTModel(nn.Module):
    """
    RGM model for MNIST digit recognition.
    
    Features:
    - Hierarchical recognition and generative pathways
    - Lateral connections between levels
    - Batch normalization and dropout
    - Xavier/Glorot initialization
    
    Args:
        config: Model configuration dictionary
    """
    
    def __init__(self, config: dict):
        """Initialize the model."""
        super().__init__()
        self.logger = RGMLogging.get_logger("rgm.model")
        self._validate_config(config)
        self.config = config
        
        # Extract dimensions
        self.input_dim = config["architecture"]["input_dim"]
        self.hidden_dims = config["architecture"]["hidden_dims"]
        self.latent_dim = config["architecture"]["latent_dim"]
        
        # Setup layers
        self._setup_recognition_path()
        self._setup_generative_path()
        self._setup_lateral_connections()
        
        # Initialize weights
        self._initialize_weights()
        
        self.logger.info("✓ Model initialized successfully")
        
    def _validate_config(self, config: dict) -> None:
        """Validate model configuration."""
        required_keys = {
            "architecture": [
                "input_dim",
                "hidden_dims",
                "latent_dim"
            ],
            "training": [
                "dropout_rate",
                "batch_norm"
            ]
        }
        
        for section, keys in required_keys.items():
            if section not in config:
                raise ValueError(f"Missing config section: {section}")
            for key in keys:
                if key not in config[section]:
                    raise ValueError(
                        f"Missing config key: {section}.{key}"
                    )
                    
        # Validate dimensions
        if config["architecture"]["input_dim"] != 784:
            raise ValueError(
                f"Input dimension must be 784 for MNIST, "
                f"got {config['architecture']['input_dim']}"
            )
            
        if not isinstance(config["architecture"]["hidden_dims"], list):
            raise ValueError("hidden_dims must be a list")
            
        if any(d <= 0 for d in config["architecture"]["hidden_dims"]):
            raise ValueError("All hidden dimensions must be positive")
            
        if config["architecture"]["latent_dim"] <= 0:
            raise ValueError("latent_dim must be positive")
            
    def _setup_recognition_path(self) -> None:
        """Setup recognition pathway layers."""
        dims = [self.input_dim] + self.hidden_dims + [self.latent_dim]
        layers = []
        
        for i in range(len(dims)-1):
            # Linear layer
            layers.append(nn.Linear(dims[i], dims[i+1]))
            
            # Batch norm
            if self.config["training"]["batch_norm"]:
                layers.append(nn.BatchNorm1d(dims[i+1]))
                
            # Activation
            layers.append(nn.ReLU())
            
            # Dropout
            if i < len(dims)-2:  # No dropout on final layer
                layers.append(
                    nn.Dropout(self.config["training"]["dropout_rate"])
                )
                
        self.recognition_path = nn.Sequential(*layers)
        
    def _setup_generative_path(self) -> None:
        """Setup generative pathway layers."""
        dims = [self.latent_dim] + self.hidden_dims[::-1] + [self.input_dim]
        layers = []
        
        for i in range(len(dims)-1):
            # Linear layer
            layers.append(nn.Linear(dims[i], dims[i+1]))
            
            # Batch norm
            if self.config["training"]["batch_norm"]:
                layers.append(nn.BatchNorm1d(dims[i+1]))
                
            # Activation
            if i < len(dims)-2:
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Sigmoid())  # Final activation
                
            # Dropout
            if i < len(dims)-2:  # No dropout on final layer
                layers.append(
                    nn.Dropout(self.config["training"]["dropout_rate"])
                )
                
        self.generative_path = nn.Sequential(*layers)
        
    def _setup_lateral_connections(self) -> None:
        """Setup lateral connection matrices."""
        dims = [self.input_dim] + self.hidden_dims + [self.latent_dim]
        self.lateral_matrices = nn.ParameterList([
            nn.Parameter(torch.eye(d)) for d in dims
        ])
        
    def _initialize_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(
                    module.weight,
                    gain=nn.init.calculate_gain('relu')
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def _validate_batch(self, x: torch.Tensor) -> None:
        """
        Validate input batch dimensions.
        
        Args:
            x: Input batch tensor
            
        Raises:
            ValueError: If dimensions are invalid
        """
        if x.dim() != 2:
            raise ValueError(
                f"Expected 2D tensor [batch_size, {self.input_dim}], "
                f"got shape {x.shape}"
            )
            
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, "
                f"got {x.shape[1]}"
            )
            
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Tuple of (reconstruction, latent)
        """
        self._validate_batch(x)
        
        # Recognition path
        h = x
        activations = [h]
        for i, layer in enumerate(self.recognition_path):
            h = layer(h)
            if isinstance(layer, nn.Linear):
                h = torch.matmul(h, self.lateral_matrices[i])
                activations.append(h)
                
        latent = h
        
        # Generative path
        h = latent
        for i, layer in enumerate(self.generative_path):
            h = layer(h)
            if isinstance(layer, nn.Linear):
                h = torch.matmul(h, self.lateral_matrices[-(i+1)])
                
        reconstruction = h
        
        return reconstruction, latent 