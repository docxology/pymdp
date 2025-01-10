"""
Learning Utilities
================

Utilities for structure and active learning.
"""

import torch
from typing import Dict, List, Tuple, Optional

class RGMLearningUtils:
    """Learning utility functions"""
    
    @staticmethod
    def fast_structure_learning(exemplars: Dict[int, torch.Tensor],
                              n_examples: int,
                              svd_config: Dict) -> Dict[str, torch.Tensor]:
        """Perform fast structure learning"""
        structure = {}
        
        for digit, examples in exemplars.items():
            # Process each example
            for example in examples[:n_examples]:
                # Get blocks
                blocks = RGMLearningUtils.get_blocks(example, svd_config)
                
                # Update structure
                RGMLearningUtils.update_structure(structure, blocks, digit)
                
        return structure
    
    @staticmethod
    def active_learning_update(image: torch.Tensor,
                             current_model: Dict,
                             beta: float = 512.0) -> Tuple[bool, float]:
        """Determine if active learning update should occur"""
        # Compute expected free energy
        G = RGMLearningUtils.compute_expected_free_energy(
            image, current_model, beta
        )
        
        # Compare with current free energy
        current_G = current_model.get('free_energy', float('inf'))
        should_update = G < current_G
        
        return should_update, G
    
    @staticmethod
    def update_structure(structure: Dict,
                        blocks: torch.Tensor,
                        digit: int):
        """Update structure with new blocks"""
        if digit not in structure:
            structure[digit] = []
            
        unique_states = RGMLearningUtils.get_unique_states(blocks)
        structure[digit].extend(unique_states)
    
    @staticmethod
    def get_unique_states(blocks: torch.Tensor) -> List[torch.Tensor]:
        """Get unique states from blocks"""
        states = blocks.reshape(blocks.shape[0], -1)
        unique = torch.unique(states, dim=0)
        return list(unique)
    
    @staticmethod
    def compute_expected_free_energy(image: torch.Tensor,
                                   model: Dict,
                                   beta: float) -> float:
        """Compute expected free energy"""
        # Compute accuracy
        accuracy = RGMLearningUtils.compute_accuracy(image, model)
        
        # Compute complexity
        complexity = RGMLearningUtils.compute_complexity(model)
        
        return beta * (complexity - accuracy)
    
    @staticmethod
    def compute_accuracy(image: torch.Tensor,
                        model: Dict,
                        eps: float = 1e-8) -> float:
        """Compute accuracy term"""
        pred = model['likelihood'] @ image.flatten()
        return torch.log(pred + eps).sum()
    
    @staticmethod
    def compute_complexity(model: Dict,
                         eps: float = 1e-8) -> float:
        """Compute complexity term"""
        prior = model.get('prior', torch.ones_like(model['likelihood']))
        kl = torch.sum(
            model['likelihood'] * (
                torch.log(model['likelihood'] + eps) -
                torch.log(prior + eps)
            )
        )
        return kl 