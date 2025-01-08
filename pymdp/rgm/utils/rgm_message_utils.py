"""
Message Passing Utilities
=======================

Utilities for hierarchical message passing.
"""

import torch
from typing import List, Dict, Optional, Tuple

class RGMMessageUtils:
    """Message passing utility functions"""
    
    @staticmethod
    def propagate_beliefs(beliefs: List[torch.Tensor],
                         observations: torch.Tensor,
                         likelihood_maps: Dict[str, torch.Tensor],
                         max_iters: int = 50,
                         tolerance: float = 1e-6) -> List[torch.Tensor]:
        """Perform full belief propagation"""
        for _ in range(max_iters):
            prev_beliefs = [b.clone() for b in beliefs]
            
            # Bottom-up pass
            beliefs = RGMMessageUtils.bottom_up_pass(
                beliefs, observations, likelihood_maps
            )
            
            # Top-down pass
            beliefs = RGMMessageUtils.top_down_pass(
                beliefs, likelihood_maps
            )
            
            # Check convergence
            if RGMMessageUtils.check_convergence(beliefs, prev_beliefs, tolerance):
                break
                
        return beliefs
    
    @staticmethod
    def bottom_up_pass(beliefs: List[torch.Tensor],
                      observations: torch.Tensor,
                      likelihood_maps: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """Perform bottom-up message passing"""
        updated_beliefs = []
        msg = observations
        
        for level, belief in enumerate(beliefs):
            # Compute bottom-up message
            msg = RGMMessageUtils.compute_bottom_up_message(
                msg, likelihood_maps[f'D{level}']
            )
            
            # Update belief
            updated_belief = RGMMessageUtils.update_belief(
                belief, msg, None  # No top-down message yet
            )
            
            updated_beliefs.append(updated_belief)
            
        return updated_beliefs
    
    @staticmethod
    def top_down_pass(beliefs: List[torch.Tensor],
                     likelihood_maps: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """Perform top-down message passing"""
        updated_beliefs = beliefs.copy()
        msg = None
        
        for level in reversed(range(len(beliefs))):
            # Compute top-down message
            if msg is not None:
                msg = RGMMessageUtils.compute_top_down_message(
                    msg, likelihood_maps[f'D{level+1}']
                )
                
                # Update belief
                updated_beliefs[level] = RGMMessageUtils.update_belief(
                    beliefs[level], None, msg  # No bottom-up message
                )
            
            msg = beliefs[level]
            
        return updated_beliefs
    
    @staticmethod
    def compute_bottom_up_message(source: torch.Tensor,
                                likelihood: torch.Tensor,
                                eps: float = 1e-8) -> torch.Tensor:
        """Compute bottom-up message"""
        return torch.log(likelihood + eps) @ source
    
    @staticmethod
    def compute_top_down_message(source: torch.Tensor,
                               likelihood: torch.Tensor,
                               eps: float = 1e-8) -> torch.Tensor:
        """Compute top-down message"""
        return likelihood.t() @ source
    
    @staticmethod
    def update_belief(belief: torch.Tensor,
                     bottom_up_msg: Optional[torch.Tensor] = None,
                     top_down_msg: Optional[torch.Tensor] = None,
                     eps: float = 1e-8) -> torch.Tensor:
        """Update belief using messages"""
        log_belief = torch.log(belief + eps)
        
        if bottom_up_msg is not None:
            log_belief = log_belief + bottom_up_msg
            
        if top_down_msg is not None:
            log_belief = log_belief + top_down_msg
            
        belief = torch.exp(log_belief)
        return belief / belief.sum()
    
    @staticmethod
    def check_convergence(current: List[torch.Tensor],
                         previous: List[torch.Tensor],
                         tolerance: float) -> bool:
        """Check belief convergence"""
        diffs = [torch.abs(c - p).max() for c, p in zip(current, previous)]
        return all(d < tolerance for d in diffs) 