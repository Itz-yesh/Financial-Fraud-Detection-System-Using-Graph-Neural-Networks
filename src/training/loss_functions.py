"""
Loss Functions
Custom loss functions for handling class imbalance
Reference: Tong et al. (2023) - HHLN-GNN for Imbalanced Financial Fraud Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy Loss for class imbalance
    Implements approach from Tong et al. (2023)
    """
    
    def __init__(self, pos_weight: float = 1.0):
        """
        Initialize weighted BCE loss
        
        Args:
            pos_weight: Weight for positive (fraud) class
        """
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
    
    def forward(self, 
                logits: torch.Tensor,
                targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute weighted BCE loss
        
        Args:
            logits: Model predictions [num_nodes, 2]
            targets: Ground truth labels [num_nodes]
            mask: Optional mask for selecting nodes
            
        Returns:
            Loss value
        """
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)[:, 1]  # Probability of fraud class
        
        # Apply mask if provided
        if mask is not None:
            probs = probs[mask]
            targets = targets[mask]
        
        # Compute weighted BCE
        targets = targets.float()
        loss = -(self.pos_weight * targets * torch.log(probs + 1e-8) + 
                (1 - targets) * torch.log(1 - probs + 1e-8))
        
        return loss.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for handling severe class imbalance
    Focuses on hard-to-classify examples
    """
    
    def __init__(self, 
                 alpha: float = 0.25,
                 gamma: float = 2.0):
        """
        Initialize focal loss
        
        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter (higher = more focus on hard examples)
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute focal loss
        
        Args:
            logits: Model predictions [num_nodes, 2]
            targets: Ground truth labels [num_nodes]
            mask: Optional mask for selecting nodes
            
        Returns:
            Loss value
        """
        # Apply mask if provided
        if mask is not None:
            logits = logits[mask]
            targets = targets[mask]
        
        # Compute cross-entropy
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Get probabilities
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=2).float()
        pt = (probs * targets_one_hot).sum(dim=1)
        
        # Compute focal loss
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting
        alpha_t = self.alpha * targets_one_hot[:, 1] + (1 - self.alpha) * targets_one_hot[:, 0]
        alpha_t = alpha_t.sum(dim=-1) if alpha_t.dim() > 1 else alpha_t
        
        loss = alpha_t * focal_weight * ce_loss
        
        return loss.mean()


class BalancedCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss with automatic class balancing
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Initialize balanced CE loss
        
        Args:
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(BalancedCrossEntropyLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute balanced cross-entropy loss
        
        Args:
            logits: Model predictions [num_nodes, 2]
            targets: Ground truth labels [num_nodes]
            mask: Optional mask for selecting nodes
            
        Returns:
            Loss value
        """
        # Apply mask if provided
        if mask is not None:
            logits = logits[mask]
            targets = targets[mask]
        
        # Compute class weights
        unique_classes, class_counts = torch.unique(targets, return_counts=True)
        total_samples = len(targets)
        
        # Inverse frequency weighting
        class_weights = total_samples / (len(unique_classes) * class_counts.float())
        
        # Create weight tensor
        weights = torch.zeros(2, device=logits.device)
        for cls, weight in zip(unique_classes, class_weights):
            weights[cls] = weight
        
        # Compute weighted cross-entropy
        loss = F.cross_entropy(logits, targets, weight=weights, reduction=self.reduction)
        
        return loss


def get_loss_function(loss_type: str, **kwargs) -> nn.Module:
    """
    Factory function to get loss function
    
    Args:
        loss_type: Type of loss ('weighted_bce', 'focal', 'balanced_ce', 'ce')
        **kwargs: Additional arguments for loss function
        
    Returns:
        Loss function module
    """
    if loss_type == 'weighted_bce':
        return WeightedBCELoss(**kwargs)
    elif loss_type == 'focal':
        return FocalLoss(**kwargs)
    elif loss_type == 'balanced_ce':
        return BalancedCrossEntropyLoss(**kwargs)
    elif loss_type == 'ce':
        return nn.CrossEntropyLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
