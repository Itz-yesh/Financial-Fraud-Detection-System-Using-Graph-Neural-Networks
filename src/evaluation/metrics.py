"""
Evaluation Metrics
Comprehensive metrics for fraud detection evaluation
"""

import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score,
    confusion_matrix, classification_report
)
from typing import Dict, Tuple, Optional


class MetricsCalculator:
    """Calculates evaluation metrics for fraud detection"""
    
    def __init__(self, threshold: float = 0.5):
        """
        Initialize metrics calculator
        
        Args:
            threshold: Classification threshold
        """
        self.threshold = threshold
    
    def compute_all_metrics(self,
                           y_true: torch.Tensor,
                           y_pred_logits: torch.Tensor,
                           mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Compute all evaluation metrics
        
        Args:
            y_true: Ground truth labels
            y_pred_logits: Model predictions (logits)
            mask: Optional mask for selecting nodes
            
        Returns:
            Dictionary of metrics
        """
        # Apply mask if provided
        if mask is not None:
            y_true = y_true[mask]
            y_pred_logits = y_pred_logits[mask]
        
        # Convert to numpy
        y_true_np = y_true.cpu().numpy()
        y_pred_probs = torch.softmax(y_pred_logits, dim=1)[:, 1].cpu().numpy()
        y_pred_labels = (y_pred_probs >= self.threshold).astype(int)
        
        # Compute metrics
        metrics = {}
        
        # ROC-AUC
        try:
            metrics['roc_auc'] = roc_auc_score(y_true_np, y_pred_probs)
        except:
            metrics['roc_auc'] = 0.0
        
        # PR-AUC
        try:
            metrics['pr_auc'] = average_precision_score(y_true_np, y_pred_probs)
        except:
            metrics['pr_auc'] = 0.0
        
        # Classification metrics
        metrics['accuracy'] = accuracy_score(y_true_np, y_pred_labels)
        metrics['precision'] = precision_score(y_true_np, y_pred_labels, zero_division=0)
        metrics['recall'] = recall_score(y_true_np, y_pred_labels, zero_division=0)
        metrics['f1_score'] = f1_score(y_true_np, y_pred_labels, zero_division=0)
        
        return metrics
    
    def compute_confusion_matrix(self,
                                y_true: torch.Tensor,
                                y_pred_logits: torch.Tensor,
                                mask: Optional[torch.Tensor] = None) -> np.ndarray:
        """
        Compute confusion matrix
        
        Args:
            y_true: Ground truth labels
            y_pred_logits: Model predictions
            mask: Optional mask
            
        Returns:
            Confusion matrix
        """
        if mask is not None:
            y_true = y_true[mask]
            y_pred_logits = y_pred_logits[mask]
        
        y_true_np = y_true.cpu().numpy()
        y_pred_probs = torch.softmax(y_pred_logits, dim=1)[:, 1].cpu().numpy()
        y_pred_labels = (y_pred_probs >= self.threshold).astype(int)
        
        return confusion_matrix(y_true_np, y_pred_labels)

    # ------------------------------------------------------------------
    # Numpy-based helpers (used by mini-batch evaluator)
    # ------------------------------------------------------------------

    def compute_metrics_from_preds(self,
                                   y_true_np: np.ndarray,
                                   y_pred_probs: np.ndarray) -> Dict[str, float]:
        """
        Compute all metrics from pre-collected numpy arrays.

        Args:
            y_true_np: Ground-truth integer labels (0 / 1)
            y_pred_probs: Fraud-class probabilities in [0, 1]

        Returns:
            Dictionary of metrics
        """
        y_pred_labels = (y_pred_probs >= self.threshold).astype(int)
        metrics = {}

        try:
            metrics['roc_auc'] = roc_auc_score(y_true_np, y_pred_probs)
        except Exception:
            metrics['roc_auc'] = 0.0

        try:
            metrics['pr_auc'] = average_precision_score(y_true_np, y_pred_probs)
        except Exception:
            metrics['pr_auc'] = 0.0

        metrics['accuracy']  = accuracy_score(y_true_np, y_pred_labels)
        metrics['precision'] = precision_score(y_true_np, y_pred_labels, zero_division=0)
        metrics['recall']    = recall_score(y_true_np, y_pred_labels, zero_division=0)
        metrics['f1_score']  = f1_score(y_true_np, y_pred_labels, zero_division=0)

        return metrics

    def compute_confusion_matrix_from_preds(self,
                                            y_true_np: np.ndarray,
                                            y_pred_probs: np.ndarray) -> np.ndarray:
        """
        Compute confusion matrix from pre-collected numpy arrays.

        Args:
            y_true_np: Ground-truth integer labels
            y_pred_probs: Fraud-class probabilities in [0, 1]

        Returns:
            Confusion matrix as numpy array
        """
        y_pred_labels = (y_pred_probs >= self.threshold).astype(int)
        return confusion_matrix(y_true_np, y_pred_labels)

    
    def get_classification_report(self,
                                 y_true: torch.Tensor,
                                 y_pred_logits: torch.Tensor,
                                 mask: Optional[torch.Tensor] = None) -> str:
        """
        Get detailed classification report
        
        Args:
            y_true: Ground truth labels
            y_pred_logits: Model predictions
            mask: Optional mask
            
        Returns:
            Classification report string
        """
        if mask is not None:
            y_true = y_true[mask]
            y_pred_logits = y_pred_logits[mask]
        
        y_true_np = y_true.cpu().numpy()
        y_pred_probs = torch.softmax(y_pred_logits, dim=1)[:, 1].cpu().numpy()
        y_pred_labels = (y_pred_probs >= self.threshold).astype(int)
        
        target_names = ['Normal', 'Fraud']
        return classification_report(y_true_np, y_pred_labels, target_names=target_names)


def evaluate_model(model: torch.nn.Module,
                  data,
                  mask: torch.Tensor,
                  device: str = 'cpu') -> Dict[str, float]:
    """
    Convenience function to evaluate model
    
    Args:
        model: Trained model
        data: Graph data
        mask: Evaluation mask
        device: Device to use
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    model = model.to(device)
    data = data.to(device)
    
    with torch.no_grad():
        out = model(data.x, data.edge_index)
    
    calculator = MetricsCalculator()
    metrics = calculator.compute_all_metrics(data.y, out, mask)
    
    return metrics
