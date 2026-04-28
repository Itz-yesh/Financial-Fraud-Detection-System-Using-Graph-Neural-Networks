"""
Model Evaluator
Complete evaluation pipeline with visualization
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Optional
import os
from torch_geometric.loader import NeighborLoader

from .metrics import MetricsCalculator


class ModelEvaluator:
    """Evaluates trained models and generates reports"""
    
    def __init__(self, model: torch.nn.Module, data, device: str = 'cpu', batch_size: int = 1024):
        """
        Initialize evaluator
        
        Args:
            model: Trained model
            data: Graph data
            device: Device to use
            batch_size: Mini-batch size for evaluation
        """
        self.model = model.to(device)
        self.data = data.to(device)
        self.device = device
        self.metrics_calc = MetricsCalculator()
        self.batch_size = batch_size
        
        self.model.eval()
    
    @torch.no_grad()
    def evaluate(self, split: str = 'test') -> Dict:
        """
        Evaluate model on specified split using mini-batching
        
        Args:
            split: Data split ('train', 'val', 'test')
            
        Returns:
            Dictionary with metrics and predictions
        """
        # Determine target nodes for this split
        if split == 'train':
            input_nodes = self.data.train_mask.nonzero(as_tuple=True)[0]
        elif split == 'val':
            input_nodes = self.data.val_mask.nonzero(as_tuple=True)[0]
        else:
            input_nodes = self.data.test_mask.nonzero(as_tuple=True)[0]
            
        # Set up mini-batch loader
        loader = NeighborLoader(
            self.data,
            num_neighbors=[20, 20],  # Limit neighbors to prevent OOM
            batch_size=self.batch_size,
            input_nodes=input_nodes,
            shuffle=False
        )
        
        all_preds = []
        all_labels = []
        
        for batch in loader:
            batch = batch.to(self.device)
            out = self.model(batch.x, batch.edge_index)
            
            # Mask to only evaluate target nodes in batch
            mask = batch.batch_size
            probs = torch.softmax(out[:mask], dim=1)[:, 1].cpu().numpy()
            labels = batch.y[:mask].cpu().numpy()
            
            all_preds.extend(probs)
            all_labels.extend(labels)
            
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Compute metrics
        metrics = self.metrics_calc.compute_metrics_from_preds(all_labels, all_preds)
        
        # Confusion matrix
        cm = self.metrics_calc.compute_confusion_matrix_from_preds(all_labels, all_preds)
        
        return {
            'metrics': metrics,
            'predictions': all_preds,
            'labels': all_labels,
            'confusion_matrix': cm
        }
    
    def evaluate_all_splits(self) -> Dict:
        """
        Evaluate on all data splits
        
        Returns:
            Dictionary with results for each split
        """
        results = {}
        for split in ['train', 'val', 'test']:
            results[split] = self.evaluate(split)
        
        return results
    
    def plot_confusion_matrix(self, 
                             cm: np.ndarray,
                             save_path: Optional[str] = None):
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            save_path: Path to save plot
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Fraud'],
                   yticklabels=['Normal', 'Fraud'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curve(self,
                      labels: np.ndarray,
                      predictions: np.ndarray,
                      save_path: Optional[str] = None):
        """
        Plot ROC curve
        
        Args:
            labels: True labels
            predictions: Predicted probabilities
            save_path: Path to save plot
        """
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(labels, predictions)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, output_dir: str = 'results'):
        """
        Generate complete evaluation report
        
        Args:
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Evaluate all splits
        results = self.evaluate_all_splits()
        
        # Save metrics
        with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
            for split, result in results.items():
                f.write(f"\n{'='*50}\n")
                f.write(f"{split.upper()} SET METRICS\n")
                f.write(f"{'='*50}\n")
                for metric, value in result['metrics'].items():
                    f.write(f"{metric}: {value:.4f}\n")
        
        # Plot confusion matrices
        for split, result in results.items():
            self.plot_confusion_matrix(
                result['confusion_matrix'],
                os.path.join(output_dir, f'confusion_matrix_{split}.png')
            )
        
        # Plot ROC curves
        for split, result in results.items():
            self.plot_roc_curve(
                result['labels'],
                result['predictions'],
                os.path.join(output_dir, f'roc_curve_{split}.png')
            )
        
        print(f"Evaluation report saved to: {output_dir}")
        
        return results
