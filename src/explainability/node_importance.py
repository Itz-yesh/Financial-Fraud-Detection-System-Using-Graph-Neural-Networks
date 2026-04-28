"""
Node Importance Calculator
Computes node importance scores for explainability
"""

import torch
import numpy as np
from typing import List, Dict, Optional
from torch_geometric.utils import k_hop_subgraph


class NodeImportanceCalculator:
    """Calculates node importance for fraud predictions"""
    
    def __init__(self, model: torch.nn.Module, data):
        """
        Initialize node importance calculator
        
        Args:
            model: Trained model
            data: Graph data
        """
        self.model = model
        self.data = data
        self.model.eval()
        
    def _get_subgraph(self, node_idx: int):
        """Extracts k-hop subgraph around the node for efficient computation."""
        num_hops = getattr(self.model, 'num_hops', 3)
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx, num_hops, self.data.edge_index, relabel_nodes=True,
            num_nodes=self.data.x.size(0))
        x = self.data.x[subset]
        new_node_idx = mapping[0].item()
        return x, edge_index, new_node_idx

    
    def compute_gradient_importance(self, node_idx: int) -> Dict[str, float]:
        """
        Compute feature importance using gradients
        
        Args:
            node_idx: Index of the node
            
        Returns:
            Dictionary of feature importances
        """
        self.model.zero_grad()
        
        # Get k-hop subgraph to avoid out-of-memory errors on large graphs
        x_sub, edge_index_sub, new_node_idx = self._get_subgraph(node_idx)
        
        # Enable gradient for input
        x = x_sub.clone().requires_grad_(True)
        
        # Forward pass
        out = self.model(x, edge_index_sub)
        
        # Get prediction for node
        pred = out[new_node_idx, 1]  # Fraud class logit
        
        # Backward pass
        pred.backward()
        
        # Get gradients
        gradients = x.grad[new_node_idx].abs().cpu().numpy()
        
        # Create importance dict
        importance = {}
        for i, grad in enumerate(gradients):
            importance[f'feature_{i}'] = float(grad)
        
        return importance
    
    def compute_perturbation_importance(self, 
                                       node_idx: int,
                                       num_perturbations: int = 100) -> Dict[str, float]:
        """
        Compute feature importance using perturbation analysis
        
        Args:
            node_idx: Index of the node
            num_perturbations: Number of perturbations per feature
            
        Returns:
            Dictionary of feature importances
        """
        with torch.no_grad():
            # Get k-hop subgraph
            x_sub, edge_index_sub, new_node_idx = self._get_subgraph(node_idx)
            
            # Get original prediction
            out_orig = self.model(x_sub, edge_index_sub)
            pred_orig = torch.softmax(out_orig[new_node_idx], dim=0)[1].item()
            
            # Compute importance for each feature
            importance = {}
            num_features = x_sub.shape[1]
            
            for feat_idx in range(num_features):
                pred_diffs = []
                
                for _ in range(num_perturbations):
                    # Perturb feature
                    x_perturbed = x_sub.clone()
                    x_perturbed[new_node_idx, feat_idx] += torch.randn(1).item() * 0.1
                    
                    # Get perturbed prediction
                    out_perturbed = self.model(x_perturbed, edge_index_sub)
                    pred_perturbed = torch.softmax(out_perturbed[new_node_idx], dim=0)[1].item()
                    
                    # Compute difference
                    pred_diffs.append(abs(pred_orig - pred_perturbed))
                
                # Average importance
                importance[f'feature_{feat_idx}'] = np.mean(pred_diffs)
        
        return importance
    
    def get_top_important_features(self,
                                   node_idx: int,
                                   top_k: int = 5,
                                   method: str = 'gradient') -> List[tuple]:
        """
        Get top-k most important features for a node
        
        Args:
            node_idx: Index of the node
            top_k: Number of top features to return
            method: Importance method ('gradient' or 'perturbation')
            
        Returns:
            List of (feature_name, importance_score) tuples
        """
        if method == 'gradient':
            importance = self.compute_gradient_importance(node_idx)
        else:
            importance = self.compute_perturbation_importance(node_idx)
        
        # Sort by importance
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_features[:top_k]
    
    def explain_prediction(self, node_idx: int) -> Dict:
        """
        Generate comprehensive explanation for a prediction
        
        Args:
            node_idx: Index of the node
            
        Returns:
            Dictionary with explanation
        """
        with torch.no_grad():
            # Get k-hop subgraph
            x_sub, edge_index_sub, new_node_idx = self._get_subgraph(node_idx)
            
            # Get prediction
            out = self.model(x_sub, edge_index_sub)
            pred_prob = torch.softmax(out[new_node_idx], dim=0)[1].item()
            pred_label = 'Fraud' if pred_prob > 0.5 else 'Normal'
            true_label = 'Fraud' if self.data.y[node_idx].item() == 1 else 'Normal'
        
        # Get important features
        top_features = self.get_top_important_features(node_idx, top_k=5)
        
        explanation = {
            'node_idx': node_idx,
            'predicted_label': pred_label,
            'predicted_probability': pred_prob,
            'true_label': true_label,
            'correct': pred_label == true_label,
            'top_features': top_features
        }
        
        return explanation
