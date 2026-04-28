"""
Sampler Module
Subgraph sampling strategies for handling large graphs and class imbalance
Reference: Tong et al. (2023) - Subgraph sampling for imbalanced datasets
"""

import torch
import numpy as np
from typing import Tuple, Optional
from torch_geometric.data import Data


class SubgraphSampler:
    """
    Samples subgraphs for mini-batch training
    Handles class imbalance through strategic sampling
    """
    
    def __init__(self,
                 sampling_strategy: str = 'balanced',
                 num_neighbors: int = 10,
                 num_hops: int = 2):
        """
        Initialize subgraph sampler
        
        Args:
            sampling_strategy: Strategy ('balanced', 'uniform', 'fraud_focused')
            num_neighbors: Number of neighbors to sample per node
            num_hops: Number of hops for neighborhood sampling
        """
        self.sampling_strategy = sampling_strategy
        self.num_neighbors = num_neighbors
        self.num_hops = num_hops
    
    def sample_batch(self,
                    data: Data,
                    batch_size: int,
                    mask: torch.Tensor) -> Tuple[torch.Tensor, Data]:
        """
        Sample a batch of nodes and their subgraphs
        
        Args:
            data: Full graph data
            batch_size: Number of nodes to sample
            mask: Mask indicating which nodes can be sampled
            
        Returns:
            Tuple of (sampled node indices, subgraph data)
        """
        # Get indices of nodes that can be sampled
        available_indices = torch.where(mask)[0]
        
        if self.sampling_strategy == 'balanced':
            sampled_indices = self._balanced_sampling(data, available_indices, batch_size)
        elif self.sampling_strategy == 'fraud_focused':
            sampled_indices = self._fraud_focused_sampling(data, available_indices, batch_size)
        else:  # uniform
            sampled_indices = self._uniform_sampling(available_indices, batch_size)
        
        # Extract subgraph
        subgraph_data = self._extract_subgraph(data, sampled_indices)
        
        return sampled_indices, subgraph_data
    
    def _balanced_sampling(self,
                          data: Data,
                          available_indices: torch.Tensor,
                          batch_size: int) -> torch.Tensor:
        """
        Sample equal numbers of fraud and normal transactions
        Implements approach from Tong et al. (2023)
        """
        labels = data.y[available_indices]
        
        # Get fraud and normal indices
        fraud_mask = labels == 1
        normal_mask = labels == 0
        
        fraud_indices = available_indices[fraud_mask]
        normal_indices = available_indices[normal_mask]
        
        # Sample equal numbers from each class
        half_batch = batch_size // 2
        
        # Sample fraud cases
        if len(fraud_indices) >= half_batch:
            sampled_fraud = fraud_indices[torch.randperm(len(fraud_indices))[:half_batch]]
        else:
            # Oversample if not enough fraud cases
            sampled_fraud = fraud_indices[torch.randint(0, len(fraud_indices), (half_batch,))]
        
        # Sample normal cases
        if len(normal_indices) >= half_batch:
            sampled_normal = normal_indices[torch.randperm(len(normal_indices))[:half_batch]]
        else:
            sampled_normal = normal_indices[torch.randint(0, len(normal_indices), (half_batch,))]
        
        # Combine and shuffle
        sampled_indices = torch.cat([sampled_fraud, sampled_normal])
        sampled_indices = sampled_indices[torch.randperm(len(sampled_indices))]
        
        return sampled_indices
    
    def _fraud_focused_sampling(self,
                               data: Data,
                               available_indices: torch.Tensor,
                               batch_size: int) -> torch.Tensor:
        """
        Sample with higher probability for fraud cases
        """
        labels = data.y[available_indices]
        
        # Create sampling weights (higher for fraud)
        weights = torch.ones(len(available_indices))
        weights[labels == 1] = 3.0  # 3x weight for fraud cases
        
        # Normalize weights
        weights = weights / weights.sum()
        
        # Sample with replacement
        sampled_idx = torch.multinomial(weights, batch_size, replacement=True)
        sampled_indices = available_indices[sampled_idx]
        
        return sampled_indices
    
    def _uniform_sampling(self,
                         available_indices: torch.Tensor,
                         batch_size: int) -> torch.Tensor:
        """
        Uniform random sampling
        """
        if len(available_indices) >= batch_size:
            sampled_indices = available_indices[torch.randperm(len(available_indices))[:batch_size]]
        else:
            sampled_indices = available_indices[torch.randint(0, len(available_indices), (batch_size,))]
        
        return sampled_indices
    
    def _extract_subgraph(self,
                         data: Data,
                         center_nodes: torch.Tensor) -> Data:
        """
        Extract subgraph around center nodes
        
        Args:
            data: Full graph data
            center_nodes: Center nodes for subgraph extraction
            
        Returns:
            Subgraph data
        """
        # For simplicity, return the full graph with a mask
        # In practice, you would implement k-hop neighborhood extraction
        
        # Create a mask for the sampled nodes
        node_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        node_mask[center_nodes] = True
        
        # Create subgraph data
        subgraph = Data(
            x=data.x,
            edge_index=data.edge_index,
            y=data.y,
            batch_mask=node_mask  # Indicates which nodes are in this batch
        )
        
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            subgraph.edge_attr = data.edge_attr
        
        return subgraph


def create_sampler(strategy: str = 'balanced',
                  num_neighbors: int = 10,
                  num_hops: int = 2) -> SubgraphSampler:
    """
    Convenience function to create a sampler
    
    Args:
        strategy: Sampling strategy
        num_neighbors: Number of neighbors to sample
        num_hops: Number of hops
        
    Returns:
        SubgraphSampler instance
    """
    return SubgraphSampler(strategy, num_neighbors, num_hops)
