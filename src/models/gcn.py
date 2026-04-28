"""
Graph Convolutional Network (GCN) Model
Baseline GNN model using spectral graph convolutions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Optional


class GCN(nn.Module):
    """
    Graph Convolutional Network for fraud detection
    
    Reference: Kipf & Welling (2017) - Semi-Supervised Classification with Graph Convolutional Networks
    """
    
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.3,
                 normalize: bool = True,
                 add_self_loops: bool = True):
        """
        Initialize GCN model
        
        Args:
            in_channels: Number of input features
            hidden_channels: Number of hidden units
            num_layers: Number of GCN layers
            dropout: Dropout rate
            normalize: Whether to normalize adjacency matrix
            add_self_loops: Whether to add self-loops
        """
        super(GCN, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Create GCN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(in_channels, hidden_channels, 
                                 normalize=normalize,
                                 add_self_loops=add_self_loops))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels,
                                     normalize=normalize,
                                     add_self_loops=add_self_loops))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Output layer
        self.convs.append(GCNConv(hidden_channels, hidden_channels,
                                 normalize=normalize,
                                 add_self_loops=add_self_loops))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Classification head
        self.classifier = nn.Linear(hidden_channels, 2)  # Binary classification
    
    def forward(self, 
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Edge weights (optional)
            
        Returns:
            Logits [num_nodes, 2]
        """
        # Apply GCN layers
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final GCN layer
        x = self.convs[-1](x, edge_index, edge_weight)
        x = self.batch_norms[-1](x)
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def get_embeddings(self,
                      x: torch.Tensor,
                      edge_index: torch.Tensor,
                      edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get node embeddings before classification
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_weight: Edge weights (optional)
            
        Returns:
            Node embeddings [num_nodes, hidden_channels]
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index, edge_weight)
        x = self.batch_norms[-1](x)
        
        return x
    
    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'in_channels={self.in_channels}, '
                f'hidden_channels={self.hidden_channels}, '
                f'num_layers={self.num_layers}, '
                f'dropout={self.dropout})')
