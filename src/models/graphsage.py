"""
GraphSAGE Model
Scalable GNN with neighborhood sampling and aggregation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from typing import Optional


class GraphSAGE(nn.Module):
    """
    GraphSAGE model for fraud detection
    
    Reference: Hamilton et al. (2017) - Inductive Representation Learning on Large Graphs
    """
    
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.3,
                 aggregator: str = 'mean'):
        """
        Initialize GraphSAGE model
        
        Args:
            in_channels: Number of input features
            hidden_channels: Number of hidden units
            num_layers: Number of SAGE layers
            dropout: Dropout rate
            aggregator: Aggregation method ('mean', 'max', 'lstm')
        """
        super(GraphSAGE, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.aggregator = aggregator
        
        # Create SAGE layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(SAGEConv(in_channels, hidden_channels, 
                                  aggr=aggregator))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels,
                                      aggr=aggregator))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Output layer
        self.convs.append(SAGEConv(hidden_channels, hidden_channels,
                                  aggr=aggregator))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Classification head
        self.classifier = nn.Linear(hidden_channels, 2)
    
    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Logits [num_nodes, 2]
        """
        # Apply SAGE layers
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final SAGE layer
        x = self.convs[-1](x, edge_index)
        x = self.batch_norms[-1](x)
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def get_embeddings(self,
                      x: torch.Tensor,
                      edge_index: torch.Tensor) -> torch.Tensor:
        """
        Get node embeddings before classification
        
        Args:
            x: Node features
            edge_index: Edge indices
            
        Returns:
            Node embeddings [num_nodes, hidden_channels]
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        x = self.batch_norms[-1](x)
        
        return x
    
    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'in_channels={self.in_channels}, '
                f'hidden_channels={self.hidden_channels}, '
                f'num_layers={self.num_layers}, '
                f'aggregator={self.aggregator}, '
                f'dropout={self.dropout})')
