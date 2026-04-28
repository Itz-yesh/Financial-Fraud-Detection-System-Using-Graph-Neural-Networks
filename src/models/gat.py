"""
Graph Attention Network (GAT) Model
Uses multi-head attention mechanism to learn edge importance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Optional, Tuple


class GAT(nn.Module):
    """
    Graph Attention Network for fraud detection
    
    Reference: Veličković et al. (2018) - Graph Attention Networks
    """
    
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 128,
                 num_layers: int = 3,
                 num_heads: int = 4,
                 dropout: float = 0.3,
                 concat_heads: bool = True):
        """
        Initialize GAT model
        
        Args:
            in_channels: Number of input features
            hidden_channels: Number of hidden units per head
            num_layers: Number of GAT layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            concat_heads: Whether to concatenate or average attention heads
        """
        super(GAT, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.concat_heads = concat_heads
        
        # Create GAT layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Calculate actual hidden dimension based on concatenation
        if concat_heads:
            actual_hidden = hidden_channels * num_heads
        else:
            actual_hidden = hidden_channels
        
        # Input layer
        self.convs.append(GATConv(in_channels, hidden_channels,
                                 heads=num_heads,
                                 dropout=dropout,
                                 concat=concat_heads))
        self.batch_norms.append(nn.BatchNorm1d(actual_hidden))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(actual_hidden, hidden_channels,
                                     heads=num_heads,
                                     dropout=dropout,
                                     concat=concat_heads))
            self.batch_norms.append(nn.BatchNorm1d(actual_hidden))
        
        # Output layer (average heads for final layer)
        self.convs.append(GATConv(actual_hidden, hidden_channels,
                                 heads=num_heads,
                                 dropout=dropout,
                                 concat=False))  # Average for output
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Classification head
        self.classifier = nn.Linear(hidden_channels, 2)
        
        # Store attention weights for explainability
        self.attention_weights = None
    
    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                return_attention_weights: bool = False) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Logits [num_nodes, 2] or (logits, attention_weights)
        """
        attention_weights_list = []
        
        # Apply GAT layers
        for i, conv in enumerate(self.convs[:-1]):
            if return_attention_weights:
                x, attn = conv(x, edge_index, return_attention_weights=True)
                attention_weights_list.append(attn)
            else:
                x = conv(x, edge_index)
            
            x = self.batch_norms[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final GAT layer
        if return_attention_weights:
            x, attn = self.convs[-1](x, edge_index, return_attention_weights=True)
            attention_weights_list.append(attn)
            self.attention_weights = attention_weights_list
        else:
            x = self.convs[-1](x, edge_index)
        
        x = self.batch_norms[-1](x)
        
        # Classification
        logits = self.classifier(x)
        
        if return_attention_weights:
            return logits, attention_weights_list
        return logits
    
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
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        x = self.batch_norms[-1](x)
        
        return x
    
    def get_attention_weights(self) -> Optional[list]:
        """
        Get stored attention weights from last forward pass
        
        Returns:
            List of attention weight tuples (edge_index, attention_values)
        """
        return self.attention_weights
    
    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'in_channels={self.in_channels}, '
                f'hidden_channels={self.hidden_channels}, '
                f'num_layers={self.num_layers}, '
                f'num_heads={self.num_heads}, '
                f'dropout={self.dropout})')
