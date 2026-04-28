"""
Temporal Graph Neural Network
Processes time-evolving transaction graphs with memory modules
Reference: Saldaña-Ulloa et al. (2024) - Temporal Graph Network Algorithm for Fraud Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Optional, Tuple


class TemporalGNN(nn.Module):
    """
    Temporal GNN for streaming transaction fraud detection
    
    Key features from Saldaña-Ulloa et al. (2024):
    - Memory module to capture temporal patterns
    - Time encoding for temporal edges
    - Sliding window processing for streaming data
    """
    
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 128,
                 memory_dim: int = 100,
                 time_encoding_dim: int = 32,
                 num_temporal_layers: int = 2,
                 dropout: float = 0.3):
        """
        Initialize Temporal GNN
        
        Args:
            in_channels: Number of input features
            hidden_channels: Number of hidden units
            memory_dim: Dimension of memory module
            time_encoding_dim: Dimension of time encoding
            num_temporal_layers: Number of temporal layers
            dropout: Dropout rate
        """
        super(TemporalGNN, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.memory_dim = memory_dim
        self.time_encoding_dim = time_encoding_dim
        self.num_temporal_layers = num_temporal_layers
        self.dropout = dropout
        
        # Time encoding network
        self.time_encoder = nn.Sequential(
            nn.Linear(1, time_encoding_dim),
            nn.ReLU(),
            nn.Linear(time_encoding_dim, time_encoding_dim)
        )
        
        # Input projection with time encoding
        self.input_proj = nn.Linear(in_channels + time_encoding_dim, hidden_channels)
        
        # Temporal convolution layers
        self.temp_convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(num_temporal_layers):
            self.temp_convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Memory module (LSTM-based)
        self.memory_rnn = nn.LSTM(
            input_size=hidden_channels,
            hidden_size=memory_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Memory update network
        self.memory_updater = nn.Sequential(
            nn.Linear(memory_dim + hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # Classification head
        self.classifier = nn.Linear(hidden_channels, 2)
        
        # Initialize memory
        self.memory = None
    
    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None,
                reset_memory: bool = False) -> torch.Tensor:
        """
        Forward pass with temporal processing
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            timestamps: Node timestamps [num_nodes] (optional)
            reset_memory: Whether to reset memory module
            
        Returns:
            Logits [num_nodes, 2]
        """
        num_nodes = x.size(0)
        
        # Reset memory if requested
        if reset_memory or self.memory is None:
            self.memory = torch.zeros(num_nodes, self.memory_dim, device=x.device)
        
        # Encode timestamps
        if timestamps is not None:
            # Normalize timestamps
            time_normalized = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min() + 1e-6)
            time_encoded = self.time_encoder(time_normalized.unsqueeze(-1))
            
            # Concatenate with node features
            x = torch.cat([x, time_encoded], dim=-1)
        else:
            # Add dummy time encoding
            dummy_time = torch.zeros(num_nodes, self.time_encoding_dim, device=x.device)
            x = torch.cat([x, dummy_time], dim=-1)
        
        # Project input
        h = self.input_proj(x)
        h = F.relu(h)
        
        # Apply temporal convolutions
        for conv, bn in zip(self.temp_convs, self.batch_norms):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Update memory module
        # Combine current representation with memory
        memory_input = torch.cat([self.memory, h], dim=-1)
        h_updated = self.memory_updater(memory_input)
        
        # Update memory using LSTM
        if self.training:
            # During training, update memory
            h_seq = h.unsqueeze(1)  # [num_nodes, 1, hidden_channels]
            memory_out, (h_n, c_n) = self.memory_rnn(h_seq)
            self.memory = h_n.squeeze(0)  # Update memory
        
        # Classification
        logits = self.classifier(h_updated)
        
        return logits
    
    def get_embeddings(self,
                      x: torch.Tensor,
                      edge_index: torch.Tensor,
                      timestamps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get node embeddings before classification
        
        Args:
            x: Node features
            edge_index: Edge indices
            timestamps: Node timestamps (optional)
            
        Returns:
            Node embeddings [num_nodes, hidden_channels]
        """
        num_nodes = x.size(0)
        
        if self.memory is None:
            self.memory = torch.zeros(num_nodes, self.memory_dim, device=x.device)
        
        # Encode timestamps
        if timestamps is not None:
            time_normalized = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min() + 1e-6)
            time_encoded = self.time_encoder(time_normalized.unsqueeze(-1))
            x = torch.cat([x, time_encoded], dim=-1)
        else:
            dummy_time = torch.zeros(num_nodes, self.time_encoding_dim, device=x.device)
            x = torch.cat([x, dummy_time], dim=-1)
        
        h = self.input_proj(x)
        h = F.relu(h)
        
        for conv, bn in zip(self.temp_convs, self.batch_norms):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        memory_input = torch.cat([self.memory, h], dim=-1)
        h_updated = self.memory_updater(memory_input)
        
        return h_updated
    
    def reset_memory(self):
        """Reset memory module"""
        self.memory = None
    
    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'in_channels={self.in_channels}, '
                f'hidden_channels={self.hidden_channels}, '
                f'memory_dim={self.memory_dim}, '
                f'time_encoding_dim={self.time_encoding_dim}, '
                f'num_temporal_layers={self.num_temporal_layers})')
