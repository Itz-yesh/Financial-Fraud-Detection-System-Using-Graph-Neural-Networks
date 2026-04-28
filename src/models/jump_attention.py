"""
Jump-Attentive Graph Neural Network
Implements adaptive attention across different hop neighborhoods
Reference: Kadam et al. (2024) - Jump-Attentive Graph Neural Networks for Financial Fraud Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from typing import List, Optional


class JumpAttentionGNN(nn.Module):
    """
    Jump-Attentive GNN with adaptive attention weights across hops
    
    Key innovation from Kadam et al. (2024):
    - Learns importance of different hop neighborhoods
    - Adaptive attention weights for multi-hop aggregation
    - Better captures long-range dependencies in fraud patterns
    """
    
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 128,
                 num_hops: int = 3,
                 dropout: float = 0.3,
                 attention_type: str = 'adaptive'):
        """
        Initialize Jump-Attention GNN
        
        Args:
            in_channels: Number of input features
            hidden_channels: Number of hidden units
            num_hops: Number of hops to aggregate
            dropout: Dropout rate
            attention_type: Type of attention ('adaptive' or 'fixed')
        """
        super(JumpAttentionGNN, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_hops = num_hops
        self.dropout = dropout
        self.attention_type = attention_type
        
        # Create convolution layers for each hop
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Subsequent layers
        for _ in range(num_hops - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Jump attention mechanism (Kadam et al. 2024)
        if attention_type == 'adaptive':
            # Learnable attention weights for each hop
            self.attention_weights = nn.Parameter(torch.ones(num_hops))
            
            # Attention network to compute hop importance
            self.attention_network = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Linear(hidden_channels // 2, 1)
            )
        else:
            # Fixed uniform weights
            self.register_buffer('attention_weights', 
                               torch.ones(num_hops) / num_hops)
        
        # Combine multi-hop representations
        self.combiner = nn.Linear(hidden_channels, hidden_channels)
        
        # Classification head
        self.classifier = nn.Linear(hidden_channels, 2)
        
        # Store hop representations for analysis
        self.hop_representations = []
        self.hop_attention_scores = []
    
    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                return_hop_attention: bool = False) -> torch.Tensor:
        """
        Forward pass with jump-attention mechanism
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            return_hop_attention: Whether to return hop attention scores
            
        Returns:
            Logits [num_nodes, 2] or (logits, hop_attention)
        """
        # Store representations from each hop
        hop_outputs = []
        self.hop_representations = []
        
        h = x
        
        # Apply convolutions for each hop
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            
            # Store hop representation
            hop_outputs.append(h)
            self.hop_representations.append(h.detach())
        
        # Compute adaptive attention weights (Kadam et al. 2024)
        if self.attention_type == 'adaptive':
            # Compute attention scores for each hop
            attention_scores = []
            for hop_repr in hop_outputs:
                # Global pooling of hop representation
                pooled = torch.mean(hop_repr, dim=0, keepdim=True)
                score = self.attention_network(pooled)
                attention_scores.append(score)
            
            # Normalize attention scores
            attention_scores = torch.cat(attention_scores, dim=1)
            attention_weights = F.softmax(attention_scores, dim=1)
            attention_weights = attention_weights.squeeze(0)
            
            # Also use learnable base weights
            base_weights = F.softmax(self.attention_weights, dim=0)
            attention_weights = 0.5 * attention_weights + 0.5 * base_weights
        else:
            attention_weights = self.attention_weights
        
        self.hop_attention_scores = attention_weights.detach()
        
        # Weighted combination of hop representations
        combined = torch.zeros_like(hop_outputs[0])
        for i, hop_repr in enumerate(hop_outputs):
            combined += attention_weights[i] * hop_repr
        
        # Final transformation
        combined = self.combiner(combined)
        combined = F.relu(combined)
        
        # Classification
        logits = self.classifier(combined)
        
        if return_hop_attention:
            return logits, attention_weights
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
        hop_outputs = []
        h = x
        
        for conv, bn in zip(self.convs, self.batch_norms):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            hop_outputs.append(h)
        
        # Compute attention and combine
        if self.attention_type == 'adaptive':
            attention_scores = []
            for hop_repr in hop_outputs:
                pooled = torch.mean(hop_repr, dim=0, keepdim=True)
                score = self.attention_network(pooled)
                attention_scores.append(score)
            
            attention_scores = torch.cat(attention_scores, dim=1)
            attention_weights = F.softmax(attention_scores, dim=1).squeeze(0)
            
            base_weights = F.softmax(self.attention_weights, dim=0)
            attention_weights = 0.5 * attention_weights + 0.5 * base_weights
        else:
            attention_weights = self.attention_weights
        
        combined = torch.zeros_like(hop_outputs[0])
        for i, hop_repr in enumerate(hop_outputs):
            combined += attention_weights[i] * hop_repr
        
        combined = self.combiner(combined)
        combined = F.relu(combined)
        
        return combined
    
    def get_hop_attention_scores(self) -> torch.Tensor:
        """
        Get attention scores for each hop from last forward pass
        
        Returns:
            Attention weights [num_hops]
        """
        return self.hop_attention_scores
    
    def get_hop_representations(self) -> List[torch.Tensor]:
        """
        Get representations from each hop
        
        Returns:
            List of hop representations
        """
        return self.hop_representations
    
    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'in_channels={self.in_channels}, '
                f'hidden_channels={self.hidden_channels}, '
                f'num_hops={self.num_hops}, '
                f'attention_type={self.attention_type}, '
                f'dropout={self.dropout})')
