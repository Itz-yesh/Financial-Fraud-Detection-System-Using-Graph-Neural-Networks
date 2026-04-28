"""
Model Factory
Factory pattern for creating GNN models with configuration
"""

import torch.nn as nn
from typing import Dict, Any
from .gcn import GCN
from .graphsage import GraphSAGE
from .gat import GAT
from .jump_attention import JumpAttentionGNN
from .temporal_gnn import TemporalGNN


class ModelFactory:
    """Factory for creating GNN models"""
    
    SUPPORTED_MODELS = {
        'gcn': GCN,
        'graphsage': GraphSAGE,
        'gat': GAT,
        'jump_attention': JumpAttentionGNN,
        'temporal_gnn': TemporalGNN
    }
    
    @staticmethod
    def create_model(model_name: str,
                    in_channels: int,
                    config: Dict[str, Any]) -> nn.Module:
        """
        Create a GNN model based on configuration
        
        Args:
            model_name: Name of the model architecture
            in_channels: Number of input features
            config: Configuration dictionary
            
        Returns:
            Initialized model
        """
        model_name = model_name.lower()
        
        if model_name not in ModelFactory.SUPPORTED_MODELS:
            raise ValueError(f"Model {model_name} not supported. "
                           f"Choose from: {list(ModelFactory.SUPPORTED_MODELS.keys())}")
        
        # Get model class
        model_class = ModelFactory.SUPPORTED_MODELS[model_name]
        
        # Extract common parameters
        hidden_channels = config.get('hidden_dim', 128)
        num_layers = config.get('num_layers', 3)
        dropout = config.get('dropout', 0.3)
        
        # Create model with specific parameters
        if model_name == 'gcn':
            model = model_class(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                dropout=dropout,
                normalize=config.get('gcn', {}).get('normalize', True),
                add_self_loops=config.get('gcn', {}).get('add_self_loops', True)
            )
        
        elif model_name == 'graphsage':
            model = model_class(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                dropout=dropout,
                aggregator=config.get('graphsage', {}).get('aggregator', 'mean')
            )
        
        elif model_name == 'gat':
            model = model_class(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                num_heads=config.get('gat', {}).get('num_heads', 4),
                dropout=dropout,
                concat_heads=config.get('gat', {}).get('concat_heads', True)
            )
        
        elif model_name == 'jump_attention':
            model = model_class(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_hops=config.get('jump_attention', {}).get('num_hops', 3),
                dropout=dropout,
                attention_type=config.get('jump_attention', {}).get('attention_type', 'adaptive')
            )
        
        elif model_name == 'temporal_gnn':
            model = model_class(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                memory_dim=config.get('temporal_gnn', {}).get('memory_dim', 100),
                time_encoding_dim=config.get('temporal_gnn', {}).get('time_encoding_dim', 32),
                num_temporal_layers=config.get('temporal_gnn', {}).get('num_temporal_layers', 2),
                dropout=dropout
            )
        
        return model
    
    @staticmethod
    def get_model_info(model_name: str) -> Dict[str, Any]:
        """
        Get information about a model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information
        """
        info = {
            'gcn': {
                'name': 'Graph Convolutional Network',
                'reference': 'Kipf & Welling (2017)',
                'description': 'Baseline spectral graph convolution model'
            },
            'graphsage': {
                'name': 'GraphSAGE',
                'reference': 'Hamilton et al. (2017)',
                'description': 'Scalable GNN with neighborhood sampling'
            },
            'gat': {
                'name': 'Graph Attention Network',
                'reference': 'Veličković et al. (2018)',
                'description': 'Multi-head attention mechanism for graphs'
            },
            'jump_attention': {
                'name': 'Jump-Attentive GNN',
                'reference': 'Kadam et al. (2024)',
                'description': 'Adaptive attention across different hop neighborhoods'
            },
            'temporal_gnn': {
                'name': 'Temporal Graph Network',
                'reference': 'Saldaña-Ulloa et al. (2024)',
                'description': 'Temporal GNN with memory for streaming transactions'
            }
        }
        
        return info.get(model_name.lower(), {})


def create_model(model_name: str,
                in_channels: int,
                config: Dict[str, Any]) -> nn.Module:
    """
    Convenience function to create a model
    
    Args:
        model_name: Name of the model
        in_channels: Number of input features
        config: Configuration dictionary
        
    Returns:
        Initialized model
    """
    return ModelFactory.create_model(model_name, in_channels, config)
