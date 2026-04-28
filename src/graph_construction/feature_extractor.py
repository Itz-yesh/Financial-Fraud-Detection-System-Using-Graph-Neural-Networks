"""
Feature Extractor Module
Extracts node and edge features for graph neural networks
Includes statistical aggregations and graph-based features
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple
import networkx as nx


class FeatureExtractor:
    """Extracts features for graph nodes and edges"""
    
    def __init__(self):
        """Initialize feature extractor"""
        self.feature_names = []
    
    def extract_node_features(self, 
                             df: pd.DataFrame,
                             base_features: List[str]) -> Tuple[torch.Tensor, List[str]]:
        """
        Extract comprehensive node features
        
        Args:
            df: DataFrame with transaction data
            base_features: List of base feature column names
            
        Returns:
            Tuple of (feature tensor, feature names)
        """
        features_list = []
        feature_names = []
        
        # Add base features
        for col in base_features:
            if col in df.columns:
                features_list.append(df[col].values.reshape(-1, 1))
                feature_names.append(col)
        
        # Add statistical features
        stat_features, stat_names = self._extract_statistical_features(df)
        if stat_features is not None:
            features_list.append(stat_features)
            feature_names.extend(stat_names)
        
        # Add temporal features
        temporal_features, temporal_names = self._extract_temporal_features(df)
        if temporal_features is not None:
            features_list.append(temporal_features)
            feature_names.extend(temporal_names)
        
        # Concatenate all features
        if features_list:
            all_features = np.hstack(features_list)
        else:
            all_features = np.ones((len(df), 1))
            feature_names = ['dummy']
        
        self.feature_names = feature_names
        return torch.tensor(all_features, dtype=torch.float), feature_names
    
    def _extract_statistical_features(self, 
                                      df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Extract statistical aggregation features"""
        if 'amount' not in df.columns:
            return None, []
        
        features = []
        names = []
        
        # Amount statistics per user
        if 'user_id' in df.columns:
            user_stats = df.groupby('user_id')['amount'].agg(['mean', 'std', 'min', 'max'])
            user_stats = user_stats.fillna(0)
            
            # Map back to transactions
            for stat in ['mean', 'std', 'min', 'max']:
                df[f'user_amount_{stat}'] = df['user_id'].map(user_stats[stat])
                features.append(df[f'user_amount_{stat}'].values.reshape(-1, 1))
                names.append(f'user_amount_{stat}')
        
        # Amount statistics per merchant
        if 'merchant_id' in df.columns:
            merchant_stats = df.groupby('merchant_id')['amount'].agg(['mean', 'std'])
            merchant_stats = merchant_stats.fillna(0)
            
            for stat in ['mean', 'std']:
                df[f'merchant_amount_{stat}'] = df['merchant_id'].map(merchant_stats[stat])
                features.append(df[f'merchant_amount_{stat}'].values.reshape(-1, 1))
                names.append(f'merchant_amount_{stat}')
        
        if features:
            return np.hstack(features), names
        return None, []
    
    def _extract_temporal_features(self, 
                                   df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Extract temporal pattern features"""
        if 'timestamp' not in df.columns:
            return None, []
        
        features = []
        names = []
        
        # Convert to datetime if needed
        if df['timestamp'].dtype == 'object':
            timestamps = pd.to_datetime(df['timestamp'])
        else:
            timestamps = df['timestamp']
        
        # Time since first transaction (normalized)
        time_since_first = (timestamps - timestamps.min()).dt.total_seconds()
        time_since_first = time_since_first / time_since_first.max()
        features.append(time_since_first.values.reshape(-1, 1))
        names.append('time_since_first')
        
        # Transaction velocity (transactions per hour for user)
        if 'user_id' in df.columns:
            df['temp_timestamp'] = timestamps
            user_velocity = df.groupby('user_id').apply(
                lambda x: len(x) / ((x['temp_timestamp'].max() - x['temp_timestamp'].min()).total_seconds() / 3600 + 1)
            )
            df['user_velocity'] = df['user_id'].map(user_velocity)
            features.append(df['user_velocity'].fillna(0).values.reshape(-1, 1))
            names.append('user_velocity')
            df.drop('temp_timestamp', axis=1, inplace=True)
        
        if features:
            return np.hstack(features), names
        return None, []
    
    def extract_graph_features(self,
                              edge_index: torch.Tensor,
                              num_nodes: int) -> Tuple[torch.Tensor, List[str]]:
        """
        Extract graph-based features (degree, clustering coefficient, etc.)
        
        Args:
            edge_index: Edge index tensor
            num_nodes: Number of nodes
            
        Returns:
            Tuple of (graph features, feature names)
        """
        # Convert to NetworkX for graph metrics
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        edges = edge_index.t().numpy()
        G.add_edges_from(edges)
        
        features = []
        names = []
        
        # Node degree
        degrees = dict(G.degree())
        degree_values = np.array([degrees[i] for i in range(num_nodes)])
        features.append(degree_values.reshape(-1, 1))
        names.append('degree')
        
        # Clustering coefficient
        clustering = nx.clustering(G)
        clustering_values = np.array([clustering[i] for i in range(num_nodes)])
        features.append(clustering_values.reshape(-1, 1))
        names.append('clustering_coefficient')
        
        # PageRank (limited iterations for speed)
        try:
            pagerank = nx.pagerank(G, max_iter=10)
            pagerank_values = np.array([pagerank[i] for i in range(num_nodes)])
            features.append(pagerank_values.reshape(-1, 1))
            names.append('pagerank')
        except:
            pass  # Skip if PageRank fails
        
        if features:
            all_features = np.hstack(features)
            return torch.tensor(all_features, dtype=torch.float), names
        
        return torch.zeros((num_nodes, 1)), ['dummy']
    
    def extract_edge_features(self,
                             df: pd.DataFrame,
                             edge_index: torch.Tensor) -> torch.Tensor:
        """
        Extract edge features
        
        Args:
            df: DataFrame with transaction data
            edge_index: Edge index tensor
            
        Returns:
            Edge feature tensor
        """
        num_edges = edge_index.shape[1]
        edge_features = []
        
        # Edge type (based on connection reason)
        # This is a simplified version - in practice, track edge creation reason
        edge_types = torch.ones(num_edges, 1)
        edge_features.append(edge_types)
        
        # Temporal distance if timestamps available
        if 'timestamp' in df.columns:
            timestamps = pd.to_datetime(df['timestamp'])
            
            temporal_dists = []
            for i in range(num_edges):
                src, dst = edge_index[:, i]
                time_diff = abs((timestamps.iloc[dst] - timestamps.iloc[src]).total_seconds())
                temporal_dists.append(time_diff)
            
            temporal_dists = torch.tensor(temporal_dists, dtype=torch.float).reshape(-1, 1)
            # Normalize
            temporal_dists = temporal_dists / (temporal_dists.max() + 1e-6)
            edge_features.append(temporal_dists)
        
        return torch.cat(edge_features, dim=1)


def extract_features(df: pd.DataFrame,
                    base_features: List[str],
                    edge_index: torch.Tensor = None) -> Dict:
    """
    Convenience function to extract all features
    
    Args:
        df: Input DataFrame
        base_features: List of base feature columns
        edge_index: Edge index tensor (optional)
        
    Returns:
        Dictionary with node features, edge features, and feature names
    """
    extractor = FeatureExtractor()
    
    node_features, feature_names = extractor.extract_node_features(df, base_features)
    
    result = {
        'node_features': node_features,
        'feature_names': feature_names
    }
    
    if edge_index is not None:
        edge_features = extractor.extract_edge_features(df, edge_index)
        result['edge_features'] = edge_features
        
        graph_features, graph_feature_names = extractor.extract_graph_features(
            edge_index, len(df)
        )
        result['graph_features'] = graph_features
        result['graph_feature_names'] = graph_feature_names
    
    return result
