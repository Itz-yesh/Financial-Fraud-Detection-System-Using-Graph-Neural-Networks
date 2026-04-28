"""
Graph Builder Module
Constructs transaction graphs from tabular data
Implements heterogeneous graph construction (Cheng 2024)
and temporal edges (Saldaña-Ulloa et al. 2024)
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, HeteroData
from typing import Dict, List, Tuple, Optional
import networkx as nx


class GraphBuilder:
    """Builds transaction graphs from tabular data"""
    
    def __init__(self,
                 node_types: List[str] = None,
                 edge_types: List[str] = None,
                 temporal_edges: bool = True,
                 time_window_hours: int = 24):
        """
        Initialize graph builder
        
        Args:
            node_types: Types of nodes to create
            edge_types: Types of edges to create
            temporal_edges: Whether to include temporal edges
            time_window_hours: Time window for temporal edges
        """
        self.node_types = node_types or ['transaction']
        self.edge_types = edge_types or ['transaction_to_transaction']
        self.temporal_edges = temporal_edges
        self.time_window_hours = time_window_hours
        
        self.node_mappings = {}
        self.edge_index_dict = {}
    
    def build_graph(self, 
                   df: pd.DataFrame,
                   feature_columns: List[str]) -> Data:
        """
        Build homogeneous graph from transaction data
        
        Args:
            df: DataFrame with transaction data
            feature_columns: List of feature column names
            
        Returns:
            PyTorch Geometric Data object
        """
        # Create node features
        node_features = self._create_node_features(df, feature_columns)
        
        # Create edges
        edge_index, edge_attr = self._create_edges(df)
        
        # Create labels
        labels = self._create_labels(df)
        
        # Create masks for train/val/test
        train_mask, val_mask, test_mask = self._create_masks(df)
        
        # Build PyG Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=labels,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask
        )
        
        print(f"\nGraph Statistics:")
        print(f"  Nodes: {data.num_nodes:,}")
        print(f"  Edges: {data.num_edges:,}")
        print(f"  Features per node: {data.num_node_features}")
        print(f"  Average degree: {data.num_edges / data.num_nodes:.2f}")
        
        return data
    
    def build_heterogeneous_graph(self,
                                  df: pd.DataFrame,
                                  feature_columns: List[str]) -> HeteroData:
        """
        Build heterogeneous graph with multiple node and edge types
        Implements approach from Cheng (2024)
        
        Args:
            df: DataFrame with transaction data
            feature_columns: List of feature column names
            
        Returns:
            PyTorch Geometric HeteroData object
        """
        data = HeteroData()
        
        # Create transaction nodes
        transaction_features = self._create_node_features(df, feature_columns)
        data['transaction'].x = transaction_features
        data['transaction'].y = self._create_labels(df)
        
        # Create user nodes if user_id exists
        if 'user_id' in df.columns:
            user_features, user_mapping = self._create_user_nodes(df)
            data['user'].x = user_features
            self.node_mappings['user'] = user_mapping
            
            # Create user-transaction edges
            user_tx_edges = self._create_user_transaction_edges(df, user_mapping)
            data['user', 'makes', 'transaction'].edge_index = user_tx_edges
        
        # Create merchant nodes if merchant_id exists
        if 'merchant_id' in df.columns:
            merchant_features, merchant_mapping = self._create_merchant_nodes(df)
            data['merchant'].x = merchant_features
            self.node_mappings['merchant'] = merchant_mapping
            
            # Create transaction-merchant edges
            tx_merchant_edges = self._create_transaction_merchant_edges(df, merchant_mapping)
            data['transaction', 'to', 'merchant'].edge_index = tx_merchant_edges
        
        # Create temporal transaction-transaction edges
        if self.temporal_edges and 'timestamp' in df.columns:
            tx_tx_edges = self._create_temporal_transaction_edges(df)
            data['transaction', 'precedes', 'transaction'].edge_index = tx_tx_edges
        
        # Create masks
        train_mask, val_mask, test_mask = self._create_masks(df)
        data['transaction'].train_mask = train_mask
        data['transaction'].val_mask = val_mask
        data['transaction'].test_mask = test_mask
        
        print(f"\nHeterogeneous Graph Statistics:")
        for node_type in data.node_types:
            print(f"  {node_type} nodes: {data[node_type].num_nodes:,}")
        for edge_type in data.edge_types:
            print(f"  {edge_type} edges: {data[edge_type].num_edges:,}")
        
        return data
    
    def _create_node_features(self, 
                             df: pd.DataFrame,
                             feature_columns: List[str]) -> torch.Tensor:
        """Create node feature matrix"""
        # Select feature columns that exist in dataframe
        available_features = [col for col in feature_columns if col in df.columns]
        
        if not available_features:
            # If no features available, use dummy features
            return torch.ones((len(df), 1), dtype=torch.float)
        
        features = df[available_features].values
        return torch.tensor(features, dtype=torch.float)
    
    def _create_edges(self, df: pd.DataFrame) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Create edges based on shared attributes and temporal proximity
        
        Returns:
            Tuple of (edge_index, edge_attr)
        """
        edges = []
        edge_features = []
        
        # Create edges based on shared users
        if 'user_id' in df.columns:
            user_edges = self._create_edges_by_attribute(df, 'user_id')
            edges.extend(user_edges)
            edge_features.extend([1.0] * len(user_edges))  # edge type 1
        
        # Create edges based on shared merchants
        if 'merchant_id' in df.columns:
            merchant_edges = self._create_edges_by_attribute(df, 'merchant_id')
            edges.extend(merchant_edges)
            edge_features.extend([2.0] * len(merchant_edges))  # edge type 2
        
        # Create temporal edges (Saldaña-Ulloa et al. 2024)
        if self.temporal_edges and 'timestamp' in df.columns:
            temporal_edges = self._create_temporal_edges(df)
            edges.extend(temporal_edges)
            edge_features.extend([3.0] * len(temporal_edges))  # edge type 3
        
        if not edges:
            # Create a simple sequential graph if no edges found
            edges = [[i, i+1] for i in range(len(df)-1)]
            edge_features = [0.0] * len(edges)
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float).unsqueeze(1)
        
        return edge_index, edge_attr
    
    def _safe_groupby_column(self, df: pd.DataFrame, attribute: str) -> pd.Series:
        """
        Return a memory-safe groupby-ready series.
        
        Pandas Categorical dtype causes `.groups` to pre-allocate one slot per
        category, exploding RAM on large datasets (PaySim, IEEE-CIS, etc.).
        Converting to plain object/string before groupby avoids this.
        """
        col = df[attribute]
        if hasattr(col, 'cat'):          # CategoricalDtype
            col = col.astype(str)
        return col
    
    def _create_edges_by_attribute(self,
                                   df: pd.DataFrame,
                                   attribute: str,
                                   max_edges_per_group: int = 10) -> List[List[int]]:
        """
        Create edges between transactions sharing an attribute.

        Uses a star (hub-and-spoke) topology instead of all-pairs to keep
        edge count O(n) per group rather than O(n^2). The first node in each
        group acts as the hub; up to max_edges_per_group spokes are added.
        This is critical for PaySim where accounts have thousands of
        transactions — all-pairs would produce billions of edges.
        """
        edges = []

        # Use a positional integer index so iloc-based access is fast
        # and we never rely on the DataFrame's own (potentially large) index.
        df_reset = df.reset_index(drop=True)
        groupby_col = self._safe_groupby_column(df_reset, attribute)

        # Iterate group-by-group — pandas never builds the full .groups dict
        for _, group in df_reset.groupby(groupby_col, sort=False):
            indices = group.index.tolist()   # positional indices after reset

            if len(indices) < 2:
                continue

            # Sample spokes if the group is large
            if len(indices) > max_edges_per_group + 1:
                hub = indices[0]
                spokes = np.random.choice(
                    indices[1:], max_edges_per_group, replace=False
                ).tolist()
            else:
                hub = indices[0]
                spokes = indices[1:]

            # Bidirectional hub→spoke edges only (star topology)
            for spoke in spokes:
                edges.append([hub, spoke])
                edges.append([spoke, hub])

        return edges
    
    def _create_temporal_edges(self,
                              df: pd.DataFrame,
                              max_neighbors: int = 50) -> List[List[int]]:
        """
        Create temporal edges based on time proximity.
        Implements approach from Saldaña-Ulloa et al. (2024).

        Handles two timestamp formats:
          - Numeric (int/float): PaySim 'step' column = integer hours since
            simulation start. Window is compared as plain integer hours.
          - Datetime: real datetime columns from other datasets. Window is
            compared as pd.Timedelta.
        """
        edges = []

        ts_col = df['timestamp']

        # --- Normalise to a sortable series ---------------------------------
        if np.issubdtype(ts_col.dtype, np.number):
            # PaySim: integer steps (hours). Keep as numeric; window = hours.
            timestamps = ts_col.reset_index(drop=True)
            time_window = self.time_window_hours          # plain int/float
        else:
            # Datetime string or already datetime dtype
            if ts_col.dtype == 'object':
                timestamps = pd.to_datetime(ts_col).reset_index(drop=True)
            else:
                timestamps = ts_col.reset_index(drop=True)
            time_window = pd.Timedelta(hours=self.time_window_hours)
        # --------------------------------------------------------------------

        # Sort by timestamp (argsort returns positional order)
        sorted_pos = timestamps.argsort().values        # ndarray of positions
        sorted_times = timestamps.iloc[sorted_pos]

        # For each transaction connect to temporally nearby transactions
        for i in range(len(sorted_pos)):
            idx = sorted_pos[i]
            time = sorted_times.iat[i]

            j = i + 1
            neighbors = 0
            while j < len(sorted_pos) and neighbors < max_neighbors:
                next_idx = sorted_pos[j]
                next_time = sorted_times.iat[j]

                if next_time - time > time_window:
                    break

                edges.append([int(idx), int(next_idx)])
                neighbors += 1
                j += 1

        return edges
    
    def _create_user_nodes(self, df: pd.DataFrame) -> Tuple[torch.Tensor, Dict]:
        """Create user nodes with aggregated features"""
        unique_users = df['user_id'].unique()
        user_mapping = {user_id: idx for idx, user_id in enumerate(unique_users)}
        
        # Aggregate user features (e.g., average transaction amount, count)
        user_features = []
        for user_id in unique_users:
            user_txs = df[df['user_id'] == user_id]
            
            features = [
                len(user_txs),  # transaction count
                user_txs['amount'].mean() if 'amount' in df.columns else 0,
                user_txs['amount'].std() if 'amount' in df.columns else 0,
                user_txs['is_fraud'].mean() if 'is_fraud' in df.columns else 0,
            ]
            user_features.append(features)
        
        return torch.tensor(user_features, dtype=torch.float), user_mapping
    
    def _create_merchant_nodes(self, df: pd.DataFrame) -> Tuple[torch.Tensor, Dict]:
        """Create merchant nodes with aggregated features"""
        unique_merchants = df['merchant_id'].unique()
        merchant_mapping = {merchant_id: idx for idx, merchant_id in enumerate(unique_merchants)}
        
        merchant_features = []
        for merchant_id in unique_merchants:
            merchant_txs = df[df['merchant_id'] == merchant_id]
            
            features = [
                len(merchant_txs),  # transaction count
                merchant_txs['amount'].mean() if 'amount' in df.columns else 0,
                merchant_txs['amount'].std() if 'amount' in df.columns else 0,
                merchant_txs['is_fraud'].mean() if 'is_fraud' in df.columns else 0,
            ]
            merchant_features.append(features)
        
        return torch.tensor(merchant_features, dtype=torch.float), merchant_mapping
    
    def _create_user_transaction_edges(self, 
                                       df: pd.DataFrame,
                                       user_mapping: Dict) -> torch.Tensor:
        """Create edges from users to transactions"""
        edges = []
        for tx_idx, row in df.iterrows():
            user_id = row['user_id']
            if user_id in user_mapping:
                user_idx = user_mapping[user_id]
                edges.append([user_idx, tx_idx])
        
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    def _create_transaction_merchant_edges(self,
                                           df: pd.DataFrame,
                                           merchant_mapping: Dict) -> torch.Tensor:
        """Create edges from transactions to merchants"""
        edges = []
        for tx_idx, row in df.iterrows():
            merchant_id = row['merchant_id']
            if merchant_id in merchant_mapping:
                merchant_idx = merchant_mapping[merchant_id]
                edges.append([tx_idx, merchant_idx])
        
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    def _create_temporal_transaction_edges(self, df: pd.DataFrame) -> torch.Tensor:
        """Create temporal edges between transactions"""
        temporal_edges = self._create_temporal_edges(df)
        return torch.tensor(temporal_edges, dtype=torch.long).t().contiguous()
    
    def _create_labels(self, df: pd.DataFrame) -> torch.Tensor:
        """Create label tensor"""
        if 'is_fraud' in df.columns:
            labels = df['is_fraud'].values
        else:
            labels = np.zeros(len(df))
        
        return torch.tensor(labels, dtype=torch.long)
    
    def _create_masks(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create train/val/test masks"""
        n = len(df)
        
        # Check if split column exists
        if 'split' in df.columns:
            train_mask = torch.tensor(df['split'] == 'train', dtype=torch.bool)
            val_mask = torch.tensor(df['split'] == 'val', dtype=torch.bool)
            test_mask = torch.tensor(df['split'] == 'test', dtype=torch.bool)
        else:
            # Create default masks
            train_mask = torch.zeros(n, dtype=torch.bool)
            val_mask = torch.zeros(n, dtype=torch.bool)
            test_mask = torch.zeros(n, dtype=torch.bool)
            
            # Use indices if available
            train_end = int(0.7 * n)
            val_end = int(0.85 * n)
            
            train_mask[:train_end] = True
            val_mask[train_end:val_end] = True
            test_mask[val_end:] = True
        
        return train_mask, val_mask, test_mask


def build_graph_from_dataframe(df: pd.DataFrame,
                               feature_columns: List[str],
                               heterogeneous: bool = False) -> Data:
    """
    Convenience function to build graph from DataFrame
    
    Args:
        df: Input DataFrame
        feature_columns: List of feature columns
        heterogeneous: Whether to build heterogeneous graph
        
    Returns:
        PyTorch Geometric Data or HeteroData object
    """
    builder = GraphBuilder()
    
    if heterogeneous:
        return builder.build_heterogeneous_graph(df, feature_columns)
    else:
        return builder.build_graph(df, feature_columns)
