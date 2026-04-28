"""
Attention Visualizer
Visualizes attention weights from GAT and Jump-Attention models
Reference: Cheng (2024) - Explainability in GNN fraud detection
"""

import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import Optional, List, Tuple
import os
from torch_geometric.utils import k_hop_subgraph


class AttentionVisualizer:
    """Visualizes attention weights for explainability"""
    
    def __init__(self, model: torch.nn.Module, data):
        """
        Initialize attention visualizer
        
        Args:
            model: Trained GAT or Jump-Attention model
            data: Graph data
        """
        self.model = model
        self.data = data
        self.model.eval()
    
    @torch.no_grad()
    def _infer_subgraph(self, node_idx: int, num_hops: int = 2) -> torch.Tensor:
        """
        Run inference only on the local k-hop subgraph of node_idx.
        Returns fraud-class probabilities for ALL nodes in that subgraph,
        indexed by their ORIGINAL global node ids.
        """
        # Extract the k-hop subgraph around node_idx
        subset, sub_edge_index, mapping, _ = k_hop_subgraph(
            node_idx=node_idx,
            num_hops=num_hops,
            edge_index=self.data.edge_index,
            relabel_nodes=True,   # re-index nodes 0..len(subset)-1
            num_nodes=self.data.num_nodes,
        )
        sub_x = self.data.x[subset]  # features of subgraph nodes

        out = self.model(sub_x, sub_edge_index)  # small forward pass
        probs = torch.softmax(out, dim=1)[:, 1]  # fraud probabilities

        # Build a global_node_id -> probability mapping
        global_probs = {int(global_id): float(probs[local_id])
                        for local_id, global_id in enumerate(subset.tolist())}
        return global_probs

    @torch.no_grad()
    def get_attention_weights(self, node_idx: int) -> Optional[np.ndarray]:
        """
        Get attention weights for a specific node

        Args:
            node_idx: Index of the node

        Returns:
            Attention weights array
        """
        if hasattr(self.model, 'forward') and 'attention' in self.model.__class__.__name__.lower():
            if hasattr(self.model, 'get_attention_weights'):
                # Run only on the local subgraph to avoid OOM
                subset, sub_edge_index, _, _ = k_hop_subgraph(
                    node_idx=node_idx,
                    num_hops=2,
                    edge_index=self.data.edge_index,
                    relabel_nodes=True,
                    num_nodes=self.data.num_nodes,
                )
                sub_x = self.data.x[subset]
                _ = self.model(sub_x, sub_edge_index, return_attention_weights=True)
                attention_weights = self.model.get_attention_weights()
                return attention_weights

        return None
    
    def visualize_node_attention(self,
                                node_idx: int,
                                k_neighbors: int = 10,
                                save_path: Optional[str] = None):
        """
        Visualize attention weights for a node's neighborhood
        
        Args:
            node_idx: Index of the node to visualize
            k_neighbors: Number of neighbors to show
            save_path: Path to save visualization
        """
        # Get k-hop neighbors around the target node to avoid loading entire graph into NetworkX
        subset, sub_edge_index, _, _ = k_hop_subgraph(
            node_idx=node_idx,
            num_hops=1,
            edge_index=self.data.edge_index,
            relabel_nodes=False,
            num_nodes=self.data.num_nodes,
        )
        edge_index = sub_edge_index.cpu().numpy()
        
        # Build NetworkX graph only for the subgraph
        G = nx.Graph()
        G.add_edges_from(edge_index.T)
        
        # Get neighbors
        if node_idx in G:
            neighbors = list(nx.neighbors(G, node_idx))[:k_neighbors]
        else:
            print(f"Node {node_idx} not in graph or has no edges")
            return
        
        # Create subgraph
        subgraph_nodes = [node_idx] + neighbors
        subgraph = G.subgraph(subgraph_nodes)
        
        # Get node labels (fraud or normal)
        node_labels = {}
        for node in subgraph_nodes:
            label = 'Fraud' if self.data.y[node].item() == 1 else 'Normal'
            node_labels[node] = f"{node}\n({label})"
        
        # Get predictions — run only on the local subgraph to avoid OOM
        global_probs = self._infer_subgraph(node_idx, num_hops=2)
        
        # Node colors based on fraud probability (fall back to 0.0 if not in subgraph)
        node_colors = [global_probs.get(node, 0.0) for node in subgraph_nodes]
        
        # Plot
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(subgraph, k=2, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(subgraph, pos,
                              node_color=node_colors,
                              node_size=1000,
                              cmap='RdYlGn_r',
                              vmin=0, vmax=1,
                              alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(subgraph, pos, alpha=0.3, width=2)
        
        # Draw labels
        nx.draw_networkx_labels(subgraph, pos, node_labels, font_size=8)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', 
                                   norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca())
        cbar.set_label('Fraud Probability', rotation=270, labelpad=20)
        
        plt.title(f'Attention Neighborhood for Node {node_idx}')
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_hop_attention(self,
                               save_path: Optional[str] = None):
        """
        Visualize hop attention scores for Jump-Attention model
        
        Args:
            save_path: Path to save visualization
        """
        if not hasattr(self.model, 'get_hop_attention_scores'):
            print("Model does not support hop attention visualization")
            return
        
        # Forward pass to get hop attention — use a small random subset to avoid OOM
        with torch.no_grad():
            # Sample up to 1000 nodes for the hop-attention forward pass
            num_nodes = self.data.num_nodes
            sample_size = min(1000, num_nodes)
            sample_idx = torch.randperm(num_nodes)[:sample_size]
            subset, sub_edge_index, _, _ = k_hop_subgraph(
                node_idx=sample_idx,
                num_hops=2,
                edge_index=self.data.edge_index,
                relabel_nodes=True,
                num_nodes=num_nodes,
            )
            sub_x = self.data.x[subset]
            _ = self.model(sub_x, sub_edge_index)
            hop_scores = self.model.get_hop_attention_scores()
        
        if hop_scores is None:
            print("No hop attention scores available")
            return
        
        hop_scores = hop_scores.cpu().numpy()
        
        # Plot
        plt.figure(figsize=(10, 6))
        hops = np.arange(len(hop_scores))
        plt.bar(hops, hop_scores, color='steelblue', alpha=0.7)
        plt.xlabel('Hop Number', fontsize=12)
        plt.ylabel('Attention Weight', fontsize=12)
        plt.title('Jump-Attention: Hop Importance Scores', fontsize=14)
        plt.xticks(hops)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def explain_predictions(self,
                           node_indices: List[int],
                           output_dir: str = 'results/explanations'):
        """
        Generate explanations for multiple nodes
        
        Args:
            node_indices: List of node indices to explain
            output_dir: Directory to save explanations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for idx in node_indices:
            self.visualize_node_attention(
                idx,
                save_path=os.path.join(output_dir, f'node_{idx}_attention.png')
            )
        
        # Visualize hop attention if available
        self.visualize_hop_attention(
            save_path=os.path.join(output_dir, 'hop_attention.png')
        )
        
        print(f"Explanations saved to: {output_dir}")
