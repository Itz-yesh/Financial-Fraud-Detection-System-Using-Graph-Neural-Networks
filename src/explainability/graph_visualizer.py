"""
Graph Visualizer
Creates graph visualizations highlighting fraud patterns
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import Optional, List
import os


class GraphVisualizer:
    """Visualizes transaction graphs with fraud patterns"""
    
    def __init__(self, data):
        """
        Initialize graph visualizer
        
        Args:
            data: Graph data
        """
        self.data = data
    
    def visualize_subgraph(self,
                          center_nodes: List[int],
                          k_hops: int = 2,
                          save_path: Optional[str] = None):
        """
        Visualize subgraph around center nodes
        
        Args:
            center_nodes: List of center node indices
            k_hops: Number of hops to include
            save_path: Path to save visualization
        """
        # Build NetworkX graph
        edge_index = self.data.edge_index.cpu().numpy()
        G = nx.Graph()
        G.add_edges_from(edge_index.T)
        
        # Get k-hop subgraph
        subgraph_nodes = set(center_nodes)
        for _ in range(k_hops):
            new_nodes = set()
            for node in subgraph_nodes:
                if node in G:
                    new_nodes.update(G.neighbors(node))
            subgraph_nodes.update(new_nodes)
        
        subgraph = G.subgraph(list(subgraph_nodes))
        
        # Get node colors based on fraud label
        node_colors = []
        for node in subgraph.nodes():
            if self.data.y[node].item() == 1:
                node_colors.append('red')  # Fraud
            else:
                node_colors.append('lightblue')  # Normal
        
        # Plot
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(subgraph, k=1, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(subgraph, pos,
                              node_color=node_colors,
                              node_size=300,
                              alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(subgraph, pos, alpha=0.2, width=1)
        
        # Highlight center nodes
        nx.draw_networkx_nodes(subgraph, pos,
                              nodelist=center_nodes,
                              node_color='yellow',
                              node_size=500,
                              edgecolors='black',
                              linewidths=2)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Fraud'),
            Patch(facecolor='lightblue', label='Normal'),
            Patch(facecolor='yellow', edgecolor='black', label='Center Node')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.title(f'Transaction Graph ({len(subgraph.nodes())} nodes, {len(subgraph.edges())} edges)')
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_fraud_clusters(self,
                                 max_nodes: int = 500,
                                 save_path: Optional[str] = None):
        """
        Visualize fraud clusters in the graph
        
        Args:
            max_nodes: Maximum number of nodes to visualize
            save_path: Path to save visualization
        """
        # Build NetworkX graph
        edge_index = self.data.edge_index.cpu().numpy()
        G = nx.Graph()
        G.add_edges_from(edge_index.T)
        
        # Sample nodes if graph is too large
        if len(G.nodes()) > max_nodes:
            # Sample fraud nodes and some normal nodes
            fraud_nodes = [i for i in G.nodes() if self.data.y[i].item() == 1]
            normal_nodes = [i for i in G.nodes() if self.data.y[i].item() == 0]
            
            sampled_fraud = fraud_nodes[:min(len(fraud_nodes), max_nodes // 2)]
            sampled_normal = normal_nodes[:min(len(normal_nodes), max_nodes // 2)]
            sampled_nodes = sampled_fraud + sampled_normal
            
            G = G.subgraph(sampled_nodes)
        
        # Get node colors
        node_colors = ['red' if self.data.y[node].item() == 1 else 'lightblue' 
                      for node in G.nodes()]
        
        # Plot
        plt.figure(figsize=(16, 12))
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # Draw
        nx.draw_networkx_nodes(G, pos,
                              node_color=node_colors,
                              node_size=100,
                              alpha=0.7)
        nx.draw_networkx_edges(G, pos, alpha=0.1, width=0.5)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Fraud'),
            Patch(facecolor='lightblue', label='Normal')
        ]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
        
        plt.title('Fraud Clusters in Transaction Graph', fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
