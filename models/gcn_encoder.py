"""
SADMaNS — GCN Encoder
==========================================

Phase 5: Implementation of the Graph Convolutional Network (GCN) encoder.
Maps 17-dimensional node features to 64-dimensional graph embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GCNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, dropout: float = 0.2):
        super(GCNEncoder, self).__init__()
        
        # GCNConv(17 -> 64)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        
        # GCNConv(64 -> 64)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for the GCN encoder.
        
        Args:
            x (Tensor): Node feature matrix [num_nodes, in_channels]
            edge_index (LongTensor): Graph connectivity [2, num_edges]
            edge_weight (Tensor): Edge weights [num_edges]
            batch (LongTensor, optional): Batch vector assigning each node to a graph.
            
        Returns:
            Tensor: Graph-level embedding [num_graphs, hidden_channels]
        """
        # First GCN Layer
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second GCN Layer
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        
        # Global Mean Pooling
        # If batch is None, assume all nodes belong to a single graph
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        graph_embed = global_mean_pool(x, batch)
        
        return graph_embed
