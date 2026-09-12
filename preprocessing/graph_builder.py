"""
SADMaNS — Graph Builder
==========================================

Phase 5: Constructs PyTorch Geometric graphs (destination-port 
interaction) for each 1-minute temporal window.
Selects top-K destination ports.
"""

import torch
from torch_geometric.data import Data
import pandas as pd
from typing import List

from config import TOP_K_PORTS, NODE_FEATURES

def select_top_k_ports(window_nodes: pd.DataFrame, k: int = TOP_K_PORTS) -> pd.DataFrame:
    """
    Selects top-K destination ports based on flow_count descending,
    with dst_port ascending as tie-breaker.
    """
    # Sort by flow_count descending, dst_port ascending
    sorted_nodes = window_nodes.sort_values(
        by=['flow_count', 'dst_port'], 
        ascending=[False, True]
    )
    # Take top K
    return sorted_nodes.head(k)

def build_window_graph(window_id: int, window_start: pd.Timestamp, segment_id: int, top_k_nodes: pd.DataFrame) -> Data:
    """
    Constructs a PyG Data object for a single window.
    Nodes: selected destination ports.
    Edges: fully connected (since they co-occur in the same window).
    Edge weights: min(flow_i, flow_j) / max(flow_i, flow_j)
    """
    num_nodes = len(top_k_nodes)
    
    if num_nodes == 0:
        return None
        
    # Node features (exact 17 dimensions)
    x = torch.tensor(top_k_nodes[NODE_FEATURES].values, dtype=torch.float32)
    
    # Destination ports (for tracking/metadata)
    ports = top_k_nodes['dst_port'].values
    flow_counts = top_k_nodes['flow_count'].values
    
    edge_index_list = []
    edge_weight_list = []
    
    # Fully connected undirected graph (including self loops if needed, 
    # but GCNConv adds self-loops automatically. We'll add standard undirected edges).
    # We'll just add all combinations i, j (including i=j if we want, but let's do all pairs).
    for i in range(num_nodes):
        for j in range(num_nodes):
            edge_index_list.append([i, j])
            
            # Edge weight formula: min / max
            fi = flow_counts[i]
            fj = flow_counts[j]
            weight = min(fi, fj) / max(fi, fj)
            edge_weight_list.append(weight)
            
    if len(edge_index_list) > 0:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight_list, dtype=torch.float32)
    else:
        # Edge case for 1 node, though the loop above creates a self-loop (0,0) weight 1.0.
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty(0, dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
    data.window_id = window_id
    data.window_start = window_start
    data.segment_id = segment_id
    data.dst_ports = ports  # Metadata
    
    return data

def build_all_graphs(node_states: pd.DataFrame) -> List[Data]:
    """
    Iterates over all windows and builds the graphs.
    """
    graphs = []
    
    # Group by window_id
    grouped = node_states.groupby('window_id')
    
    for w_id, w_nodes in grouped:
        w_start = w_nodes['window_start'].iloc[0]
        seg_id = w_nodes['segment_id'].iloc[0]
        
        # Select top K
        top_k = select_top_k_ports(w_nodes)
        
        # Build graph
        g = build_window_graph(w_id, w_start, seg_id, top_k)
        if g is not None:
            graphs.append(g)
            
    return graphs
