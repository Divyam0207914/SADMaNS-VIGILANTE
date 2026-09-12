"""
SADMaNS — Phase 5 Entry Point
==========================================

Runs the Network State Builder, Top-K Graph Construction, 
and GCN Encoder Smoke Test.
"""

import os
import sys
import time

import pandas as pd
import torch

from config import PROJECT_ROOT, DEVICE, NODE_FEATURE_DIM, GCN_HIDDEN_DIM, GCN_DROPOUT
from preprocessing.state_builder import build_global_states, build_node_states
from preprocessing.graph_builder import build_all_graphs
from models.gcn_encoder import GCNEncoder


def _sep(char: str = "=", width: int = 70) -> str:
    return char * width


def main() -> int:
    print(_sep())
    print("  SADMaNS — PHASE 5: STATE + GRAPH + GCN")
    print(_sep())
    print()

    data_dir = os.path.join(PROJECT_ROOT, "data")
    
    clean_path = os.path.join(data_dir, "02-15-2018_cleaned.parquet")
    window_index_path = os.path.join(data_dir, "window_index.parquet")
    
    if not os.path.isfile(clean_path) or not os.path.isfile(window_index_path):
        print("[ERROR] Required inputs not found. Run Phase 3 and 4 first.")
        return 1
        
    from preprocessing.windowing import create_time_windows
    
    print(f"Loading cleaned dataset: {clean_path}")
    df_clean = pd.read_parquet(clean_path)
    print("Flooring timestamps to create window_start column...")
    df_clean = create_time_windows(df_clean)
    
    print(f"Loading window index:    {window_index_path}")
    window_index = pd.read_parquet(window_index_path)
    
    # Filter only populated windows for state building
    populated_windows = window_index[window_index['flow_count'] > 0]
    
    print("\n[ 1. BUILDING GLOBAL STATES ]")
    start = time.time()
    global_states = build_global_states(df_clean, populated_windows)
    print(f"  Generated {len(global_states)} global states in {time.time() - start:.2f}s")
    
    # Validation 1: Consistency Global Flow Count
    global_flow_sum = global_states['flow_count'].sum()
    print(f"  Total global flow count: {global_flow_sum}")
    if global_flow_sum != len(df_clean):
        print(f"  [WARNING] Global state flow sum ({global_flow_sum}) != Cleaned rows ({len(df_clean)})")
    
    print("\n[ 2. BUILDING NODE STATES ]")
    start = time.time()
    node_states = build_node_states(df_clean, populated_windows)
    print(f"  Generated {len(node_states)} node states in {time.time() - start:.2f}s")
    
    # Validation 2: Consistency Node Flow Count
    node_flow_sum = node_states['flow_count'].sum()
    print(f"  Total node flow count: {node_flow_sum}")
    if node_flow_sum != global_flow_sum:
        print(f"  [WARNING] Node state flow sum ({node_flow_sum}) != Global flow sum ({global_flow_sum})")
        
    print("\n[ 3. BUILDING GRAPHS (TOP-K) ]")
    start = time.time()
    graphs = build_all_graphs(node_states)
    print(f"  Generated {len(graphs)} graphs in {time.time() - start:.2f}s")
    
    print("\n[ 4. GCN SMOKE TEST ]")
    print(f"  Initializing GCN (Input: {NODE_FEATURE_DIM}, Hidden: {GCN_HIDDEN_DIM}) on {DEVICE}")
    model = GCNEncoder(in_channels=NODE_FEATURE_DIM, hidden_channels=GCN_HIDDEN_DIM, dropout=GCN_DROPOUT)
    model = model.to(DEVICE)
    model.eval()  # Smoke test, no training
    
    embeddings = []
    
    start = time.time()
    with torch.no_grad():
        for i, g in enumerate(graphs):
            g = g.to(DEVICE)
            
            # Forward pass
            out = model(g.x, g.edge_index, g.edge_attr)
            
            # Move back to CPU for storage
            out = out.cpu().numpy()
            
            # Expect shape [1, 64]
            if out.shape != (1, GCN_HIDDEN_DIM):
                print(f"  [ERROR] Graph {i} output shape {out.shape} != (1, {GCN_HIDDEN_DIM})")
                return 1
                
            embeddings.append({
                'window_id': g.window_id,
                'window_start': g.window_start,
                'segment_id': g.segment_id,
                'embedding': out[0]
            })
            
    print(f"  Smoke test completed successfully in {time.time() - start:.2f}s")
    
    print("\n[ 5. SAVING ARTIFACTS ]")
    
    global_path = os.path.join(data_dir, "global_states.parquet")
    global_states.to_parquet(global_path, index=False)
    print(f"  Saved {global_path}")
    
    node_path = os.path.join(data_dir, "node_states.parquet")
    node_states.to_parquet(node_path, index=False)
    print(f"  Saved {node_path}")
    
    graphs_path = os.path.join(data_dir, "graphs.pt")
    torch.save(graphs, graphs_path)
    print(f"  Saved {graphs_path}")
    
    # Prepare embeddings flat dataframe
    embed_cols = [f'embed_{i}' for i in range(GCN_HIDDEN_DIM)]
    embed_records = []
    for e in embeddings:
        rec = {
            'window_id': e['window_id'],
            'window_start': e['window_start'],
            'segment_id': e['segment_id']
        }
        for i, val in enumerate(e['embedding']):
            rec[f'embed_{i}'] = val
        embed_records.append(rec)
        
    embed_df = pd.DataFrame(embed_records)
    embed_path = os.path.join(data_dir, "graph_embeddings.parquet")
    embed_df.to_parquet(embed_path, index=False)
    print(f"  Saved {embed_path}")
    
    # Final Validation Checks for Report
    
    # NaN/Inf checks
    global_nan = global_states.isna().sum().sum()
    node_nan = node_states.isna().sum().sum()
    embed_nan = embed_df.isna().sum().sum()
    
    print("\n" + _sep("-"))
    print("  PHASE 5 VALIDATION REPORT")
    print(_sep("-"))
    
    print(f"\n[ STATES ]")
    print(f"Global States:       {len(global_states)} (Expected: {len(populated_windows)})")
    print(f"Node States:         {len(node_states)}")
    print(f"Global NaN count:    {global_nan}")
    print(f"Node NaN count:      {node_nan}")
    
    print(f"\n[ GRAPHS ]")
    print(f"Total Graphs:        {len(graphs)}")
    nodes_per_graph = [g.num_nodes for g in graphs]
    print(f"Nodes per Graph:     Min={min(nodes_per_graph)}, Max={max(nodes_per_graph)}")
    print(f"Top-K applied:       {max(nodes_per_graph) <= 50}")
    
    print(f"\n[ GCN EMBEDDINGS ]")
    print(f"Total Embeddings:    {len(embed_df)}")
    print(f"Embedding NaN count: {embed_nan}")
    
    # Cross Check Attack Windows
    attack_wids = window_index[window_index['contains_attack'] == True]['window_id'].values
    embed_wids = embed_df['window_id'].values
    all_attacks_present = all(wid in embed_wids for wid in attack_wids)
    print(f"Attack Windows Preserved: {all_attacks_present}")
    
    print("\n" + _sep())
    print("  ✓ PHASE 5 NETWORK STATE + GRAPH + GCN COMPLETE")
    print(_sep())

    return 0

if __name__ == "__main__":
    sys.exit(main())
