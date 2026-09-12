import os
import json
import torch
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from models.full_world_model import FullWorldModel
from config import PROJECT_ROOT, DEVICE

def check_self_loops(graph_builder_path):
    with open(graph_builder_path, 'r') as f:
        content = f.read()
    if 'self_loop' in content.lower():
        return True
    return False

def check_file_size(path):
    if os.path.exists(path):
        return os.path.getsize(path)
    return -1

def run_audit():
    results = {}
    
    # 1. FILE EXISTENCE AND SIZES
    artifacts = [
        "data/global_states.parquet",
        "data/node_states.parquet",
        "data/graphs.pt",
        "data/graph_embeddings.parquet",
        "data/temporal_sequences.pt",
        "data/window_index.parquet",
        "checkpoints/best_world_model.pt",
        "checkpoints/state_scaler.pkl",
        "evaluation/results.json",
        "evaluation/training_history.json"
    ]
    
    file_info = {}
    for a in artifacts:
        p = os.path.join(PROJECT_ROOT, a)
        file_info[a] = {
            "exists": os.path.exists(p),
            "size_bytes": check_file_size(p)
        }
    results["artifacts"] = file_info

    # 2. DATA AUDIT
    try:
        window_index = pd.read_parquet(os.path.join(PROJECT_ROOT, "data/window_index.parquet"))
        global_states = pd.read_parquet(os.path.join(PROJECT_ROOT, "data/global_states.parquet"))
        node_states = pd.read_parquet(os.path.join(PROJECT_ROOT, "data/node_states.parquet"))
        
        results["data_audit"] = {
            "window_index_count": len(window_index),
            "global_states_count": len(global_states),
            "node_states_count": len(node_states),
            "global_states_cols": len(global_states.columns) - 1, # -1 for window_id
            "node_states_cols": len(node_states.columns) - 2 # -2 for window_id, dst_port
        }
    except Exception as e:
        results["data_audit"] = {"error": str(e)}

    # 3. GRAPH AUDIT
    try:
        graphs = torch.load(os.path.join(PROJECT_ROOT, "data/graphs.pt"), weights_only=False)
        g_shapes = [g.x.shape for g in graphs]
        max_nodes = max([s[0] for s in g_shapes])
        
        actual_self_loop_count = 0
        graphs_with_self_loops = 0
        total_self_loop_edges = 0
        
        edge_validation = {
            "valid_dims": True,
            "source_ge_0": True,
            "dest_ge_0": True,
            "idx_lt_nodes": True,
            "weights_finite": True,
            "weights_in_range": True,
            "no_nan_inf": True,
            "is_undirected": True
        }
        
        for g in graphs:
            num_nodes = g.x.size(0)
            e_idx = g.edge_index
            e_attr = getattr(g, 'edge_attr', None)
            if e_attr is None:
                e_attr = getattr(g, 'edge_weight', None)
            
            if e_idx.dim() != 2 or e_idx.size(0) != 2:
                edge_validation["valid_dims"] = False
                
            src = e_idx[0]
            dst = e_idx[1]
            
            # Check self loops
            self_loops = (src == dst).sum().item()
            if self_loops > 0:
                actual_self_loop_count += 1
                graphs_with_self_loops += 1
                total_self_loop_edges += self_loops
                
            if (src < 0).any().item(): edge_validation["source_ge_0"] = False
            if (dst < 0).any().item(): edge_validation["dest_ge_0"] = False
            if (src >= num_nodes).any().item() or (dst >= num_nodes).any().item():
                edge_validation["idx_lt_nodes"] = False
                
            if e_attr is not None:
                if not torch.isfinite(e_attr).all().item():
                    edge_validation["weights_finite"] = False
                    edge_validation["no_nan_inf"] = False
                if (e_attr <= 0).any().item() or (e_attr > 1).any().item():
                    edge_validation["weights_in_range"] = False
                    
            # check undirected
            # we can check by converting to set of tuples
            edges = set([tuple(x) for x in e_idx.t().tolist()])
            for s, d in edges:
                if (d, s) not in edges:
                    edge_validation["is_undirected"] = False
                    break
        
        results["graph_audit"] = {
            "total_graphs": len(graphs),
            "max_nodes": max_nodes,
            "feature_dim": g_shapes[0][1],
            "actual_self_loop_count": actual_self_loop_count,
            "graphs_with_self_loops": graphs_with_self_loops,
            "total_self_loop_edges": total_self_loop_edges,
            "edge_validation": edge_validation,
            "gcn_training_status": "UNTRAINED"
        }
    except Exception as e:
        results["graph_audit"] = {"error": str(e)}

    # 4. SEQUENCE AUDIT
    try:
        seqs = torch.load(os.path.join(PROJECT_ROOT, "data/temporal_sequences.pt"), weights_only=False)
        results["sequence_audit"] = {
            "total_sequences": len(seqs['metadata']),
            "input_shape": list(seqs['inputs'].shape),
            "target_shape": list(seqs['targets'].shape)
        }
    except Exception as e:
        results["sequence_audit"] = {"error": str(e)}
        
    # 5. CHECKPOINT & INFERENCE AUDIT
    try:
        chk_path = os.path.join(PROJECT_ROOT, "checkpoints/best_world_model.pt")
        chk = torch.load(chk_path, weights_only=False, map_location='cpu')
        
        results["checkpoint_audit"] = {
            "has_model_state": 'model_state_dict' in chk,
            "has_config": 'config' in chk,
            "has_threshold": 'threshold' in chk,
            "has_scaler_path": 'scaler_path' in chk
        }
        
        # Test inference
        model = FullWorldModel()
        model.load_state_dict(chk['model_state_dict'])
        model.eval()
        
        # take first sequence
        sample_input = seqs['inputs'][0:1].cpu()
        with torch.no_grad():
            out = model.inference(sample_input, risk_threshold=chk.get('threshold', 0.5))
            
        results["inference_audit"] = {
            "success": True,
            "keys_returned": list(out[0].keys()),
            "next_state_shape": list(out[0]['next_state'].shape)
        }
    except Exception as e:
        results["checkpoint_audit"] = {"error": str(e)}
        results["inference_audit"] = {"error": str(e)}
        
    with open(os.path.join(PROJECT_ROOT, "evaluation/phase5_7_audit.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print("Audit script completed.")

if __name__ == "__main__":
    run_audit()
