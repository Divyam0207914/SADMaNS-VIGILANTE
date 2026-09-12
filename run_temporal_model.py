"""
SADMaNS — Phase 6 Entry Point
==========================================

Runs the Temporal Sequence Builder and GRU World Model Smoke Test.
"""

import os
import sys
import time
import pandas as pd
import torch

from config import PROJECT_ROOT, DEVICE, SEQ_LEN, GRU_INPUT_DIM, GRU_HIDDEN_DIM, GLOBAL_STATE_DIM
from preprocessing.sequence_builder import build_temporal_sequences
from models.gru_world_model import GRUWorldModel

def _sep(char: str = "=", width: int = 70) -> str:
    return char * width


def main() -> int:
    print(_sep())
    print("  SADMaNS — PHASE 6: TEMPORAL SEQUENCE + GRU WORLD MODEL")
    print(_sep())
    print()

    data_dir = os.path.join(PROJECT_ROOT, "data")
    
    global_states_path = os.path.join(data_dir, "global_states.parquet")
    graph_embeds_path = os.path.join(data_dir, "graph_embeddings.parquet")
    
    if not os.path.isfile(global_states_path) or not os.path.isfile(graph_embeds_path):
        print("[ERROR] Required inputs not found. Run Phase 5 first.")
        return 1
        
    print(f"Loading global states:    {global_states_path}")
    global_states = pd.read_parquet(global_states_path)
    
    print(f"Loading graph embeddings: {graph_embeds_path}")
    graph_embeddings = pd.read_parquet(graph_embeds_path)
    
    # 1. BUILD SEQUENCES
    print("\n[ 1. BUILDING TEMPORAL SEQUENCES ]")
    start = time.time()
    sequences = build_temporal_sequences(global_states, graph_embeddings)
    print(f"  Generated {len(sequences)} valid continuous sequences in {time.time() - start:.2f}s")
    
    if len(sequences) == 0:
        print("  [ERROR] No valid sequences could be built. Check temporal continuity.")
        return 1
        
    # Validation
    seg_ids = [seq['segment_id'] for seq in sequences]
    unique_segs = set(seg_ids)
    print(f"  Sequences span {len(unique_segs)} temporal segments.")
    for seg in unique_segs:
        count = seg_ids.count(seg)
        print(f"    Segment {seg}: {count} sequences")
        
    # Check input dimensions of the first sequence
    first_seq = sequences[0]
    in_shape = first_seq['input_tensor'].shape
    out_shape = first_seq['target_tensor'].shape
    
    print(f"  Input tensor shape:  {in_shape} (Expected: [{SEQ_LEN}, 77])")
    print(f"  Target tensor shape: {out_shape} (Expected: [13])")
    
    if list(in_shape) != [SEQ_LEN, GRU_INPUT_DIM] or list(out_shape) != [GLOBAL_STATE_DIM]:
        print("  [ERROR] Invalid tensor dimensions.")
        return 1
        
    # Convert list of dicts to a single dictionary containing batched tensors and metadata
    # for easy saving/loading
    input_tensors = torch.stack([s['input_tensor'] for s in sequences])
    target_tensors = torch.stack([s['target_tensor'] for s in sequences])
    
    sequence_dataset = {
        'metadata': [
            {
                'sequence_id': s['sequence_id'],
                'segment_id': s['segment_id'],
                'input_start': s['input_start'],
                'input_end': s['input_end'],
                'target_window_id': s['target_window_id'],
                'target_window_start': s['target_window_start'],
            } for s in sequences
        ],
        'inputs': input_tensors,
        'targets': target_tensors
    }
    
    # 2. GRU SMOKE TEST
    print("\n[ 2. GRU WORLD MODEL SMOKE TEST ]")
    print(f"  Initializing GRU (Input: {GRU_INPUT_DIM}, Hidden: {GRU_HIDDEN_DIM}) on {DEVICE}")
    model = GRUWorldModel(
        input_dim=GRU_INPUT_DIM, 
        hidden_dim=GRU_HIDDEN_DIM, 
        num_layers=1, 
        output_dim=GLOBAL_STATE_DIM
    ).to(DEVICE)
    
    model.eval()
    
    # Take a small batch (e.g., 4 sequences)
    batch_size = min(4, len(sequences))
    x_batch = input_tensors[:batch_size].to(DEVICE)
    y_batch = target_tensors[:batch_size].to(DEVICE)
    
    start = time.time()
    with torch.no_grad():
        pred_next_state, final_hidden = model(x_batch, return_hidden=True)
        
    print(f"  Smoke test completed successfully in {time.time() - start:.2f}s")
    print(f"  Batch Input:       {x_batch.shape}")
    print(f"  Final Hidden:      {final_hidden.shape}")
    print(f"  Predicted Target:  {pred_next_state.shape}")
    print(f"  Actual Target:     {y_batch.shape}")
    
    # NaN check
    has_nan = torch.isnan(pred_next_state).any().item()
    print(f"  Outputs contain NaN: {has_nan}")
    if has_nan:
        print("  [ERROR] GRU produced NaN outputs.")
        return 1
    
    # 3. SAVING ARTIFACTS
    print("\n[ 3. SAVING ARTIFACTS ]")
    seq_path = os.path.join(data_dir, "temporal_sequences.pt")
    torch.save(sequence_dataset, seq_path)
    print(f"  Saved {seq_path}")
    
    print("\n" + _sep("-"))
    print("  PHASE 6 VALIDATION REPORT")
    print(_sep("-"))
    
    total_expected = 572
    
    print(f"Total Candidate Windows: {total_expected}")
    print(f"Total Valid Sequences:   {len(sequences)}")
    print(f"Sequence Length:         {SEQ_LEN}")
    print(f"Input Dimension:         {GRU_INPUT_DIM} (64 GCN + 13 Global)")
    print(f"Target Dimension:        {GLOBAL_STATE_DIM} (Global State)")
    
    print("\n" + _sep())
    print("  ✓ PHASE 6 TEMPORAL SEQUENCE + GRU COMPLETE")
    print(_sep())

    return 0

if __name__ == "__main__":
    sys.exit(main())