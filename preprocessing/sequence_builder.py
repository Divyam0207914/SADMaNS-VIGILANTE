"""
SADMaNS — Sequence Builder
==========================================

Phase 6: Constructs temporal sequences of network states.
Combines 64-dim graph embeddings + 13-dim global states = 77 dims.
Creates [5, 77] input sequences and [13] target next states.
Respects temporal continuity and segment boundaries.
"""

import pandas as pd
import torch
from typing import List, Dict, Any

from config import SEQ_LEN, GLOBAL_STATE_FEATURES

def build_temporal_sequences(
    global_states: pd.DataFrame, 
    graph_embeddings: pd.DataFrame
) -> List[Dict[str, Any]]:
    """
    Constructs chronologically continuous temporal sequences.
    Ensures that t-4 to t+1 are consecutive 1-minute windows
    within the same segment.
    """
    # Sort chronologically
    global_states = global_states.sort_values(by='window_start').reset_index(drop=True)
    graph_embeddings = graph_embeddings.sort_values(by='window_start').reset_index(drop=True)
    
    # Merge global states and graph embeddings on window_id, window_start, segment_id
    combined = pd.merge(
        global_states, 
        graph_embeddings, 
        on=['window_id', 'window_start', 'segment_id'],
        how='inner'
    )
    
    # Expected graph embedding columns
    embed_cols = [f'embed_{i}' for i in range(64)]
    
    sequences = []
    
    # We need SEQ_LEN + 1 consecutive windows for a valid (Input, Target) pair
    required_length = SEQ_LEN + 1
    
    num_windows = len(combined)
    
    for i in range(num_windows - required_length + 1):
        # Extract candidate slice
        candidate = combined.iloc[i : i + required_length]
        
        # 1. Segment Check: All windows must belong to the same segment
        if candidate['segment_id'].nunique() > 1:
            continue
            
        # 2. Continuity Check: Each step must be exactly 1 minute apart
        timestamps = candidate['window_start'].tolist()
        is_continuous = True
        for j in range(1, len(timestamps)):
            diff = (timestamps[j] - timestamps[j-1]).total_seconds()
            if diff != 60.0:
                is_continuous = False
                break
                
        if not is_continuous:
            continue
            
        # Candidate is valid!
        
        # Build Input (first SEQ_LEN steps)
        input_df = candidate.iloc[:SEQ_LEN]
        
        # Feature vector: [embed_0..63] + [global_1..13]
        input_embeds = input_df[embed_cols].values
        input_globals = input_df[GLOBAL_STATE_FEATURES].values
        
        # Concatenate features
        # Shape: [SEQ_LEN, 77]
        import numpy as np
        input_features = np.concatenate([input_embeds, input_globals], axis=1).astype(np.float32)
        input_tensor = torch.tensor(input_features, dtype=torch.float32)
        
        # Build Target (last step)
        target_df = candidate.iloc[-1]
        target_globals = target_df[GLOBAL_STATE_FEATURES].values.astype(np.float32)
        target_tensor = torch.tensor(target_globals, dtype=torch.float32)
        
        # Sequence metadata
        seq_dict = {
            'sequence_id': len(sequences),
            'segment_id': int(target_df['segment_id']),
            'input_start': input_df['window_start'].iloc[0],
            'input_end': input_df['window_start'].iloc[-1],
            'target_window_id': int(target_df['window_id']),
            'target_window_start': target_df['window_start'],
            'input_tensor': input_tensor,
            'target_tensor': target_tensor
        }
        
        sequences.append(seq_dict)
        
    return sequences
