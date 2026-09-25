import os
import time
import json
import numpy as np
import pandas as pd
import torch

from config import PROJECT_ROOT, CHECKPOINT_DIR, SCALER_PATH, GLOBAL_STATE_FEATURES, DEVICE
from models.early_warning import EarlyWarningEngine
from models.explanation_engine import ExplanationEngine

def safe_float(val):
    if val is None or pd.isna(val) or np.isnan(val) or np.isinf(val):
        return None
    return float(val)

def precompute_dashboard_cache():
    print("==================================================")
    print(" DASHBOARD PRECOMPUTATION")
    print("==================================================")

    model_path = os.path.join(CHECKPOINT_DIR, "best_world_model.pt")
    scaler_path = os.path.join(CHECKPOINT_DIR, "state_scaler.pkl")
    seq_path = os.path.join(PROJECT_ROOT, "data/temporal_sequences.pt")
    idx_path = os.path.join(PROJECT_ROOT, "data/window_index.parquet")
    
    out_path = os.path.join(PROJECT_ROOT, "data/dashboard_cache.json")

    print(f"[1/5] Loading models and datasets...")
    engine = EarlyWarningEngine(checkpoint_path=model_path, scaler_path=scaler_path)
    explainer = ExplanationEngine(scaler=engine.scaler)
    
    seqs = torch.load(seq_path, weights_only=False, map_location='cpu')
    idx_df = pd.read_parquet(idx_path).set_index('window_id')

    inputs = seqs['inputs']
    metadata = seqs['metadata']
    total_seqs = len(inputs)
    
    print(f"      Found {total_seqs} sequences.")
    
    cache_data = {
        "version": 1,
        "total_sequences": total_seqs,
        "sequences": {}
    }
    
    print(f"[2/5] Running inference and explanation for all sequences...")
    t0 = time.time()
    
    for seq_idx in range(total_seqs):
        if seq_idx % 50 == 0:
            print(f"      Processing sequence {seq_idx}/{total_seqs}...")
            
        seq_tensor = inputs[seq_idx]
        meta = metadata[seq_idx]
        t_timestamp = meta['input_end']
        
        t_row = idx_df[idx_df['window_end'] == t_timestamp]
        if len(t_row) == 0:
            is_currently_attack = False
            t_w_id = -1
        else:
            is_currently_attack = bool(t_row.iloc[0]['contains_attack'])
            t_w_id = int(t_row.index[0])
            
        current_state_unscaled = seq_tensor[-1, 64:].numpy()
        current_state_scaled = torch.tensor(engine.scaler.transform(current_state_unscaled.reshape(1, -1)), dtype=torch.float32)
        
        t_inf_start = time.time()
        warning_data = engine.evaluate_sequence(
            sequence_tensor=seq_tensor,
            current_state_tensor=current_state_scaled,
            is_currently_attack=is_currently_attack,
            current_window_id=t_w_id
        )
        inf_time = time.time() - t_inf_start
        
        pred_state_scaled = np.array(warning_data['predicted_state_delta']) + current_state_scaled.numpy().flatten()
        
        t_exp_start = time.time()
        explanation = explainer.generate_explanation(
            current_state_unscaled=current_state_unscaled,
            pred_state_scaled=pred_state_scaled,
            warning_data=warning_data
        )
        exp_time = time.time() - t_exp_start
        
        # Clean explanation features for JSON
        cleaned_top = []
        for f in explanation.get("top_changed_features", []):
            cleaned_top.append({
                "feature": f.get("feature"),
                "current_value": safe_float(f.get("current_value")),
                "forecasted_value": safe_float(f.get("forecasted_value")),
                "signed_change": safe_float(f.get("signed_change")),
                "rank": f.get("rank")
            })
            
        cleaned_all = []
        for f in explanation.get("all_feature_changes", []):
            cleaned_all.append({
                "feature": f.get("feature"),
                "current_value": safe_float(f.get("current_value")),
                "forecasted_value": safe_float(f.get("forecasted_value")),
                "absolute_change": safe_float(f.get("absolute_change")),
                "relative_change": f.get("relative_change") if isinstance(f.get("relative_change"), str) else safe_float(f.get("relative_change")),
                "signed_scaled_change": safe_float(f.get("signed_scaled_change")),
                "absolute_scaled_change": safe_float(f.get("absolute_scaled_change"))
            })
        
        cache_data["sequences"][str(seq_idx)] = {
            "seq_idx": seq_idx,
            "timestamp": str(t_timestamp),
            "is_currently_attack": bool(is_currently_attack),
            "current_window_id": t_w_id,
            "risk_probability": safe_float(explanation.get("risk_probability", 0.0)),
            "warning_level": explanation.get("warning_level", "NORMAL"),
            "predicted_attack_type": explanation.get("predicted_attack_type", "Unknown"),
            "stage": explanation.get("stage", "NORMAL"),
            "mitre": explanation.get("mitre", None),
            "summary": explanation.get("summary", ""),
            "top_changed_features": cleaned_top,
            "all_feature_changes": cleaned_all,
            "inf_time": inf_time,
            "exp_time": exp_time
        }
        
    duration = time.time() - t0
    print(f"[3/5] Inference complete in {duration:.2f}s")
    
    print(f"[4/5] Saving to {out_path}...")
    with open(out_path, 'w') as f:
        json.dump(cache_data, f, indent=2)
        
    print(f"[5/5] Validating output...")
    with open(out_path, 'r') as f:
        loaded = json.load(f)
        
    assert loaded["total_sequences"] == total_seqs, "Mismatch in sequence count"
    assert len(loaded["sequences"]) == total_seqs, "Mismatch in cached dictionary length"
    
    # Check a sample
    sample = loaded["sequences"]["0"]
    assert "risk_probability" in sample
    assert "warning_level" in sample
    assert "predicted_attack_type" in sample
    assert "timestamp" in sample
    
    print("\nDashboard precomputation complete.")
    print(f"Sequences processed: {total_seqs}")
    print(f"Cache file: {out_path}")
    print(f"Cache file size: {os.path.getsize(out_path) / 1024:.2f} KB")
    print("Validation: PASSED")

if __name__ == "__main__":
    precompute_dashboard_cache()
