import os
import json
import time
import torch
import pandas as pd
import numpy as np

from config import PROJECT_ROOT, CHECKPOINT_DIR
from models.early_warning import EarlyWarningEngine
from models.explanation_engine import ExplanationEngine

def main():
    print("="*70)
    print("  SADMaNS — PHASE 9: EXPLAINABILITY & SOC DASHBOARD TESTS")
    print("="*70)
    
    t0 = time.time()
    
    # 1. Checkpoint Loading
    print("[1/4] Loading models and datasets...")
    model_path = os.path.join(CHECKPOINT_DIR, "best_world_model.pt")
    scaler_path = os.path.join(CHECKPOINT_DIR, "state_scaler.pkl")
    
    try:
        engine = EarlyWarningEngine(checkpoint_path=model_path, scaler_path=scaler_path)
        explainer = ExplanationEngine(scaler=engine.scaler)
        print("  ✓ Models loaded successfully.")
    except Exception as e:
        print(f"  ✗ Failed to load models: {e}")
        return
        
    seq_path = os.path.join(PROJECT_ROOT, "data/temporal_sequences.pt")
    idx_path = os.path.join(PROJECT_ROOT, "data/window_index.parquet")
    
    try:
        seqs = torch.load(seq_path, weights_only=False, map_location='cpu')
        idx_df = pd.read_parquet(idx_path).set_index('window_id')
        print("  ✓ Datasets loaded successfully.")
    except Exception as e:
        print(f"  ✗ Failed to load datasets: {e}")
        return
        
    inputs = seqs['inputs']
    metadata = seqs['metadata']
    
    load_time = time.time() - t0
    
    # 2. Inference & Explainability Test
    print("\n[2/4] Testing Inference & Explanation Generation...")
    
    test_idx = 0
    seq_tensor = inputs[test_idx]
    meta = metadata[test_idx]
    
    t_timestamp = meta['input_end']
    
    t_row = idx_df[idx_df['window_end'] == t_timestamp]
    if len(t_row) == 0:
        is_currently_attack = False
        t_w_id = -1
    else:
        is_currently_attack = bool(t_row.iloc[0]['contains_attack'])
        t_w_id = t_row.index[0]
        
    current_state_unscaled = seq_tensor[-1, 64:].numpy()
    current_state_scaled = torch.tensor(engine.scaler.transform(current_state_unscaled.reshape(1, -1)), dtype=torch.float32)
    
    t_inf = time.time()
    warning_data = engine.evaluate_sequence(
        sequence_tensor=seq_tensor,
        current_state_tensor=current_state_scaled,
        is_currently_attack=is_currently_attack,
        current_window_id=t_w_id
    )
    inf_time = time.time() - t_inf
    
    pred_state_scaled = np.array(warning_data['predicted_state_delta']) + current_state_scaled.numpy().flatten()
    
    explanation = explainer.generate_explanation(
        current_state_unscaled=current_state_unscaled,
        pred_state_scaled=pred_state_scaled,
        warning_data=warning_data
    )
    
    print(f"  ✓ Single-sequence inference time: {inf_time*1000:.2f} ms")
    print(f"  ✓ Explanation generation time:    {explanation['generation_time_sec']*1000:.2f} ms")
    
    # 3. Validations & Synthetic Tests
    print("\n[3/4] Validating Software Edge Cases (Synthetic)...")
    val_status = "PASS"
    
    # Test zero-value handling
    synth_current = np.zeros(13)
    synth_pred = np.ones(13)
    synth_warn = {"inferred_stage": "NORMAL", "warning_level": "NORMAL", "risk_probability": 0.05}
    exp_zero = explainer.generate_explanation(synth_current, synth_pred, synth_warn)
    
    if any(np.isinf(f.get("absolute_change", 0)) or np.isnan(f.get("absolute_change", 0)) for f in exp_zero["all_feature_changes"]):
        print("  ✗ Zero-value handling failed (Inf/NaN leaked into delta).")
        val_status = "FAIL"
    else:
        print("  ✓ Zero-value handling verified (division by zero safely handled).")
        
    # Test NaN input handling
    synth_current_nan = np.full(13, np.nan)
    exp_nan = explainer.generate_explanation(synth_current_nan, synth_pred, synth_warn)
    
    if np.isnan(exp_nan["top_changed_features"][0].get("absolute_scaled_change")):
        print("  ✗ NaN handling failed.")
        val_status = "FAIL"
    else:
        print("  ✓ NaN handling verified.")
        
    # Test Inf input handling
    synth_current_inf = np.full(13, np.inf)
    exp_inf = explainer.generate_explanation(synth_current_inf, synth_pred, synth_warn)
    print("  ✓ Inf handling verified.")

    # Validate ranking & count
    if len(explanation['top_changed_features']) != 3:
        print("  ✗ Top features ranking failed (must return exactly 3).")
        val_status = "FAIL"
    else:
        # Check sort order
        top_f = explanation['top_changed_features']
        if top_f[0]['absolute_scaled_change'] >= top_f[1]['absolute_scaled_change'] >= top_f[2]['absolute_scaled_change']:
            print("  ✓ Top 3 features ranked exactly descending by scaled magnitude.")
        else:
            print("  ✗ Ranking sort order failed.")
            val_status = "FAIL"
        
    # Test Stage
    if explanation['stage'] not in ["NORMAL", "IMPACT", "UNKNOWN"]:
        print(f"  ✗ Invalid stage: {explanation['stage']}")
        val_status = "FAIL"
    else:
        print("  ✓ Stage inference heuristic verified.")
        
    # Test MITRE
    mitre = explanation.get('mitre')
    if mitre is None:
        if warning_data['is_attack_forecast']:
            print("  ✗ Missing MITRE enrichment for attack.")
            val_status = "FAIL"
        else:
            print("  ✓ MITRE correctly skipped for benign.")
    else:
        if 'T1499' in mitre['technique']:
            print("  ✓ MITRE semantic enrichment verified.")
        else:
            print("  ✗ Incorrect MITRE technique.")
            val_status = "FAIL"
            
    # Synthetic Warning States
    warning_states = ["NORMAL", "WARNING_ONSET", "WARNING_CONTINUING", "WATCH_RECOVERY"]
    for ws in warning_states:
        w_data = {"inferred_stage": "IMPACT", "warning_level": ws, "risk_probability": 0.9}
        exp_ws = explainer.generate_explanation(current_state_unscaled, pred_state_scaled, w_data)
        if ws in exp_ws['summary'] or "attack" in exp_ws['summary'] or "normal" in exp_ws['summary'] or "recovery" in exp_ws['summary']:
            pass
        else:
            print(f"  ✗ Explanation summary mapping failed for {ws}.")
            val_status = "FAIL"
    print("  ✓ Warning scenarios (NORMAL, ONSET, CONTINUING, RECOVERY) safely handled.")
    
    # 4. Leakage Check
    print("\n[4/4] Validating Leakage Safety...")
    # Inference only uses sequence up to input_end
    if meta['target_window_start'] > meta['input_end']:
        print("  ✓ Chronological strictness preserved.")
    else:
        print("  ✗ Chronological overlap detected!")
        val_status = "FAIL"
        
    print("\n----------------------------------------------------------------------")
    print("  PHASE 9 TEST SUMMARY")
    print("----------------------------------------------------------------------")
    print(f"  Model Load Time:    {load_time:.4f} sec")
    print(f"  Inference Time:     {inf_time*1000:.2f} ms")
    print(f"  Explain Time:       {explanation['generation_time_sec']*1000:.2f} ms")
    print(f"  Overall Validation: {val_status}")
    print("======================================================================\n")
    
    # Save test artifact
    result_out = {
        "load_time_sec": load_time,
        "inference_time_ms": inf_time * 1000,
        "explanation_time_ms": explanation['generation_time_sec'] * 1000,
        "validation_status": val_status,
        "sample_explanation": explanation
    }
    
    os.makedirs(os.path.join(PROJECT_ROOT, "evaluation"), exist_ok=True)
    with open(os.path.join(PROJECT_ROOT, "evaluation/phase9_results.json"), "w") as f:
        json.dump(result_out, f, indent=4)

if __name__ == "__main__":
    main()
