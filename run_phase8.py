import os
import json
import torch
import pandas as pd
import numpy as np
from config import PROJECT_ROOT, CHECKPOINT_DIR, SCALER_PATH
from models.early_warning import EarlyWarningEngine
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, confusion_matrix

def main():
    print("="*70)
    print("  SADMaNS — PHASE 8: EARLY ATTACK WARNING")
    print("="*70)
    
    # Load data
    seq_path = os.path.join(PROJECT_ROOT, "data/temporal_sequences.pt")
    idx_path = os.path.join(PROJECT_ROOT, "data/window_index.parquet")
    
    seqs = torch.load(seq_path, weights_only=False, map_location='cpu')
    idx_df = pd.read_parquet(idx_path)
    idx_df = idx_df.set_index('window_id')
    
    inputs = seqs['inputs']
    metadata = seqs['metadata']
    
    # Same chronological split as Phase 7
    total = len(inputs)
    train_size = int(0.70 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size
    
    test_inputs = inputs[train_size + val_size:]
    test_meta = metadata[train_size + val_size:]
    
    engine = EarlyWarningEngine(
        checkpoint_path=os.path.join(CHECKPOINT_DIR, "best_world_model.pt"),
        scaler_path=os.path.join(CHECKPOINT_DIR, "state_scaler.pkl")
    )
    
    warnings = []
    y_true_attacks = []
    y_pred_warnings = []
    
    attack_onsets = 0
    onsets_warned = 0
    
    results_list = []
    
    print(f"\n[ 1. RUNNING INFERENCE ON TEST SET ]")
    print(f"  Test samples: {len(test_inputs)}")
    
    for i in range(len(test_inputs)):
        seq = test_inputs[i]
        meta = test_meta[i]
        
        target_w_id = meta['target_window_id']
        # The sequence input ends at t.
        # But what is t's window_id? The sequence length is 5.
        # Let's find t in the idx_df by matching timestamp.
        # Alternatively, target_window_id is t+1, so t is target_window_id - 1 ? No, because of missing windows.
        # Instead, just use the index dataframe with the input_end timestamp.
        
        t_timestamp = meta['input_end']
        
        # current state t
        t_row = idx_df[idx_df['window_end'] == t_timestamp]
        if len(t_row) == 0:
            # Fallback
            is_currently_attack = False
            t_w_id = -1
        else:
            is_currently_attack = bool(t_row.iloc[0]['contains_attack'])
            t_w_id = t_row.index[0]
            
        # Target t+1
        target_row = idx_df.loc[target_w_id]
        actual_t_plus_1_attack = bool(target_row['contains_attack'])
        
        # Scale current state (from input tensor, last window, indices 64:77)
        current_state_unscaled = seq[-1, 64:].unsqueeze(0).cpu().numpy()
        current_state_scaled = torch.tensor(engine.scaler.transform(current_state_unscaled), dtype=torch.float32)
        
        # Run Early Warning Engine
        out = engine.evaluate_sequence(
            sequence_tensor=seq,
            current_state_tensor=current_state_scaled,
            is_currently_attack=is_currently_attack,
            current_window_id=t_w_id
        )
        
        # Record
        is_warning_generated = out['is_attack_onset'] or out['warning_level'] in ["WARNING_ONSET", "WARNING_CONTINUING"]
        
        y_true_attacks.append(actual_t_plus_1_attack)
        y_pred_warnings.append(is_warning_generated)
        
        if actual_t_plus_1_attack and not is_currently_attack:
            attack_onsets += 1
            if out['warning_level'] == "WARNING_ONSET":
                onsets_warned += 1
                
        out['actual_t_plus_1_attack'] = actual_t_plus_1_attack
        out['timestamp'] = str(meta['target_window_start'])
        results_list.append(out)

    print(f"\n[ 2. EVALUATING WARNING LOGIC ]")
    
    y_pred_probs = [r['risk_probability'] for r in results_list]
    
    precision = precision_score(y_true_attacks, y_pred_warnings, zero_division=0)
    recall = recall_score(y_true_attacks, y_pred_warnings, zero_division=0)
    f1 = f1_score(y_true_attacks, y_pred_warnings, zero_division=0)
    pr_auc = average_precision_score(y_true_attacks, y_pred_probs)
    
    # Persistence baseline
    # For predicting t+1, persistence just uses t.
    y_persistence = [r['is_current_attack'] for r in results_list]
    base_precision = precision_score(y_true_attacks, y_persistence, zero_division=0)
    base_recall = recall_score(y_true_attacks, y_persistence, zero_division=0)
    base_f1 = f1_score(y_true_attacks, y_persistence, zero_division=0)
    
    onset_recall = onsets_warned / attack_onsets if attack_onsets > 0 else 0.0
    
    # False warning rate (False Positives / Total Negatives)
    tn, fp, fn, tp = confusion_matrix(y_true_attacks, y_pred_warnings, labels=[False, True]).ravel()
    false_warning_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    total_benign_targets = tn + fp
    total_attack_targets = fn + tp
    total_warnings = fp + tp
    false_warnings = fp
    
    print("\n----------------------------------------------------------------------")
    print("  PHASE 8 EARLY WARNING EVALUATION REPORT")
    print("----------------------------------------------------------------------")
    print("\n[ ATTACK ONSET METRICS ]")
    print(f"  Total Attack Onsets:    {attack_onsets}")
    print(f"  Onsets Warned:          {onsets_warned}")
    print(f"  Onset Recall:           {onset_recall:.4f}")
    
    print("\n[ WARNING CONFUSION MATRIX ]")
    print(f"  TN: {tn} | FP: {fp}")
    print(f"  FN: {fn} | TP: {tp}")
    print(f"  Total Benign Target Windows:  {total_benign_targets}")
    print(f"  Total Attack Target Windows:  {total_attack_targets}")
    print(f"  Total Warnings Issued:        {total_warnings}")
    print(f"  False Warnings:               {false_warnings}")
    
    print("\n[ WARNING METRICS ]")
    print(f"  False Warning Rate:     {false_warning_rate:.4f}")
    print(f"  Precision:              {precision:.4f}")
    print(f"  Recall:                 {recall:.4f}")
    print(f"  F1-Score:               {f1:.4f}")
    print(f"  Continuous PR-AUC:      {pr_auc:.4f}")
    
    print("\n[ PERSISTENCE BASELINE COMPARISON ]")
    print(f"  Persistence Precision:  {base_precision:.4f}")
    print(f"  Persistence Recall:     {base_recall:.4f}")
    print(f"  Persistence F1-Score:   {base_f1:.4f}")
    
    if f1 > base_f1:
        print("  -> Warning engine outperforms persistence.")
    else:
        print("  -> Warning engine does NOT outperform persistence.")
        
    print("\n======================================================================")
    print("  ✓ PHASE 8 EARLY WARNING EVALUATION COMPLETE")
    print("======================================================================\n")
    
    # Save artifacts
    analysis = {
        "onset_recall": onset_recall,
        "false_warning_rate": false_warning_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "attack_onsets_actual": attack_onsets,
        "attack_onsets_warned": onsets_warned,
        "persistence_precision": base_precision,
        "persistence_recall": base_recall,
        "persistence_f1": base_f1
    }
    
    with open(os.path.join(PROJECT_ROOT, "evaluation/phase8_warning_analysis.json"), 'w') as f:
        json.dump(analysis, f, indent=4)
        
    with open(os.path.join(PROJECT_ROOT, "evaluation/phase8_results.json"), 'w') as f:
        json.dump(results_list, f, indent=4)

if __name__ == "__main__":
    main()
