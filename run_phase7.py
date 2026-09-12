"""
SADMaNS — Phase 7 Entry Point
==========================================

Runs the Forecasting + Attack Prediction Heads + Evaluation.
Includes Multi-Task Training, Chronological Split, Baselines, and Thresholding.
"""

import os
import sys
import time
import json
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, 
    precision_score, recall_score, f1_score, 
    confusion_matrix, average_precision_score, roc_auc_score
)

from config import (
    PROJECT_ROOT, DEVICE, LEARNING_RATE, BATCH_SIZE, EPOCHS, WEIGHT_DECAY,
    STATE_LOSS_WEIGHT, RISK_LOSS_WEIGHT, TYPE_LOSS_WEIGHT, ATTACK_TYPE_TO_ID
)
from models.full_world_model import FullWorldModel


def _sep(char: str = "=", width: int = 70) -> str:
    return char * width

def get_class_weights(y_train):
    """Calculates class weights to handle imbalance in training data."""
    counts = np.bincount(y_train.numpy(), minlength=3)
    # Prevent division by zero and NaN cross-entropy loss by providing a fallback weight
    weights = [1.0 if c == 0 else len(y_train) / (3.0 * c) for c in counts]
    return torch.tensor(weights, dtype=torch.float32)

def main() -> int:
    print(_sep())
    print("  SADMaNS — PHASE 7: FORECASTING + ATTACK PREDICTION")
    print(_sep())
    print()

    data_dir = os.path.join(PROJECT_ROOT, "data")
    checkpoints_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    eval_dir = os.path.join(PROJECT_ROOT, "evaluation")
    
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    
    seq_path = os.path.join(data_dir, "temporal_sequences.pt")
    window_index_path = os.path.join(data_dir, "window_index.parquet")
    
    if not os.path.isfile(seq_path) or not os.path.isfile(window_index_path):
        print("[ERROR] Required inputs not found. Run Phase 6 first.")
        return 1
        
    print(f"Loading temporal sequences: {seq_path}")
    sequence_dataset = torch.load(seq_path, weights_only=False)
    
    print(f"Loading window index:       {window_index_path}")
    window_index = pd.read_parquet(window_index_path)
    
    # 1. EXTRACT TARGET LABELS (t+1)
    metadata = sequence_dataset['metadata']
    inputs = sequence_dataset['inputs']
    targets = sequence_dataset['targets']
    
    num_sequences = len(metadata)
    
    risk_labels = []
    type_labels = []
    prev_risk_labels = [] # For baseline persistence attack(t)
    
    for meta in metadata:
        target_w_id = meta['target_window_id']
        current_w_id = meta['input_end'] # Using timestamp for joining
        
        # Target window label
        w_row = window_index[window_index['window_id'] == target_w_id]
        if len(w_row) == 0:
            print(f"[ERROR] Target window {target_w_id} not found in index.")
            return 1
            
        w_row = w_row.iloc[0]
        
        # Risk (Binary)
        is_attack = bool(w_row['contains_attack'])
        risk_labels.append(1.0 if is_attack else 0.0)
        
        # Type
        if not is_attack:
            type_id = ATTACK_TYPE_TO_ID["Benign"]
        elif w_row['has_goldeneye']:
            type_id = ATTACK_TYPE_TO_ID["DoS attacks-GoldenEye"]
        elif w_row['has_slowloris']:
            type_id = ATTACK_TYPE_TO_ID["DoS attacks-Slowloris"]
        else:
            type_id = ATTACK_TYPE_TO_ID["Benign"]
            
        type_labels.append(type_id)
        
        # Current window label for baseline
        curr_row = window_index[window_index['window_start'] == current_w_id]
        if len(curr_row) > 0:
            prev_risk_labels.append(1.0 if bool(curr_row.iloc[0]['contains_attack']) else 0.0)
        else:
            prev_risk_labels.append(0.0)
        
    risk_tensor = torch.tensor(risk_labels, dtype=torch.float32)
    type_tensor = torch.tensor(type_labels, dtype=torch.long)
    prev_risk_tensor = torch.tensor(prev_risk_labels, dtype=torch.float32)
    
    # 2. CHRONOLOGICAL SPLIT (70% Train, 15% Val, 15% Test)
    train_size = int(0.7 * num_sequences)
    val_size = int(0.15 * num_sequences)
    test_size = num_sequences - train_size - val_size
    
    train_idx = slice(0, train_size)
    val_idx = slice(train_size, train_size + val_size)
    test_idx = slice(train_size + val_size, num_sequences)
    
    print(f"\n[ 1. DATASET SPLITTING ]")
    print(f"  Chronological split applied.")
    print(f"  Train samples:      {train_size}")
    print(f"  Validation samples: {val_size}")
    print(f"  Test samples:       {test_size}")
    
    # Extract splits
    X_train, y_state_train = inputs[train_idx], targets[train_idx]
    y_risk_train, y_type_train = risk_tensor[train_idx], type_tensor[train_idx]
    
    X_val, y_state_val = inputs[val_idx], targets[val_idx]
    y_risk_val, y_type_val = risk_tensor[val_idx], type_tensor[val_idx]
    
    X_test, y_state_test = inputs[test_idx], targets[test_idx]
    y_risk_test, y_type_test = risk_tensor[test_idx], type_tensor[test_idx]
    
    y_risk_prev_test = prev_risk_tensor[test_idx] # Baseline
    
    # 3. SCALING & CLASS WEIGHTS (Fit on Train only)
    print(f"\n[ 2. SCALING & CLASS WEIGHTS ]")
    scaler = StandardScaler()
    scaler.fit(y_state_train.numpy())
    
    y_state_train_scaled = torch.tensor(scaler.transform(y_state_train.numpy()), dtype=torch.float32)
    y_state_val_scaled = torch.tensor(scaler.transform(y_state_val.numpy()), dtype=torch.float32)
    y_state_test_scaled = torch.tensor(scaler.transform(y_state_test.numpy()), dtype=torch.float32)
    
    scaler_path = os.path.join(checkpoints_dir, "state_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
        
    # Attack Type Weights
    type_weights = get_class_weights(y_type_train).to(DEVICE)
    print(f"  Attack Type Weights calculated: {type_weights.cpu().numpy()}")
    
    # Risk Weight (pos_weight)
    num_pos = y_risk_train.sum()
    num_neg = len(y_risk_train) - num_pos
    pos_weight = torch.tensor([num_neg / max(num_pos, 1.0)], dtype=torch.float32).to(DEVICE)
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_state_train_scaled, y_risk_train, y_type_train)
    val_dataset = TensorDataset(X_val, y_state_val_scaled, y_risk_val, y_type_val)
    test_dataset = TensorDataset(X_test, y_state_test_scaled, y_risk_test, y_type_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 4. TRAINING SETUP
    print(f"\n[ 3. MODEL TRAINING ]")
    model = FullWorldModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    criterion_state = nn.MSELoss()
    criterion_risk = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_type = nn.CrossEntropyLoss(weight=type_weights)
    
    best_val_loss = float('inf')
    import copy
    best_model_state = copy.deepcopy(model.state_dict())
    training_history = []
    
    print(f"  Training for {EPOCHS} epochs...")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        tr_s_loss = 0.0
        tr_r_loss = 0.0
        tr_t_loss = 0.0
        
        for x_b, ys_b, yr_b, yt_b in train_loader:
            x_b, ys_b, yr_b, yt_b = x_b.to(DEVICE), ys_b.to(DEVICE), yr_b.to(DEVICE), yt_b.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(x_b)
            
            loss_state = criterion_state(outputs["next_state"], ys_b)
            loss_risk = criterion_risk(outputs["risk_logits"], yr_b)
            loss_type = criterion_type(outputs["type_logits"], yt_b)
            
            loss = (STATE_LOSS_WEIGHT * loss_state) + (RISK_LOSS_WEIGHT * loss_risk) + (TYPE_LOSS_WEIGHT * loss_type)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * x_b.size(0)
            tr_s_loss += loss_state.item() * x_b.size(0)
            tr_r_loss += loss_risk.item() * x_b.size(0)
            tr_t_loss += loss_type.item() * x_b.size(0)
            
        train_loss /= train_size
        tr_s_loss /= train_size
        tr_r_loss /= train_size
        tr_t_loss /= train_size
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_s_loss = 0.0
        val_r_loss = 0.0
        val_t_loss = 0.0
        
        all_val_risk_probs = []
        all_val_risk_trues = []
        
        with torch.no_grad():
            for x_b, ys_b, yr_b, yt_b in val_loader:
                x_b, ys_b, yr_b, yt_b = x_b.to(DEVICE), ys_b.to(DEVICE), yr_b.to(DEVICE), yt_b.to(DEVICE)
                outputs = model(x_b)
                
                loss_state = criterion_state(outputs["next_state"], ys_b)
                loss_risk = criterion_risk(outputs["risk_logits"], yr_b)
                loss_type = criterion_type(outputs["type_logits"], yt_b)
                
                loss = (STATE_LOSS_WEIGHT * loss_state) + (RISK_LOSS_WEIGHT * loss_risk) + (TYPE_LOSS_WEIGHT * loss_type)
                val_loss += loss.item() * x_b.size(0)
                
                val_s_loss += loss_state.item() * x_b.size(0)
                val_r_loss += loss_risk.item() * x_b.size(0)
                val_t_loss += loss_type.item() * x_b.size(0)
                
                all_val_risk_probs.extend(torch.sigmoid(outputs["risk_logits"]).cpu().numpy())
                all_val_risk_trues.extend(yr_b.cpu().numpy())
                
        val_loss /= val_size
        val_s_loss /= val_size
        val_r_loss /= val_size
        val_t_loss /= val_size
        
        training_history.append({
            "epoch": epoch + 1,
            "train_total_loss": train_loss,
            "train_state_loss": tr_s_loss,
            "train_risk_loss": tr_r_loss,
            "train_type_loss": tr_t_loss,
            "val_total_loss": val_loss,
            "val_state_loss": val_s_loss,
            "val_risk_loss": val_r_loss,
            "val_type_loss": val_t_loss
        })
        
        print(f"  Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        import math
        if math.isnan(val_loss):
            print("  [WARNING] Validation Loss is NaN!")
            
        import copy
        if val_loss < best_val_loss and not math.isnan(val_loss):
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            
    print(f"  Training completed in {time.time() - start_time:.2f}s")
    
    with open(os.path.join(eval_dir, "training_history.json"), "w") as f:
        json.dump(training_history, f, indent=4)
        
    # THRESHOLD SELECTION on VAL
    best_threshold = 0.5
    best_f1 = -1
    for th in np.arange(0.1, 1.0, 0.1):
        preds = (np.array(all_val_risk_probs) >= th).astype(int)
        f1 = f1_score(all_val_risk_trues, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = th
    print(f"  Selected Risk Threshold from Val: {best_threshold:.2f}")

    # Save Best Model
    model.load_state_dict(best_model_state)
    checkpoint_path = os.path.join(checkpoints_dir, "best_world_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'GRU_INPUT_DIM': 77,
            'GRU_HIDDEN_DIM': 128,
            'GLOBAL_STATE_DIM': 13,
            'STATE_LOSS_WEIGHT': STATE_LOSS_WEIGHT,
            'RISK_LOSS_WEIGHT': RISK_LOSS_WEIGHT,
            'TYPE_LOSS_WEIGHT': TYPE_LOSS_WEIGHT
        },
        'threshold': best_threshold,
        'scaler_path': scaler_path
    }, checkpoint_path)
    print(f"  Saved best model checkpoint to {checkpoint_path}")
    
    # 5. EVALUATION
    print(f"\n[ 4. EVALUATION (TEST SET) ]")
    model.eval()
    
    all_pred_states_scaled = []
    all_true_states_scaled = []
    
    all_risk_probs = []
    all_true_risks = []
    
    all_type_preds = []
    all_true_types = []
    
    with torch.no_grad():
        for x_b, ys_b, yr_b, yt_b in test_loader:
            x_b = x_b.to(DEVICE)
            
            outputs = model(x_b)
            
            all_pred_states_scaled.append(outputs["next_state"].cpu().numpy())
            all_true_states_scaled.append(ys_b.numpy())
            
            risk_probs = torch.sigmoid(outputs["risk_logits"]).cpu().numpy()
            all_risk_probs.extend(risk_probs)
            all_true_risks.extend(yr_b.numpy())
            
            type_probs = torch.softmax(outputs["type_logits"], dim=-1).cpu().numpy()
            type_preds = np.argmax(type_probs, axis=-1)
            all_type_preds.extend(type_preds)
            all_true_types.extend(yt_b.numpy())
            
    # Regression Metrics
    preds_scaled = np.vstack(all_pred_states_scaled)
    trues_scaled = np.vstack(all_true_states_scaled)
    
    preds_orig = scaler.inverse_transform(preds_scaled)
    trues_orig = scaler.inverse_transform(trues_scaled)
    
    rmse_scaled = np.sqrt(mean_squared_error(trues_scaled, preds_scaled))
    mae_scaled = mean_absolute_error(trues_scaled, preds_scaled)
    
    rmse_orig = np.sqrt(mean_squared_error(trues_orig, preds_orig))
    mae_orig = mean_absolute_error(trues_orig, preds_orig)
    
    # Baseline Persistence Forecasting (S_t+1 = S_t)
    persistence_preds_orig = X_test[:, -1, -13:].numpy()
    base_rmse_orig = np.sqrt(mean_squared_error(trues_orig, persistence_preds_orig))
    base_mae_orig = mean_absolute_error(trues_orig, persistence_preds_orig)
    
    # Risk Metrics
    num_attacks_test = int(sum(all_true_risks))
    
    if num_attacks_test == 0:
        pr_auc = 0.0
        roc_auc = 0.0
        print("  [WARNING] No attacks in test set for ROC-AUC/PR-AUC.")
    else:
        pr_auc = average_precision_score(all_true_risks, all_risk_probs)
        roc_auc = roc_auc_score(all_true_risks, all_risk_probs)
        
    risk_preds_bin = (np.array(all_risk_probs) >= best_threshold).astype(int)
    risk_prec = precision_score(all_true_risks, risk_preds_bin, zero_division=0)
    risk_rec = recall_score(all_true_risks, risk_preds_bin, zero_division=0)
    risk_f1 = f1_score(all_true_risks, risk_preds_bin, zero_division=0)
    risk_cm = confusion_matrix(all_true_risks, risk_preds_bin).tolist()
    
    # Risk Baseline Persistence (attack_t+1 = attack_t)
    base_risk_preds = y_risk_prev_test.numpy()
    base_risk_prec = precision_score(all_true_risks, base_risk_preds, zero_division=0)
    base_risk_rec = recall_score(all_true_risks, base_risk_preds, zero_division=0)
    base_risk_f1 = f1_score(all_true_risks, base_risk_preds, zero_division=0)
    
    # Type Metrics
    type_f1_macro = f1_score(all_true_types, all_type_preds, average='macro', zero_division=0)
    type_f1_weighted = f1_score(all_true_types, all_type_preds, average='weighted', zero_division=0)
    type_cm = confusion_matrix(all_true_types, all_type_preds).tolist()
    
    # Type Baseline Majority
    # Find majority in train
    maj_class = int(torch.mode(y_type_train).values.item())
    base_type_preds = np.full_like(all_true_types, fill_value=maj_class)
    base_type_f1_macro = f1_score(all_true_types, base_type_preds, average='macro', zero_division=0)
    
    # Compile Results
    results = {
        "dataset": {
            "total_sequences": num_sequences
        },
        "split": {
            "train": train_size,
            "validation": val_size,
            "test": test_size
        },
        "forecasting": {
            "rmse_scaled": float(rmse_scaled),
            "mae_scaled": float(mae_scaled),
            "rmse_original": float(rmse_orig),
            "mae_original": float(mae_orig)
        },
        "forecasting_baseline": {
            "rmse_original": float(base_rmse_orig),
            "mae_original": float(base_mae_orig)
        },
        "attack_risk": {
            "pr_auc": float(pr_auc),
            "roc_auc": float(roc_auc),
            "precision": float(risk_prec),
            "recall": float(risk_rec),
            "f1_score": float(risk_f1),
            "confusion_matrix": risk_cm
        },
        "attack_risk_baseline": {
            "precision": float(base_risk_prec),
            "recall": float(base_risk_rec),
            "f1_score": float(base_risk_f1)
        },
        "attack_type": {
            "macro_f1": float(type_f1_macro),
            "weighted_f1": float(type_f1_weighted),
            "confusion_matrix": type_cm
        },
        "attack_type_baseline": {
            "macro_f1": float(base_type_f1_macro)
        },
        "support": {
            "test_total": test_size,
            "test_attacks": num_attacks_test
        }
    }
    
    with open(os.path.join(eval_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n" + _sep("-"))
    print("  PHASE 7 VALIDATION REPORT")
    print(_sep("-"))
    
    print("\n[ FORECASTING METRICS ]")
    print(f"  Model RMSE (Original):  {rmse_orig:.4f}")
    print(f"  Baseline RMSE (Orig):   {base_rmse_orig:.4f}")
    
    print("\n[ ATTACK RISK METRICS ]")
    print(f"  PR-AUC:                 {pr_auc:.4f}")
    print(f"  ROC-AUC:                {roc_auc:.4f}")
    print(f"  Precision:              {risk_prec:.4f}")
    print(f"  Recall:                 {risk_rec:.4f}")
    print(f"  F1-Score:               {risk_f1:.4f}")
    print(f"  Baseline F1-Score:      {base_risk_f1:.4f}")
    
    print("\n[ ATTACK TYPE METRICS ]")
    print(f"  Macro F1:               {type_f1_macro:.4f}")
    print(f"  Baseline Macro F1:      {base_type_f1_macro:.4f}")
    
    print("\n[ TEST DATA DISTRIBUTION ]")
    print(f"  Total Windows:          {test_size}")
    print(f"  Attack Windows:         {num_attacks_test}")
    
    print("\n" + _sep())
    print("  ✓ PHASE 7 FORECASTING + ATTACK PREDICTION COMPLETE")
    print(_sep())

    return 0

if __name__ == "__main__":
    sys.exit(main())
