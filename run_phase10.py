import os
import json
import time
import numpy as np
import pandas as pd
import torch

from config import PROJECT_ROOT, CHECKPOINT_DIR
from models.early_warning import EarlyWarningEngine
from models.explanation_engine import ExplanationEngine


def run_synthetic_tests(explainer, engine):
    print("\n[6/8] Running Synthetic Edge-Case Software Behavior Tests...")
    
    val_status = "PASS"
    
    synth_current = np.zeros(13)
    synth_pred = np.ones(13)
    synth_warn = {"inferred_stage": "NORMAL", "warning_level": "NORMAL", "risk_probability": 0.05, "predicted_attack_type": "Benign"}
    
    # 1. Normal/Benign
    try:
        exp_benign = explainer.generate_explanation(synth_current, synth_pred, synth_warn)
    except Exception as e:
        print(f"  ✗ Synthetic NORMAL test failed: {e}")
        val_status = "FAIL"
        
    # 2. NaN Input
    synth_nan = np.full(13, np.nan)
    try:
        exp_nan = explainer.generate_explanation(synth_nan, synth_pred, synth_warn)
    except Exception as e:
        print(f"  ✗ Synthetic NaN test failed: {e}")
        val_status = "FAIL"
        
    # 3. Inf Input
    synth_inf = np.full(13, np.inf)
    try:
        exp_inf = explainer.generate_explanation(synth_inf, synth_pred, synth_warn)
    except Exception as e:
        print(f"  ✗ Synthetic Inf test failed: {e}")
        val_status = "FAIL"
        
    # 4. Warning States
    warning_states = ["NORMAL", "WARNING_ONSET", "WARNING_CONTINUING", "WATCH_RECOVERY"]
    for ws in warning_states:
        w_data = {"inferred_stage": "IMPACT", "warning_level": ws, "risk_probability": 0.9, "predicted_attack_type": "Slowloris"}
        try:
            exp_ws = explainer.generate_explanation(synth_current, synth_pred, w_data)
        except Exception as e:
            print(f"  ✗ Synthetic Warning State ({ws}) failed: {e}")
            val_status = "FAIL"
            
    if val_status == "PASS":
        print("  ✓ All synthetic control-flow tests passed.")
    return val_status == "PASS"


def main():
    print("=" * 75)
    print(" SADMaNS / VIGILANTE — PHASE 10: END-TO-END INTEGRATION & SIH VALIDATION")
    print("=" * 75)

    start = time.time()

    # ---------------------------------------------------------
    # PATHS
    # ---------------------------------------------------------
    seq_path = os.path.join(PROJECT_ROOT, "data/temporal_sequences.pt")
    idx_path = os.path.join(PROJECT_ROOT, "data/window_index.parquet")
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_world_model.pt")
    scaler_path = os.path.join(CHECKPOINT_DIR, "state_scaler.pkl")

    # ---------------------------------------------------------
    # LOAD
    # ---------------------------------------------------------
    print("\n[1/8] Loading Models and Dependencies...")
    
    t_load_start = time.time()
    engine = EarlyWarningEngine(checkpoint_path=checkpoint_path, scaler_path=scaler_path)
    explainer = ExplanationEngine(scaler=engine.scaler)

    seqs = torch.load(seq_path, weights_only=False, map_location="cpu")
    idx_df = pd.read_parquet(idx_path).set_index("window_id")

    inputs = seqs["inputs"]
    metadata = seqs["metadata"]

    load_time = time.time() - t_load_start
    print(f"  ✓ Dependencies loaded in {load_time:.3f} sec.")

    # ---------------------------------------------------------
    # TEST SPLIT
    # ---------------------------------------------------------
    total = len(inputs)
    train_size = int(0.70 * total)
    val_size = int(0.15 * total)
    test_start = train_size + val_size

    test_inputs = inputs[test_start:]
    test_metadata = metadata[test_start:]

    print(f"\n[2/8] Historical Replay Configuration")
    print(f"  Total historical sequences available: {total}")
    print(f"  Testing split size: {len(test_inputs)}")

    # ---------------------------------------------------------
    # RUN INTEGRATION
    # ---------------------------------------------------------
    print("\n[3/8] Running End-to-End Integration (Phase 7 → 8 → 9)...")

    results = []
    valid_count = 0
    invalid_count = 0

    warning_counts = {
        "NORMAL": 0,
        "WARNING_ONSET": 0,
        "WARNING_CONTINUING": 0,
        "WATCH_RECOVERY": 0,
        "UNKNOWN": 0
    }

    t_inf_total = 0.0
    t_exp_total = 0.0

    for i in range(len(test_inputs)):
        seq = test_inputs[i]
        meta = test_metadata[i]

        t_timestamp = meta["input_end"]
        t_row = idx_df[idx_df["window_end"] == t_timestamp]

        if len(t_row) == 0:
            invalid_count += 1
            continue

        is_currently_attack = bool(t_row.iloc[0]["contains_attack"])
        current_window_id = t_row.index[0]

        current_state_unscaled = seq[-1, 64:].cpu().numpy().astype(np.float64)

        current_state_scaled = torch.tensor(
            engine.scaler.transform(current_state_unscaled.reshape(1, -1)),
            dtype=torch.float32
        )

        # Phase 8 inference
        t_i = time.time()
        warning_data = engine.evaluate_sequence(
            sequence_tensor=seq,
            current_state_tensor=current_state_scaled,
            is_currently_attack=is_currently_attack,
            current_window_id=current_window_id
        )
        t_inf_total += time.time() - t_i

        warning_level = warning_data.get("warning_level", "UNKNOWN")
        warning_counts[warning_level] = warning_counts.get(warning_level, 0) + 1

        predicted_state_delta = np.asarray(warning_data["predicted_state_delta"], dtype=np.float64)
        predicted_state_scaled = current_state_scaled.cpu().numpy().flatten() + predicted_state_delta

        # Phase 9 explanation
        t_e = time.time()
        explanation = explainer.generate_explanation(
            current_state_unscaled=current_state_unscaled,
            pred_state_scaled=predicted_state_scaled,
            warning_data=warning_data
        )
        t_exp_total += time.time() - t_e

        # Structured final output object
        valid = True
        
        # Check numerical safety
        risk = explanation.get("risk_probability", None)
        if risk is None or not np.isfinite(float(risk)): valid = False
        
        for feature in explanation.get("all_feature_changes", []):
            for key in ["current_value", "forecasted_value", "absolute_scaled_change"]:
                value = feature.get(key)
                if not np.isfinite(float(value)): valid = False

        if valid:
            valid_count += 1
        else:
            invalid_count += 1

        results.append({
            "timestamp": str(t_timestamp),
            "current_window_id": int(current_window_id),
            "current_attack": bool(explanation["current_attack"]),
            "risk_probability": float(risk),
            "predicted_attack_type": explanation["predicted_attack_type"],
            "warning_level": warning_level,
            "is_attack_forecast": bool(warning_data.get("is_attack_forecast", False)),
            "is_early_warning": bool(warning_data.get("is_early_warning", False)),
            "stage": explanation["stage"],
            "mitre": explanation.get("mitre"),
            "top_changed_features": [f["feature"] for f in explanation["top_changed_features"]],
            "explanation_summary": explanation.get("summary"),
            "valid": valid
        })

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------
    print("\n[4/8] Integration Validation Results")
    print(f"  Valid structured outputs:   {valid_count}")
    print(f"  Invalid outputs (skipped/nan): {invalid_count}")
    
    print("\n[5/8] Final Leakage Audit")
    leakage_pass = True
    for meta in test_metadata:
        if not (meta["target_window_start"] > meta["input_end"]):
            leakage_pass = False
            break
            
    if leakage_pass:
        print("  ✓ Chronological strictness preserved. Actual t+1 remains excluded from inference.")
    else:
        print("  ✗ Temporal leakage detected.")

    # Synthetic Tests
    synth_pass = run_synthetic_tests(explainer, engine)
    
    # ---------------------------------------------------------
    # PERFORMANCE
    # ---------------------------------------------------------
    print("\n[7/8] Performance Measurements")
    avg_inf = (t_inf_total / valid_count) * 1000 if valid_count > 0 else 0
    avg_exp = (t_exp_total / valid_count) * 1000 if valid_count > 0 else 0
    print(f"  Model Load Time:          {load_time:.3f} sec")
    print(f"  Avg Single Inference:     {avg_inf:.2f} ms")
    print(f"  Avg Single Explanation:   {avg_exp:.2f} ms")

    # ---------------------------------------------------------
    # FINAL STATUS
    # ---------------------------------------------------------
    total_time = time.time() - start
    final_status = "PASS" if (invalid_count == 0 and leakage_pass and synth_pass) else "FAIL"

    print("\n[8/8] FINAL PHASE 10 SUMMARY")
    print(f"  Warning Distribution: {warning_counts}")
    print(f"  Overall Validation Status: {final_status}")
    print(f"  Total Runtime: {total_time:.3f} sec")

    # Save
    os.makedirs(os.path.join(PROJECT_ROOT, "evaluation"), exist_ok=True)
    out_path = os.path.join(PROJECT_ROOT, "evaluation/phase10_results.json")
    
    output = {
        "phase": 10,
        "status": final_status,
        "load_time_sec": load_time,
        "avg_inference_time_ms": avg_inf,
        "avg_explanation_time_ms": avg_exp,
        "total_test_sequences": len(test_inputs),
        "valid_outputs": valid_count,
        "warning_distribution": warning_counts,
        "leakage_audit_pass": leakage_pass,
        "synthetic_tests_pass": synth_pass,
        "sample_outputs": results[:5]
    }
    
    with open(out_path, "w") as f:
        json.dump(output, f, indent=4, default=str)
        
    print(f"\nSaved JSON Artifact: {out_path}")


if __name__ == "__main__":
    main()
