import numpy as np
import time
from config import GLOBAL_STATE_FEATURES

class ExplanationEngine:
    def __init__(self, scaler):
        """
        Initializes the explanation engine.
        scaler: The Phase 7 state scaler (expected to be StandardScaler or similar).
        """
        self.scaler = scaler

    def generate_explanation(self, current_state_unscaled, pred_state_scaled, warning_data):
        """
        Generates a structured explanation based on the forecasted state changes.
        
        Args:
            current_state_unscaled: [13] array of original feature values at t
            pred_state_scaled: [13] array of forecasted feature values at t+1 in scaled space
            warning_data: dictionary output from EarlyWarningEngine
            
        Returns:
            dict: Structured explanation object
        """
        start_time = time.time()
        
        # 1. Prepare states
        # Replace nan/inf to prevent scaler crashing
        safe_curr_unscaled = np.nan_to_num(current_state_unscaled, nan=0.0, posinf=0.0, neginf=0.0)
        safe_pred_scaled = np.nan_to_num(pred_state_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale current state to compare with forecasted state
        current_state_scaled = self.scaler.transform(safe_curr_unscaled.reshape(1, -1)).flatten()
        
        # Unscale forecasted state to show absolute values
        pred_state_unscaled = self.scaler.inverse_transform(safe_pred_scaled.reshape(1, -1)).flatten()
        
        def safe_num(val):
            if np.isnan(val) or np.isinf(val):
                return 0.0
            return float(val)
            
        feature_changes = []
        
        for i, feature_name in enumerate(GLOBAL_STATE_FEATURES):
            curr_val = safe_num(current_state_unscaled[i])
            pred_val = safe_num(pred_state_unscaled[i])
            
            # Unscaled deltas
            signed_delta = pred_val - curr_val
            abs_delta = abs(signed_delta)
            
            # Relative change (safe division)
            if curr_val != 0:
                rel_change = signed_delta / abs(curr_val)
                # JSON safe guard just in case
                if np.isnan(rel_change) or np.isinf(rel_change):
                    rel_change = "inf" if signed_delta > 0 else ("-inf" if signed_delta < 0 else 0.0)
            else:
                rel_change = "inf" if signed_delta > 0 else ("-inf" if signed_delta < 0 else 0.0)
                
            # Scaled delta (meaningful for cross-feature magnitude ranking)
            curr_scaled = float(current_state_scaled[i])
            pred_scaled = float(pred_state_scaled[i])
            
            # Catch NaNs from scaled state just in case input was corrupted
            if np.isnan(curr_scaled) or np.isnan(pred_scaled):
                signed_scaled_delta = 0.0
                abs_scaled_delta = 0.0
            else:
                signed_scaled_delta = pred_scaled - curr_scaled
                abs_scaled_delta = abs(signed_scaled_delta)
            
            feature_changes.append({
                "feature": feature_name,
                "current_value": curr_val,
                "forecasted_value": pred_val,
                "signed_change": signed_delta,
                "absolute_change": abs_delta,
                "relative_change": rel_change,
                "signed_scaled_change": signed_scaled_delta,
                "absolute_scaled_change": abs_scaled_delta
            })
            
        # 2. Rank features by magnitude of forecasted change (in scaled space)
        # We sort descending by absolute scaled change
        feature_changes.sort(key=lambda x: x["absolute_scaled_change"], reverse=True)
        
        for rank, feat in enumerate(feature_changes):
            feat["rank"] = rank + 1
            
        top_changed = feature_changes[:3]
        
        # 3. Formulate Summary
        stage = warning_data.get("inferred_stage", "UNKNOWN")
        w_level = warning_data.get("warning_level", "UNKNOWN")
        risk = warning_data.get("risk_probability", 0.0)
        
        if w_level == "WARNING_ONSET":
            summary = "The model forecasts an attack onset. "
        elif w_level == "WARNING_CONTINUING":
            summary = "The model forecasts an ongoing attack. "
        elif w_level == "WATCH_RECOVERY":
            summary = "The model forecasts recovery from an ongoing attack. "
        else:
            summary = "The network is forecasted to remain normal. "
            
        summary += f"The feature showing strongest deviation is '{top_changed[0]['feature']}'."
        
        explanation_time = time.time() - start_time
        
        return {
            "summary": summary,
            "risk_probability": risk,
            "predicted_attack_type": warning_data.get("predicted_attack_type", "Unknown"),
            "warning_level": w_level,
            "top_changed_features": top_changed,
            "all_feature_changes": feature_changes,
            "current_attack": warning_data.get("is_current_attack", False),
            "attack_onset": warning_data.get("is_attack_onset", False),
            "early_warning": warning_data.get("is_early_warning", False),
            "stage": stage,
            "mitre": warning_data.get("mitre_enrichment", None),
            "generation_time_sec": explanation_time
        }
