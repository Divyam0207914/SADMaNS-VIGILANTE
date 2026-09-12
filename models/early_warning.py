import torch
import numpy as np
import joblib
from models.full_world_model import FullWorldModel
from config import DEVICE, GLOBAL_STATE_FEATURES

class EarlyWarningEngine:
    def __init__(self, checkpoint_path, scaler_path):
        """
        Initializes the early warning engine from the Phase 7 world model.
        """
        self.device = DEVICE
        self.model = FullWorldModel().to(self.device)
        chk = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(chk['model_state_dict'])
        self.model.eval()
        self.scaler = joblib.load(scaler_path)
        self.risk_threshold = chk.get('threshold', 0.10)
        
    def evaluate_sequence(self, sequence_tensor: torch.Tensor, current_state_tensor: torch.Tensor, is_currently_attack: bool, current_window_id: int):
        """
        Evaluate a sequence in real-time.
        
        sequence_tensor: [1, 5, 77]
        current_state_tensor: [1, 13] (scaled or original, we use it just for delta if scaled)
        is_currently_attack: bool, is time t an attack?
        current_window_id: int
        """
        # Ensure dimensions
        if sequence_tensor.dim() == 2:
            sequence_tensor = sequence_tensor.unsqueeze(0)
            
        sequence_tensor = sequence_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.inference(sequence_tensor, risk_threshold=self.risk_threshold)
            out_dict = outputs[0]
            
        pred_state_scaled = out_dict['next_state']
        current_state_scaled = current_state_tensor[0].cpu().numpy() if current_state_tensor.dim() == 2 else current_state_tensor.cpu().numpy()
        
        # Calculate deviation in scaled space
        delta_scaled = pred_state_scaled - current_state_scaled
        
        # Explainability: which features changed the most in scaled space?
        abs_delta = np.abs(delta_scaled)
        top_indices = np.argsort(abs_delta)[::-1][:3]
        
        top_changed_features = [
            {"feature": GLOBAL_STATE_FEATURES[i], "change": float(delta_scaled[i])}
            for i in top_indices
        ]
        
        is_attack_forecast = out_dict['is_attack']
        
        if is_attack_forecast and not is_currently_attack:
            warning_level = "WARNING_ONSET"
            is_onset = True
        elif is_attack_forecast and is_currently_attack:
            warning_level = "WARNING_CONTINUING"
            is_onset = False
        elif not is_attack_forecast and is_currently_attack:
            warning_level = "WATCH_RECOVERY"
            is_onset = False
        else:
            warning_level = "NORMAL"
            is_onset = False
            
        return {
            "window_id": int(current_window_id),
            "risk_probability": float(out_dict['risk_probability']),
            "warning_level": warning_level,
            "predicted_attack_type": out_dict['attack_type_str'],
            "predicted_state_delta": delta_scaled.tolist(),
            "top_changed_features": top_changed_features,
            "is_current_attack": bool(is_currently_attack),
            "is_attack_onset": bool(is_onset),
            "is_early_warning": bool(is_onset), 
            "inferred_stage": out_dict['inferred_stage'],
            "mitre_enrichment": out_dict['mitre_enrichment']
        }
