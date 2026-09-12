"""
SADMaNS — Final World Model
==========================================

Phase 7: The complete world model combining:
- Next-State Forecasting (from Phase 6 GRU)
- Attack Risk Prediction
- Attack Type Prediction
- Stage Inference (Heuristic)
- MITRE ATT&CK Enrichment (Semantic)
"""

import torch
import torch.nn as nn
from typing import Dict, Any

from models.gru_world_model import GRUWorldModel
from models.attack_heads import AttackRiskHead, AttackTypeHead
from config import ID_TO_ATTACK_TYPE

class FullWorldModel(nn.Module):
    def __init__(self, input_dim: int = 77, hidden_dim: int = 128, output_dim: int = 13, num_attack_classes: int = 3):
        super(FullWorldModel, self).__init__()
        
        # 1. Base GRU Model (Predicts Next State)
        self.world_model = GRUWorldModel(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            num_layers=1, 
            output_dim=output_dim
        )
        
        # 2. Attack Risk Head (Predicts Binary Risk Logits)
        self.risk_head = AttackRiskHead(input_dim=hidden_dim)
        
        # 3. Attack Type Head (Predicts 3-Class Logits)
        self.type_head = AttackTypeHead(input_dim=hidden_dim, num_classes=num_attack_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multi-task training/evaluation.
        
        Args:
            x (Tensor): Input sequence [B, SEQ_LEN, 77]
            
        Returns:
            Dict containing:
                - next_state (Tensor [B, 13]): Predicted state S_(t+1)
                - risk_logits (Tensor [B]): Binary attack risk logits
                - type_logits (Tensor [B, 3]): Multi-class attack type logits
        """
        # Get next state prediction and the 128-d temporal representation
        next_state, final_hidden = self.world_model(x, return_hidden=True)
        
        # Pass 128-d representation to attack heads
        risk_logits = self.risk_head(final_hidden)
        type_logits = self.type_head(final_hidden)
        
        return {
            "next_state": next_state,
            "risk_logits": risk_logits,
            "type_logits": type_logits
        }

    @torch.no_grad()
    def inference(self, x: torch.Tensor, risk_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Clean inference pipeline for a single batch.
        Provides model outputs, post-processing, heuristic stages, and MITRE enrichment.
        """
        self.eval()
        outputs = self.forward(x)
        
        # 1. Model Output processing
        next_state = outputs["next_state"]
        
        # Sigmoid for risk probability
        risk_probs = torch.sigmoid(outputs["risk_logits"])
        
        # Softmax for type probabilities
        type_probs = torch.softmax(outputs["type_logits"], dim=-1)
        type_preds = torch.argmax(type_probs, dim=-1)
        
        batch_size = x.size(0)
        results = []
        
        for i in range(batch_size):
            # 2. Post-Processing
            predicted_type_id = type_preds[i].item()
            predicted_type_str = ID_TO_ATTACK_TYPE.get(predicted_type_id, "Unknown")
            risk_prob = risk_probs[i].item()
            
            # Apply threshold for binary risk
            is_attack = risk_prob >= risk_threshold
            
            # Reconcile risk and type if needed, but here we just report both
            
            # 3. Stage Inference (Heuristic)
            inferred_stage = "IMPACT" if is_attack else "NORMAL"
            
            # 4. MITRE ATT&CK Enrichment (Semantic)
            # DoS attacks (GoldenEye/Slowloris) map to Endpoint Denial of Service
            mitre_enrichment = None
            if is_attack:
                mitre_enrichment = {
                    "tactic": "TA0040 (Impact)",
                    "technique": "T1499 (Endpoint Denial of Service)",
                    "note": "Semantic mapping based on heuristic DoS identification. Not a learned attribution."
                }
                
            results.append({
                "next_state": next_state[i].cpu().numpy(),
                "risk_probability": risk_prob,
                "is_attack": is_attack,
                "attack_type_id": predicted_type_id,
                "attack_type_str": predicted_type_str,
                "inferred_stage": inferred_stage,
                "mitre_enrichment": mitre_enrichment
            })
            
        return results
