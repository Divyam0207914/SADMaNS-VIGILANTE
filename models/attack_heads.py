"""
SADMaNS — Attack Prediction Heads
==========================================

Phase 7: Implementation of the Attack Risk and Attack Type heads.
These operate on the 128-dimensional GRU temporal representation.
"""

import torch
import torch.nn as nn

class AttackRiskHead(nn.Module):
    """
    Predicts binary attack risk for the target window.
    Architecture: 128 -> 64 -> 1
    Outputs logits (no sigmoid).
    """
    def __init__(self, input_dim: int = 128):
        super(AttackRiskHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input: [B, 128] from GRU hidden
        return self.net(x).squeeze(-1)  # Output: [B]

class AttackTypeHead(nn.Module):
    """
    Predicts the attack type for the target window.
    Classes: 0 (Benign), 1 (GoldenEye), 2 (Slowloris)
    Architecture: 128 -> 64 -> 3
    Outputs logits.
    """
    def __init__(self, input_dim: int = 128, num_classes: int = 3):
        super(AttackTypeHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input: [B, 128]
        return self.net(x)  # Output: [B, num_classes]
