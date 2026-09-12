"""
SADMaNS — GRU World Model
==========================================

Phase 6: Implementation of the GRU world model.
Predicts the next global state based on a sequence of 
77-dimensional state representations.
"""

import torch
import torch.nn as nn

class GRUWorldModel(nn.Module):
    def __init__(self, input_dim: int = 77, hidden_dim: int = 128, num_layers: int = 1, output_dim: int = 13):
        super(GRUWorldModel, self).__init__()
        
        # Temporal modeling: GRU
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # State Decoder: 128 -> 64 -> 13
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x: torch.Tensor, return_hidden: bool = False):
        """
        Forward pass for the GRU world model.
        
        Args:
            x (Tensor): Input sequence [B, SEQ_LEN, 77]
            return_hidden (bool): If True, returns (next_state, final_hidden)
            
        Returns:
            Tensor: Predicted next global state [B, 13]
            (and optionally the final hidden state [B, 128])
        """
        # GRU outputs:
        # out: [B, SEQ_LEN, hidden_dim]
        # h_n: [num_layers, B, hidden_dim]
        out, h_n = self.gru(x)
        
        # We want the final timestep representation for the batch
        # This is either out[:, -1, :] or h_n[-1, :, :]
        final_hidden = out[:, -1, :]
        
        # Decode to predict the next global state
        next_state = self.decoder(final_hidden)
        
        if return_hidden:
            return next_state, final_hidden
            
        return next_state
