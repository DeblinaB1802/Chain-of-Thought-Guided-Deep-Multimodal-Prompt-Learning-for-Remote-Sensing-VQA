# models/prompt_chain.py
import torch
import torch.nn as nn

class PromptChainCell(nn.Module):
    """
    Token-wise GRU-like update over prompt token vectors.
    Inputs: prev_S [B, m, d], cur_P [B, m, d] -> output S [B, m, d]
    Shared linear params across tokens (applied to last-dim concatenation).
    """
    def __init__(self, d):
        super().__init__()
        self.z_lin = nn.Linear(2 * d, d)
        self.r_lin = nn.Linear(2 * d, d)
        self.h_lin = nn.Linear(2 * d, d)
        # initialize nicely
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, prev_S, cur_P):
        # prev_S, cur_P: [B, m, d]
        
        x = torch.cat([prev_S, cur_P], dim=-1)  # [B, m, 2d]
        z = torch.sigmoid(self.z_lin(x))
        r = torch.sigmoid(self.r_lin(x))
        h_tilde = torch.tanh(self.h_lin(torch.cat([r * prev_S, cur_P], dim=-1)))
        S = (1 - z) * prev_S + z * h_tilde
        return S
