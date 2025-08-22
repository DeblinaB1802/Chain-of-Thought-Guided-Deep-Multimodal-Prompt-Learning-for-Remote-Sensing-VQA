# models/metanet.py
import torch.nn as nn

class MetaNetBlock(nn.Module):
    """
    One Meta-Net block (the j-th meta-net).
    - Accepts image feature `v` (shape: [B, dim]) and prev_bias (shape [B, out_dim]) or None.
    - Returns bias_j (shape [B, out_dim]).
    - Architecture inspired from paper: first linear (dim -> dim//16) then map to out_dim.
      Residual chaining: out = prev_out + MLP(v)
    """
    def __init__(self, dim, out_dim, bottleneck_factor=16, use_layernorm=True):
        super().__init__()
        hidden = max(1, dim // bottleneck_factor)
        
        self.dim_reduce = nn.Linear(dim, hidden)                                # first linear dim -> dim//16
        self.act = nn.ReLU(inplace=True)
        self.to_out = nn.Linear(hidden, out_dim)                                # Project reduced features to out_dim (out_dim = m * d)
        self.norm = nn.LayerNorm(out_dim) if use_layernorm else nn.Identity()   # normalization for stability

        # small init scale to stable prompt tuning
        nn.init.normal_(self.dim_reduce.weight, std=1e-3)
        nn.init.normal_(self.to_out.weight, std=1e-3)

    def forward(self, v, prev_bias=None):
        """
        v: [B, dim]
        prev_bias: [B, out_dim] or None
        returns: bias_j [B, out_dim]
        """
        x = self.dim_reduce(v)         # [B, hidden]
        x = self.act(x)
        x = self.to_out(x)             # [B, out_dim]
        x = self.norm(x)

        if prev_bias is None:
            return x
        else:
            return prev_bias + x       # residual addition: prev + current

class MetaNetChain(nn.Module):
    """
    Chain of MetaNetBlocks producing step-specific biases.
    - chain_length: number of steps (e.g., 3 in paper)
    - out_dim: dimension per-step bias (we'll set to prompt_len * text_embed_dim)
    """
    def __init__(self, dim, out_dim, chain_length=3, bottleneck_factor=16, use_layernorm=True):
        super().__init__()
        self.chain_length = chain_length
        self.blocks = nn.ModuleList([
            MetaNetBlock(dim, out_dim, bottleneck_factor=bottleneck_factor, use_layernorm=use_layernorm)
            for _ in range(chain_length)
        ])

    def forward(self, v):
        """
        v: [B, dim]
        returns: list of biases [bias_1, bias_2, ..., bias_n], each [B, out_dim]
        """
        biases = []
        prev = None
        for blk in self.blocks:
            prev = blk(v, prev)
            biases.append(prev)
        return biases
