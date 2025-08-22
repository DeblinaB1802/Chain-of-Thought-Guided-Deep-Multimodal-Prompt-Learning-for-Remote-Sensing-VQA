import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionClassifier(nn.Module):
    """
    Flexible fusion head for VQA/RSVQA given projected text & image embeddings.

    Args:
        embed_dim: D (dimension of text/image projected features)
        num_answers: size of answer vocabulary for RSVQA
        fusion: one of {"concat", "hadamard", "gated_tanh"}
        hidden_dims: MLP hidden sizes after fusion
        dropout: dropout prob applied after fusion and between MLP layers
        norm_inputs: if True, L2-normalize text & image embeddings before fusion
    """
    def __init__(
        self,
        embed_dim: int,
        num_answers: int,
        fusion: str = "gated_tanh",
        hidden_dims = (1024, 512),
        dropout: float = 0.3,
        norm_inputs: bool = True,
    ):
        super().__init__()
        self.D = embed_dim
        self.num_answers = num_answers
        self.fusion = fusion
        self.norm_inputs = norm_inputs
        self.dropout = nn.Dropout(dropout)

        if fusion == "concat":
            fused_dim = 2 * self.D
            self.fuse = nn.Identity()  # concatenation done in forward
        elif fusion == "hadamard":
            fused_dim = self.D
            self.fuse = nn.Identity()  # elementwise product done in forward
        elif fusion == "gated_tanh":
            # classic VQA gating: g = sigmoid(Wg [t;i]); h = tanh(Wh [t;i]); z = g * h
            in_dim = 2 * self.D
            self.Wg = nn.Linear(in_dim, self.D)
            self.Wh = nn.Linear(in_dim, self.D)
            fused_dim = self.D
        else:
            raise ValueError(f"Unknown fusion '{fusion}'")

        # MLP classifier head
        mlp_layers = []
        prev = fused_dim
        for h in (hidden_dims if isinstance(hidden_dims, (list, tuple)) else [hidden_dims]):
            mlp_layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        mlp_layers += [nn.Linear(prev, num_answers)]
        self.classifier = nn.Sequential(*mlp_layers)

        # Optional small LayerNorm on inputs to stabilize (esp. for concat)
        self.ln_t = nn.LayerNorm(self.D)
        self.ln_i = nn.LayerNorm(self.D)

    def forward(self, text_feats: torch.Tensor, image_feats: torch.Tensor) -> torch.Tensor:
        """
            text_feats:  [B, D]
            image_feats: [B, D]
            returns: logits [B, num_answers]
        """
        t = self.ln_t(text_feats)
        v = self.ln_i(image_feats)

        if self.norm_inputs:
            t = F.normalize(t, dim=-1)
            v = F.normalize(v, dim=-1)

        if self.fusion == "concat":
            z = torch.cat([t, v], dim=-1)               # [B, 2D]
        elif self.fusion == "hadamard":
            z = t * v                                   # [B, D]
        elif self.fusion == "gated_tanh":
            x = torch.cat([t, v], dim=-1)               # [B, 2D]
            g = torch.sigmoid(self.Wg(x))               # [B, D]
            h = torch.tanh(self.Wh(x))                  # [B, D]
            z = g * h                                   # [B, D]
        else:
            raise RuntimeError("Unexpected fusion type")

        z = self.dropout(z)
        logits = self.classifier(z)                     # [B, num_answers]
        return logits
