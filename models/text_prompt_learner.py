# Import necessary libraries
import torch
import torch.nn as nn
from metanet import MetaNetChain
from prompt_chain import PromptChainCell  

def _get_transformer_blocks(transformer):
    """Robustly extract transformer blocks (resblocks / layers / ModuleList)."""
    blocks = getattr(transformer, "resblocks", None)
    if blocks is not None:
        return blocks
    blocks = getattr(transformer, "layers", None)
    if blocks is not None:
        return blocks
    if isinstance(transformer, (nn.ModuleList, nn.Sequential)):
        return transformer
    # fallback: try iterating submodules
    return list(transformer.children())


class TextPromptLearner(nn.Module):
    def __init__(self,
                 clip_model,
                 prompt_len: int = 4,
                 prompt_depth: int = 9,      
                 num_vprompts: int = 9,
                 bottleneck_factor: int = 16):
        """
        - clip_model: loaded CLIP model (frozen externally)
        - prompt_len: m
        - prompt_depth: J (how many early transformer layers receive prompts)
        - num_vprompts: number of visual prompt tokens per visual layer
        """
        super().__init__()
        self.clip = clip_model
        self.prompt_len = prompt_len  # m (tokens per prompt)
        self.prompt_depth = prompt_depth # J (Numbers of text prompts)
        self.num_vprompts = num_vprompts

        # Embedding Dimension (eg: 512 or 768)
        self.text_d = clip_model.token_embedding.weight.shape[1]  # d (prompt token's embedding dimension)
        self.vis_d = clip_model.visual.positional_embedding.shape[1]

        # per-layer base prompts (shape: [J, m, d])
        self.base_prompts = nn.Parameter(torch.randn(prompt_depth, prompt_len, self.text_d) * 0.02)  # Very small prompt embedding initiation.

        # MetaNetChain producing biases - one bias per prompt layer
        out_dim = prompt_len * self.text_d   # out_dim = m * d

        # lazy init metanet if clip visual output dim unknown
        img_dim = getattr(self.clip.visual, "output_dim", None)
        if img_dim is None:
            # we will lazy-initialize MetaNetChain on first forward if necessary
            self.metanet = None
            self._metanet_img_dim = None
        else:
            self.metanet = MetaNetChain(dim=img_dim, out_dim=out_dim, chain_length=prompt_depth, bottleneck_factor=bottleneck_factor)
            self._metanet_img_dim = img_dim

        # Prompt chain GRU-like cell
        self.chain_cell = PromptChainCell(self.text_d)

        # For each prompt token we build a small projection that maps concat([base, bias]) (2*d) -> d. This implements "before GRU the base prompt and image-conditioned bias are concatenated"
        self.base_bias_proj = nn.ModuleList([
            nn.Linear(2 * self.text_d, self.text_d) for _ in range(prompt_depth)
        ])
        for lin in self.base_bias_proj:
            nn.init.xavier_uniform_(lin.weight)

        # text->visual mapping: one small MLP per prompt layer (maps flattened S -> vprompts for that layer) ---
        self.text_to_vis = nn.ModuleList([
            nn.Sequential(
                nn.Linear(prompt_len * self.text_d, max(256, prompt_len * self.text_d // 4)),
                nn.ReLU(inplace=True),
                nn.Linear(max(256, prompt_len * self.text_d // 4), num_vprompts * self.vis_d)
            ) for _ in range(prompt_depth)
        ])
        for mlp in self.text_to_vis:
            for p in mlp.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        # layer-wise visual prompt positional offsets (learnable) for each prompt layer
        self.vprompt_pos = nn.Parameter(torch.zeros(prompt_depth, num_vprompts, self.vis_d))
        nn.init.normal_(self.vprompt_pos, std=1e-3)

    def _lazy_init_metanet(self, image_feat_dim):
        if self.metanet is None:
            out_dim = self.prompt_len * self.text_d

            self.metanet = MetaNetChain(dim=image_feat_dim, out_dim=out_dim, chain_length=self.prompt_depth)
            self._metanet_img_dim = image_feat_dim

    def forward(self, image_features: torch.Tensor, tokenized_questions: torch.Tensor):
        """
        image_features: [B, img_feat_dim]  (from clip.encode_image(images) - frozen)
        tokenized_questions: [B, T]        (clip.tokenize(...))
        Returns:
            text_feats: [B, embed_dim]
            vprompts_full: [B, num_vit_layers, num_vprompts, vis_d]  (zero for non-prompt layers)
            all_text_prompts: list of length prompt_depth of S_j ([B, m, d])
        """
        B = image_features.shape[0]

        # lazy init metanet if needed
        if self._metanet_img_dim is None or self._metanet_img_dim != image_features.shape[1]:
            self._lazy_init_metanet(image_features.shape[1])

        # get biases (list length = prompt_depth, each [B, m*d])
        biases = self.metanet(image_features)   # list: biases[0], biases[1], ..., biases[J-1]

        # token embeddings (original token embedding for the whole sentence)
        tok_emb = self.clip.token_embedding(tokenized_questions)   # [B, T, d]

        # We'll split into CLS and the rest tokens
        cls_tok = tok_emb[:, :1, :]         # [B,1,d]
        rest_orig = tok_emb[:, 1:, :]       # [B, T-1, d]

        # extract text transformer blocks
        text_transformer = self.clip.transformer
        text_blocks = _get_transformer_blocks(text_transformer)  # iterable of blocks
        num_text_layers = len(list(text_blocks))

        # We'll run through all text layers but only perform deep prompting for first J = prompt_depth layers
        J = min(self.prompt_depth, len(biases), num_text_layers)

        # prev_S init: use base_prompts[0] expanded (so chain has a reasonable starting state)
        prev_S = self.base_prompts[0].unsqueeze(0).expand(B, -1, -1)  # [B, m, d]
        all_text_prompts = []
        vprompts_list = []   # will collect vprompts for layers 0..J-1

        # Current sequence x: we'll build appropriate sequence and feed blocks sequentially.
        # For the first iteration we need to create concat for layer 0 using cls_tok and rest_orig
        x = None
        current_has_prompts = False

        for layer_idx, blk in enumerate(text_blocks):
            if layer_idx < J:

                # build cur_P from base_prompt[layer_idx] and bias[layer_idx]
                base_layer = self.base_prompts[layer_idx].unsqueeze(0).expand(B, -1, -1)   # [B, m, d]
                bias_vec = biases[layer_idx].view(B, self.prompt_len, self.text_d)         # [B, m, d]

                # concat along feature dim -> [B, m, 2d]
                cat = torch.cat([base_layer, bias_vec], dim=-1)                            # [B, m, 2d]

                # project to d: cur_P [B,m,d]
                cur_P = self.base_bias_proj[layer_idx](cat)

                # GRU-like update using previous S and cur_P
                S = self.chain_cell(prev_S, cur_P)   # [B, m, d]
                all_text_prompts.append(S)

                prev_S = S  # update for next iteration

                # --- prepare sequence for this block: use latest CLS and the appropriate 'rest' tokens ---
                if x is None:
                    # first prompt layer: use original cls_tok and rest_orig
                    cls_in = cls_tok                                         # [B,1,d]
                    rest_in = rest_orig
                    concat = torch.cat([cls_in, rest_in], dim=1)                                      # [B, T-1, d]
                else:
                    # x contains output of previous block.
                    # If previous block had prompts, then previous x layout: [CLS, prompt, rest_updated]
                    if current_has_prompts:
                        cls_in = x[:, :1, :]                                  # [B,1,d]
                        rest_in = x[:, 1 + self.prompt_len:, :]              # drop previous prompts
                        # concat for this layer: [CLS, S, rest_in]
                        concat = torch.cat([cls_in, S, rest_in], dim=1)  # [B, 1+m + rest_len, d]

                # add positional embeddings slice
                pos_slice = self.clip.positional_embedding[:concat.size(1), :].unsqueeze(0).to(concat.device)
                inp = concat + pos_slice

                # transformer blocks in CLIP text usually expect shape [L, B, d] if using the transformer's forward,
                # but individual block implementations often accept [B, L, d]. We'll attempt calling blk(inp) directly;
                # if it fails, permute as fallback.
                try:
                    x = blk(inp)
                except Exception:
                    x = inp.permute(1, 0, 2)
                    x = blk(x)
                    x = x.permute(1, 0, 2)

                current_has_prompts = True

                # generate visual prompts for this same layer from S
                flat = S.view(B, -1)   # [B, m*d]
                v_out = self.text_to_vis[layer_idx](flat).view(B, self.num_vprompts, self.vis_d)  # [B, n, d_vis]
                v_out = v_out + self.vprompt_pos[layer_idx].unsqueeze(0)  # add learned offset
                vprompts_list.append(v_out)  # one entry per prompt layer

            else:
                if current_has_prompts:
                    # remove the prompt tokens from x before next block
                    cls_in = x[:, :1, :]
                    rest_in = x[:, 1:, :]
                    inp = torch.cat([cls_in, rest_in], dim=1)

                # add positional embeddings slice (ensure device)
                pos_slice = self.clip.positional_embedding[:inp.size(1), :].unsqueeze(0).to(inp.device)
                inp = inp + pos_slice

                try:
                    x = blk(inp)
                except Exception:
                    x = inp.permute(1, 0, 2)
                    x = blk(x)
                    x = x.permute(1, 0, 2)

                #Current_has_prompts = False

        # --- At the end, compute final text features ---
        # Need to locate the final token position (original EOS index) in the final sequence.
        orig_seq_lens = tokenized_questions.argmax(dim=-1)  # [B] (index in original tokenization)
        if current_has_prompts:
            # final representation sequence still contains prompts => EOS index is shifted by prompt_len
            final_positions = orig_seq_lens + self.prompt_len
        else:
            final_positions = orig_seq_lens

        text_feats = x[torch.arange(B), final_positions] @ self.clip.text_projection  # [B, embed_dim]

        return text_feats, vprompts_list, all_text_prompts