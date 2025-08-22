import torch
import torch.nn as nn

class VisualPromptLearner(nn.Module):
    def __init__(self, clip_visual, num_vprompts, prompt_depth):
        """
        clip_visual: CLIP visual transformer
        num_vprompts: number of prompt tokens per layer
        prompt_depth: how many layers actually get visual prompts (<= total blocks)
        """
        super().__init__()
        self.clip_visual = clip_visual
        self.num_vprompts = num_vprompts
        self.prompt_depth = prompt_depth

    def forward(self, images, vprompts_list):
        """
        images: [B, C, H, W]
        vprompts_list: list of length = prompt_depth (J),
                       each element [B, num_vprompts, d]
        Returns: visual features [B, d]
        """
        B = images.size(0)

        # Patchify + CLS token + pos embed 
        x = self.clip_visual.conv1(images)               # patchify
        x = x.reshape(B, x.shape[1], -1)                 # [B, d, num_patches]
        x = x.permute(0, 2, 1)                           # [B, num_patches, d]

        cls_token = self.clip_visual.class_embedding.to(x.dtype)
        cls_token = cls_token + torch.zeros(B, 1, cls_token.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls_token, x], dim=1)             # prepend CLS

        pos_embed = self.clip_visual.positional_embedding[:x.size(1), :]
        x = x + pos_embed
 
        # Transformer blocks 
        blocks = self.clip_visual.transformer.resblocks
        for l, block in enumerate(blocks):
            if l < self.prompt_depth:   # only inject prompts in first J layers
                vprompts = vprompts_list[l]              # [B, num_vprompts, d]
                if vprompts is not None and vprompts.numel() > 0:
                    cls, rest = x[:, :1, :], x[:, 1 + self.num_vprompts:, :]
                    concat = torch.cat([cls, vprompts, rest], dim=1)

                    pos = self.clip_visual.positional_embedding[:concat.size(1), :].unsqueeze(0)
                    concat = concat + pos
                    x = block(concat)

                    # drop prompts before passing to next layer
                    x = torch.cat([x[:, :1, :], x[:, 1+self.num_vprompts:, :]], dim=1)
                else:
                    pos = self.clip_visual.positional_embedding[:x.size(1), :].unsqueeze(0)
                    x = x + pos
                    x = block(x)
            else:
                # just forward without prompts
                pos = self.clip_visual.positional_embedding[:x.size(1), :].unsqueeze(0)
                x = x + pos
                x = block(x)

        # Final CLS output
        x = self.clip_visual.ln_post(x[:, 0, :])   # CLS only
        if hasattr(self.clip_visual, "proj"):
            x = x @ self.clip_visual.proj
        return x
