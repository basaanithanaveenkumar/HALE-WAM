"""DecoderTransformer: decoderTransformer."""


import torch.nn as nn

from .transformer_block import TransformerBlock


class DecoderTransformer(nn.Module):
    def __init__(self, num_layers=16, emb_dim=1024, num_heads=32, mlp_dim=512,
                 drop_fact=0.0, use_moe=True, moe_hid_scale=1.2,
                 moe_num_routed_experts=16, moe_top_k=4,
                 moe_num_shared_experts=2, gradient_checkpointing=False):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.layers = nn.ModuleList([
            TransformerBlock(
                emb_dim=emb_dim, num_heads=num_heads, mlp_dim=mlp_dim,
                drop_fact=drop_fact, causal_mask=True,
                use_moe=use_moe,
                moe_hid_scale=moe_hid_scale,
                moe_num_routed_experts=moe_num_routed_experts,
                moe_top_k=moe_top_k,
                moe_num_shared_experts=moe_num_shared_experts,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, inputs, pad_mask=None):
        from torch.utils.checkpoint import checkpoint
        x = inputs
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                # Recompute activations during backward instead of storing them.
                # Saves ~activation memory per layer at the cost of one extra forward pass.
                x = checkpoint(layer, x, pad_mask, use_reentrant=False)
            else:
                x = layer(x, pad_mask=pad_mask)
        return self.norm(x)
