"""SelfAttn: selfAttn."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttn(nn.Module):
    def __init__(self, emb_dim, return_attn_weights=False):
        super().__init__()
        self.em_dim = emb_dim

        self.query_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.key_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.val_proj = nn.Linear(emb_dim, emb_dim, bias=False)

        self.scale = 1.0 / math.sqrt(emb_dim)
        self.return_attn_weights = return_attn_weights

    def forward(self, inputs, causal_mask=None, pad_mask=None):
        # inputs: [B, Seq_len, D]
        Query = self.query_proj(inputs)  # [B, Seq_len, D]
        Key = self.key_proj(inputs)      # [B, Seq_len, D]
        Val = self.val_proj(inputs)      # [B, Seq_len, D]

        attn_scores = torch.matmul(Query, Key.transpose(-2, -1)) * self.scale  # [B, Seq_len, Seq_len]

        if causal_mask is not None:
            attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))

        # pad_mask: [B, Seq_len] with 1=real, 0=pad; mask out pad key positions
        if pad_mask is not None:
            attn_scores = attn_scores.masked_fill(pad_mask.unsqueeze(1) == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, Seq_len, Seq_len]
        attn_output = torch.matmul(attn_weights, Val)   # [B, Seq_len, D]

        if self.return_attn_weights:
            return attn_output, attn_weights
        return attn_output
