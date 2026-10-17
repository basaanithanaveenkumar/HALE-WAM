"""HeadAttn: headAttn."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeadAttn(nn.Module):
    def __init__(self, emb_dim=256, head_size=16, drop_fact=0.0, causal_mask=False, return_attn_weights=False):
        super().__init__()
        self.em_dim = emb_dim

        self.query_proj = nn.Linear(emb_dim, head_size, bias=False)
        self.key_proj = nn.Linear(emb_dim, head_size, bias=False)
        self.val_proj = nn.Linear(emb_dim, head_size, bias=False)

        self.scale = 1.0 / math.sqrt(emb_dim)
        self.causal_mask = causal_mask
        self.dropout = nn.Dropout(drop_fact)
        self.return_attn_weights = return_attn_weights

    def forward(self, inputs, pad_mask=None):
        # inputs: [B, Seq_len, D]
        B, seq_len, D = inputs.shape
        Query = self.query_proj(inputs)  # [B, Seq_len, head_size]
        Key = self.key_proj(inputs)      # [B, Seq_len, head_size]
        Val = self.val_proj(inputs)      # [B, Seq_len, head_size]

        scores = torch.matmul(Query, Key.transpose(-2, -1)) * self.scale  # [B, Seq_len, Seq_len]

        if self.causal_mask:
            causal_tril = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=inputs.device))
            scores = scores.masked_fill(causal_tril == 0, float('-inf'))

        # pad_mask: [B, Seq_len] with 1=real, 0=pad; mask out pad key positions
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask.unsqueeze(1) == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)  # [B, Seq_len, Seq_len]
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, Val)  # [B, Seq_len, head_size]

        if self.return_attn_weights:
            return attn_output, attn_weights
        return attn_output
