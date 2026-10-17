"""Expert: expert."""

import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, emb_dim, hid_dim, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(emb_dim, hid_dim, bias=False)
        self.w2 = nn.Linear(hid_dim, emb_dim, bias=False)
        self.w3 = nn.Linear(emb_dim, hid_dim, bias=False) 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # SwiGLU activation: (Swish(xW1) * xW3) * 
        # TODO need to change the activation to GeLU
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
