"""DeepseekMoE: deepseekMoE."""

import torch
import torch.nn as nn

from .expert import Expert
from .noise_best_k_router import NoiseBestKRouter


#  Added the shared expert to deepseek model
class DeepseekMoE(nn.Module):
    def __init__(self, emb_dim,hid_dim,num_router_exprts, best_k, num_shared_exprts):
        super(DeepseekMoE, self).__init__()
        self.router = NoiseBestKRouter(emb_dim, num_router_exprts, best_k)
        self.shared_experts = nn.ModuleList([
            Expert(emb_dim, hid_dim) 
            for _ in range(num_shared_exprts)
        ])
        self.routed_experts = nn.ModuleList([Expert(emb_dim,hid_dim) for _ in range(num_router_exprts)])
        self.best_k = best_k

    def forward(self, x):
        batch, seq, dim = x.shape
        x_flat = x.view(-1, dim)
        shared_output = 0
        for expert in self.shared_experts:
            shared_output += expert(x_flat)

        gating_output, idxs = self.router(x)
        final_output = torch.zeros_like(x)
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        for i, expert in enumerate(self.routed_experts):
            expert_mask = (idxs == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores
                final_output[expert_mask] += weighted_output.squeeze(1)
        return final_output + shared_output.view(batch,seq,dim)
