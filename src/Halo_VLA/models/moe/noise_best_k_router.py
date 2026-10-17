"""NoiseBestKRouter: noiseBestKRouter."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseBestKRouter(nn.Module):
    def __init__(self, emb_dim, num_exprts, best_k):
        super(NoiseBestKRouter, self).__init__()
        self.best_k = best_k
        self.bestk_layer = nn.Linear(emb_dim, num_exprts)
        self.noise_linear =nn.Linear(emb_dim, num_exprts)


    def forward(self, x):
        logits = self.bestk_layer(x)
        noise_logits = self.noise_linear(x)
        # add the gaussian noise to logits
        if self.training:
            noise = torch.randn_like(logits)*F.softplus(noise_logits)
            noisy_logits = logits + noise
        else:
            noisy_logits = logits

        best_k_logits, idxs = noisy_logits.topk(self.best_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, idxs, best_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, idxs
