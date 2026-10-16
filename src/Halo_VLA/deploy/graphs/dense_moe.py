"""Export form of ``models.moe.DeepseekMoE``: dense expert evaluation, top-k without ``topk``."""

from __future__ import annotations

import torch
from torch import nn


class DenseMoE(nn.Module):
    """Numerically identical to ``DeepseekMoE`` in eval mode, with static shapes.

    ``DeepseekMoE`` dispatches tokens with boolean masks (``flat_x[mask]``, ``if mask.any()``),
    i.e. data-dependent shapes and control flow: ``torch.export`` refuses it and the legacy ONNX
    tracer would freeze the routing decisions of the tracing batch. Here every routed expert runs
    on every token and is weighted by its gate; unselected experts have gate exactly ``0`` (softmax
    over ``-inf``), so the sum equals the sparse dispatch. Cost: ``num_experts / top_k`` times the
    routed-expert FLOPs - small for Halo-VLA's 6-expert, top-2 decoder, and the usual way MoE
    layers are deployed on NPUs and in TensorRT engines without a grouped-GEMM plugin.

    The router's top-k is computed with ``k - 1`` rounds of "mask the current maximum" (``amax``
    + ``where``) instead of ``topk`` + ``scatter``, ops every backend in this project lowers
    (including the MLX bridge and the QNN HTP). Exact ties between router logits (probability ~0
    for float activations) could select more than ``k`` experts.

    Args:
        moe: the trained ``DeepseekMoE`` (its experts / router weights are shared).
    """

    def __init__(self, moe: nn.Module) -> None:
        super().__init__()
        self.shared_experts = moe.shared_experts
        self.routed_experts = moe.routed_experts
        self.router_logits = moe.router.bestk_layer  # eval mode: no router noise
        self.top_k = int(moe.best_k)

    def gates(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.router_logits(x)
        remaining = logits
        for _ in range(self.top_k - 1):
            remaining = torch.where(remaining >= remaining.amax(-1, keepdim=True),
                                    torch.full_like(remaining, float("-inf")), remaining)
        selected = logits >= remaining.amax(-1, keepdim=True)  # the top-k logits
        sparse = torch.where(selected, logits, torch.full_like(logits, float("-inf")))
        return torch.softmax(sparse, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        flat = x.reshape(b * s, d)
        gates = self.gates(x).reshape(b * s, len(self.routed_experts))
        out = torch.zeros_like(flat)
        for expert in self.shared_experts:
            out = out + expert(flat)
        for i, expert in enumerate(self.routed_experts):
            out = out + expert(flat) * gates[:, i:i + 1]
        return out.reshape(b, s, d)
