"""
Value baseline ``V(s)`` for variance-reduced policy gradients.

Concept
-------
The vanilla REINFORCE gradient estimator,

    ∇θ J = E[ R · ∇θ log π(a|s) ],

is *unbiased* but has high variance: the magnitude of ``R`` scales every update,
so a large constant reward swamps the useful signal.  Subtracting a **baseline**
``b(s)`` that does not depend on the action leaves the gradient unbiased,

    ∇θ J = E[ (R − b(s)) · ∇θ log π(a|s) ],

because ``E_a[ b(s) ∇θ log π(a|s) ] = b(s) ∇θ E_a[1] = 0``.  The variance-
minimising choice is the state-value function ``b(s) = V^π(s)`` (the expected
return from ``s``), so we learn it with a small regression head and train it to
predict the observed returns.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ValueBaseline(nn.Module):
    """A small MLP that maps a conditioning vector to a scalar value ``V(s)``."""

    def __init__(self, cond_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """cond: [B, cond_dim] → value: [B]."""
        return self.net(cond).squeeze(-1)
