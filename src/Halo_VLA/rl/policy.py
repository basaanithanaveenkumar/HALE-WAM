"""
Gaussian action policy for RL fine-tuning of Halo-VLA.

Concept
-------
A *policy* in reinforcement learning is a conditional distribution over actions,
``π(a | s)``.  For continuous control the most common choice is a **diagonal
Gaussian**:

    π(a | s) = N(a ; μ_θ(s), diag(σ_θ²))

where ``μ_θ(s)`` is produced by a neural network and ``σ`` is a (here global,
state-independent) standard deviation.  A Gaussian policy is convenient because
three quantities every policy-gradient / preference method needs are all
closed-form:

* sampling           a ~ π(·|s)
* log-probability    log π(a | s)
* differential entropy  H[π(·|s)]   (encourages exploration when maximised)

Design
------
The policy is deliberately **decoupled from** :class:`models.halo_vla.HaloVLM`.
It consumes a plain *conditioning vector* ``cond`` of shape ``[B, cond_dim]`` —
typically the transformer hidden state at the ``<halo_action>`` token, i.e. the
same signal the supervised flow decoder is conditioned on.  Operating on a bare
tensor means the policy (and its unit tests) need nothing but ``torch`` and can
run on CPU in milliseconds.  The static helper
:meth:`GaussianActionPolicy.conditioning_from_halo` shows how to obtain that
vector from a real Halo-VLA forward pass when wiring up the training script.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Independent, Normal


class GaussianActionPolicy(nn.Module):
    """Diagonal-Gaussian policy over a flattened action chunk.

    Args:
        cond_dim:        dimensionality of the conditioning vector ``s`` (e.g.
                         ``config.emb_dim`` — the transformer hidden size).
        action_dim_flat: number of scalars in one (flattened) action sample,
                         e.g. ``action_chunk_size * action_dim``.
        hidden_dim:      width of the mean network's hidden layers.
        log_std_init:    initial value of the learnable log-standard-deviation.
        log_std_min/max: clamp range for ``log σ`` — keeps σ in a numerically
                         safe band so the distribution never collapses (σ→0,
                         infinite density) or explodes (σ→∞, no learning signal).
    """

    def __init__(
        self,
        cond_dim: int,
        action_dim_flat: int,
        hidden_dim: int = 512,
        log_std_init: float = -0.5,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.action_dim_flat = action_dim_flat
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Mean network μ_θ(s): maps the conditioning vector to the action mean.
        self.mean_net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim_flat),
        )

        # A single, state-independent log-σ vector shared across all states.
        # This is the standard "global log-std" parameterisation used by PPO and
        # friends — simple, stable, and enough to control exploration. It is a
        # learnable nn.Parameter so the optimiser can shrink σ as the policy
        # becomes confident.
        self.log_std = nn.Parameter(torch.full((action_dim_flat,), float(log_std_init)))

    # ------------------------------------------------------------------ #
    # Core distribution
    # ------------------------------------------------------------------ #
    def distribution(self, cond: torch.Tensor) -> Independent:
        """Return the per-state action distribution as a batched Independent
        Normal so that ``log_prob`` / ``entropy`` reduce over the action
        dimension and yield one scalar **per sample** (shape ``[B]``).
        """
        mean = self.mean_net(cond)                                  # [B, A]
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp().expand_as(mean)                         # [B, A]
        # Independent(..., 1) treats the last dim as event dims, so log_prob
        # sums the per-coordinate log-densities → a single value per sample.
        return Independent(Normal(mean, std), 1)

    # ------------------------------------------------------------------ #
    # Sampling and scoring
    # ------------------------------------------------------------------ #
    def sample(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw an action with the **score-function** estimator in mind.

        We use ``dist.sample()`` (NOT ``rsample``) so the returned action is
        detached from the computation graph.  ``log_prob`` is then computed on
        that fixed action, so gradients flow **only** through the distribution
        parameters (μ, σ).  This is exactly what REINFORCE / GRPO want:
        ``∇θ log π(a|s)`` with ``a`` held constant.

        Returns:
            action:  [B, action_dim_flat] — sampled action (no grad).
            log_prob:[B]                  — log π(a|s), differentiable in θ.
        """
        dist = self.distribution(cond)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def rsample(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparameterised sample (pathwise gradient ``a = μ + σ·ε``).

        Use this only when you *want* gradients to flow through the action
        itself (e.g. Dreamer-style pathwise actor updates), not for REINFORCE.
        """
        dist = self.distribution(cond)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def log_prob(self, cond: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """log π(action | cond) → [B]. Differentiable in the policy params."""
        return self.distribution(cond).log_prob(action)

    def entropy(self, cond: torch.Tensor) -> torch.Tensor:
        """Differential entropy H[π(·|cond)] → [B]. Higher = more exploration."""
        return self.distribution(cond).entropy()

    # ------------------------------------------------------------------ #
    # Integration helper
    # ------------------------------------------------------------------ #
    @staticmethod
    @torch.no_grad()
    def conditioning_from_halo(model, batch: dict, device=None) -> torch.Tensor:
        """Extract a conditioning vector from a real Halo-VLA forward pass.

        Runs ``HaloVLM.forward`` and mean-pools the transformer hidden states at
        the ``<halo_action>`` token positions into a single ``[B, emb_dim]``
        vector per sample — the natural "observation embedding" for the policy.

        Args:
            model: a ``models.halo_vla.HaloVLM`` instance.
            batch: dict with ``images``, ``input_ids``, ``attention_mask``,
                   ``states`` and optional ``image_mask`` (as the dataloaders
                   produce).
            device: optional torch device to move tensors to.

        Returns:
            cond: [B, emb_dim].  Rows are zero where a sample had no action token.
        """
        def to(x):
            return x.to(device) if (device is not None and x is not None) else x

        _, action_hiddens, _, _ = model(
            images=to(batch["images"]),
            input_ids=to(batch["input_ids"]),
            attention_mask=to(batch["attention_mask"]),
            states=to(batch["states"]),
            image_mask=to(batch.get("image_mask")),
        )
        if action_hiddens is None:
            raise ValueError(
                "No <halo_action> tokens found in batch; cannot build policy "
                "conditioning. Ensure input_ids contain the action token."
            )
        # [B, n_act, emb_dim] -> mean over action tokens -> [B, emb_dim]
        return action_hiddens.mean(dim=1)
