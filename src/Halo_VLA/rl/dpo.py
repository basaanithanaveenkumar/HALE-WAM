"""
Direct Preference Optimization (DPO) on action sequences.

DPO (Rafailov et al., 2023) skips the RL loop entirely.  Instead of sampling,
rewarding, and doing policy gradients, it directly trains the policy on
*preference pairs* with a simple supervised-style classification loss.  Its key
result: the RLHF objective "maximise reward subject to a KL constraint to a
reference policy" has a closed-form optimal policy, and substituting it back
turns reward modelling + RL into one log-sigmoid loss over preferences.

Loss
----
For a preference triple ``(s, a_w, a_l)`` (winner ``a_w`` preferred over loser
``a_l``):

    L = − log σ( β · [ (log π_θ(a_w|s) − log π_ref(a_w|s))
                       − (log π_θ(a_l|s) − log π_ref(a_l|s)) ] )

Each bracketed term is an *implicit reward* ``r(s,a) = β·log(π_θ/π_ref)``.  The
loss is exactly binary cross-entropy on "is the winner's implicit reward higher
than the loser's?".  β controls how far π_θ may drift from the reference π_ref
(larger β = tighter to the reference behaviour).

Why a reference policy?  Without the ``− log π_ref`` anchor the model could push
the winner's probability to 1 and collapse; the reference keeps the update
conservative, equivalent to the KL constraint in RLHF.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rl.policy import GaussianActionPolicy


def compute_dpo_loss(
    policy: GaussianActionPolicy,
    ref_policy: GaussianActionPolicy,
    cond: torch.Tensor,
    chosen: torch.Tensor,
    rejected: torch.Tensor,
    *,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """Compute the DPO loss for a batch of preference pairs.

    Args:
        policy:     the policy π_θ being trained.
        ref_policy: the frozen reference policy π_ref (typically the SL policy).
        cond:       [B, cond_dim] conditioning vectors.
        chosen:     [B, A] preferred actions ``a_w``.
        rejected:   [B, A] dispreferred actions ``a_l``.
        beta:       temperature β controlling deviation from the reference.

    Returns:
        (loss, metrics).
    """
    # Log-probs under the trainable policy (carry gradients).
    logp_w = policy.log_prob(cond, chosen)            # [B]
    logp_l = policy.log_prob(cond, rejected)          # [B]

    # Log-probs under the frozen reference (no gradients).
    with torch.no_grad():
        ref_logp_w = ref_policy.log_prob(cond, chosen)
        ref_logp_l = ref_policy.log_prob(cond, rejected)

    # Per-sample implicit-reward differences (log-ratios).
    chosen_logratio = logp_w - ref_logp_w             # [B]
    rejected_logratio = logp_l - ref_logp_l           # [B]

    # The DPO logit: how much more the winner is preferred over the loser.
    logits = beta * (chosen_logratio - rejected_logratio)   # [B]

    # −log σ(logits) == softplus(−logits): the binary-cross-entropy form.
    loss = F.softplus(-logits).mean()

    # Implicit rewards for logging / monitoring.
    chosen_reward = (beta * chosen_logratio).detach()
    rejected_reward = (beta * rejected_logratio).detach()

    metrics = {
        "loss": float(loss.detach()),
        # Fraction of pairs the model currently orders correctly.
        "accuracy": float((logits > 0).float().mean()),
        "margin": float((chosen_reward - rejected_reward).mean()),
        "chosen_reward": float(chosen_reward.mean()),
        "rejected_reward": float(rejected_reward.mean()),
    }
    return loss, metrics
