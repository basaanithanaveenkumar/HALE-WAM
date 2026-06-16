"""
Group Relative Policy Optimization (GRPO).

GRPO (Shao et al., DeepSeekMath, 2024) is a critic-free variant of PPO.  PPO
needs a learned value network to compute advantages; GRPO removes it by
sampling a **group** of ``G`` actions for the *same* state and using the group's
own reward statistics as the baseline.  This is a natural fit for a world-model
loop where you already roll out several imagined trajectories per state — you
might as well rank them against each other instead of training a separate value
head.

Algorithm
---------
For each state ``s`` (conditioning vector):

1. Sample a group of ``G`` actions:        a_1..a_G ~ π_old(·|s)
2. Reward each one:                         R_1..R_G
3. Group-relative advantage (the key idea):
       Â_i = (R_i − mean_j R_j) / (std_j R_j + ε)
   i.e. "how much better than its siblings", scale-normalised.  No critic.
4. Clipped PPO surrogate per sample, with ratio r_i = π_θ(a_i|s) / π_old(a_i|s):
       L_i = min( r_i · Â_i ,  clip(r_i, 1−ε, 1+ε) · Â_i )
5. Subtract a KL penalty to a frozen reference policy to stay close to it:
       J = E[ L_i ] − β · KL(π_θ ‖ π_ref)

We *minimise* ``−J``.  Because the advantage is mean-centred within each group,
the method is invariant to any constant shift of the reward and needs no value
function — only relative ordering inside the group matters.
"""

from __future__ import annotations

import torch
from torch.distributions import kl_divergence

from rl.policy import GaussianActionPolicy


def group_relative_advantage(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Group-normalise rewards into advantages.

    Args:
        rewards: [N, G] — ``G`` rewards per group, for ``N`` groups.
        eps:     numerical floor added to the std (and guards the G==1 case).

    Returns:
        advantages: [N, G] — each row is (approximately) zero-mean, unit-std.
    """
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True, unbiased=False)
    return (rewards - mean) / (std + eps)


def compute_grpo_loss(
    policy: GaussianActionPolicy,
    ref_policy: GaussianActionPolicy,
    cond: torch.Tensor,
    reward_fn,
    *,
    group_size: int = 8,
    clip_eps: float = 0.2,
    kl_beta: float = 0.04,
) -> tuple[torch.Tensor, dict]:
    """Compute the GRPO loss for a batch of states.

    Args:
        policy:     the policy π_θ being optimised.
        ref_policy: a frozen reference policy π_ref (KL anchor).  In single-update
                    on-policy GRPO this also serves as the behaviour policy π_old,
                    so the importance ratio starts at 1.
        cond:       [N, cond_dim] — one conditioning vector per group/state.
        reward_fn:  callable ``actions[M, A] -> reward[M]``.
        group_size: number of samples ``G`` per state.
        clip_eps:   PPO clip range ε.
        kl_beta:    KL-penalty weight β.

    Returns:
        (loss, metrics).
    """
    N, D = cond.shape
    G = group_size

    # Expand each state into G rows: [N, D] -> [N*G, D]. repeat_interleave keeps
    # all G copies of a state contiguous, which we rely on when reshaping to
    # [N, G] for the group statistics.
    cond_rep = cond.repeat_interleave(G, dim=0)                  # [N*G, D]

    # 1. Sample a group of actions from the (frozen) behaviour policy.
    with torch.no_grad():
        dist_old = ref_policy.distribution(cond_rep)
        actions = dist_old.sample()                             # [N*G, A]
        log_prob_old = dist_old.log_prob(actions)              # [N*G]

    # 2. Reward every sampled action.
    with torch.no_grad():
        rewards = reward_fn(actions)                           # [N*G]

    # 3. Group-relative advantages (critic-free baseline).
    adv = group_relative_advantage(rewards.view(N, G)).view(N * G)  # [N*G]

    # 4. New-policy log-prob of the same actions → importance ratio.
    log_prob_new = policy.log_prob(cond_rep, actions)          # [N*G], grad in θ
    ratio = torch.exp(log_prob_new - log_prob_old)            # [N*G]

    # Clipped surrogate: take the pessimistic (min) of clipped vs unclipped so a
    # too-large policy update on a positive advantage is not rewarded.
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    surrogate = torch.min(unclipped, clipped).mean()

    # 5. KL(π_θ ‖ π_ref): exact for two diagonal Gaussians.
    kl = kl_divergence(
        policy.distribution(cond_rep), ref_policy.distribution(cond_rep)
    ).mean()

    loss = -surrogate + kl_beta * kl

    metrics = {
        "loss": float(loss.detach()),
        "surrogate": float(surrogate.detach()),
        "kl": float(kl.detach()),
        "ratio_mean": float(ratio.mean().detach()),
        "reward_mean": float(rewards.mean()),
        "adv_abs_mean": float(adv.abs().mean()),
    }
    return loss, metrics
