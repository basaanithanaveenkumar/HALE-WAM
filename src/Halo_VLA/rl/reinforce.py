"""
REINFORCE with a learned value baseline.

Algorithm (one-step / bandit form used for action-chunk policies)
----------------------------------------------------------------
For each conditioning vector ``s`` (the observation embedding):

1. Sample an action chunk from the policy:   a ~ π_θ(·|s)
2. Score it with the reward function:        R = R(a)
3. Estimate the state value:                 V = V_φ(s)
4. Advantage (how much better than expected): A = R − V          (detached)
5. Policy (actor) loss:                       L_π = −E[ A · log π_θ(a|s) ]
6. Value (critic) loss:                       L_V = E[ (V_φ(s) − R)² ]
7. Entropy bonus (exploration):               −β · E[ H[π_θ(·|s)] ]

The total loss minimised is ``L_π + c_v · L_V − β · H``.  Minimising ``L_π``
performs gradient *ascent* on expected reward (note the leading minus), the
baseline ``V`` reduces variance without biasing the gradient, and the entropy
term keeps the policy from collapsing prematurely.

This is the classic Williams (1992) estimator with the value-function baseline
of Sutton & Barto, adapted to a single-decision continuous action chunk.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rl.policy import GaussianActionPolicy
from rl.value_baseline import ValueBaseline


def compute_reinforce_loss(
    policy: GaussianActionPolicy,
    value: ValueBaseline,
    cond: torch.Tensor,
    reward_fn,
    *,
    use_baseline: bool = True,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """Compute the REINFORCE-with-baseline loss.

    Args:
        policy:       the Gaussian action policy being trained.
        value:        the value baseline ``V_φ(s)``.
        cond:         [B, cond_dim] conditioning / observation embeddings.
        reward_fn:    callable ``actions[B,A] -> reward[B]`` (see ``rl.rewards``).
        use_baseline: if False, advantage = raw reward (vanilla REINFORCE).  Kept
                      as a flag so tests can demonstrate the variance reduction.
        entropy_coef: weight β of the entropy bonus.
        value_coef:   weight c_v of the critic regression loss.

    Returns:
        (loss, metrics) where ``loss`` is a scalar tensor and ``metrics`` is a
        dict of detached floats for logging.
    """
    # 1. Sample an action and its (differentiable) log-probability.
    #    ``sample`` returns a detached action so log_prob's gradient is the pure
    #    score function ∇θ log π(a|s).
    action, log_prob = policy.sample(cond)           # [B, A], [B]

    # 2. Reward is an external signal — no gradient flows through it.
    with torch.no_grad():
        reward = reward_fn(action)                   # [B]

    # 3. Critic prediction of the return for this state.
    value_pred = value(cond)                         # [B]

    # 4. Advantage.  Detached: the actor update treats (R − V) as a constant
    #    weight on ∇θ log π; we do not want gradients flowing into V through the
    #    actor term (V is trained by its own regression loss below).
    if use_baseline:
        advantage = (reward - value_pred).detach()   # [B]
    else:
        advantage = reward.detach()                  # [B]

    # 5. Actor loss: maximise A·logπ  ==  minimise −A·logπ.
    policy_loss = -(advantage * log_prob).mean()

    # 6. Critic loss: regress V_φ(s) onto the observed return R.
    value_loss = F.mse_loss(value_pred, reward)

    # 7. Entropy bonus: subtract entropy so minimisation *increases* it.
    entropy = policy.entropy(cond).mean()

    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

    metrics = {
        "loss": float(loss.detach()),
        "policy_loss": float(policy_loss.detach()),
        "value_loss": float(value_loss.detach()),
        "entropy": float(entropy.detach()),
        "reward_mean": float(reward.mean()),
        "advantage_mean": float(advantage.mean()),
        "advantage_std": float(advantage.std(unbiased=False)),
    }
    return loss, metrics
