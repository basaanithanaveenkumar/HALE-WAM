"""System1Policy: fast policy: current frame + proprioception + System 2 latent -> action chunk."""

from __future__ import annotations

import torch
import torch.nn as nn

from dual_system.config import DualSystemConfig

from .action_transformer import ActionTransformer
from .fast_observation_encoder import FastObservationEncoder
from .perceiver_resampler import PerceiverResampler
from .system1_registry import SYSTEM1_POLICIES


class System1Policy(nn.Module):
    """Fast policy: current frame + proprioception + System 2 latent -> action chunk.

    Pattern: Strategy. The policy kinds (``DualSystemConfig.policy``) are registered in
    :data:`SYSTEM1_POLICIES` and implemented in ``*_strategy.py``:

    * ``regression``: predict the chunk directly, masked L1 (DP-VLA, HiRT / RT-1).
    * ``diffusion``: DDPM with a cosine schedule (Ho et al., 2020; Nichol & Dhariwal, 2021) and
      DDIM sampling (LCB's 3D Diffuser Actor, RoboDual's DiT, DexVLA, OpenHelix). The network
      predicts the velocity ``v = sqrt(a) eps - sqrt(1 - a) x0`` (Salimans & Ho, 2022) rather
      than the noise: recovering ``x0 = sqrt(a) x_t - sqrt(1 - a) v`` needs no division by
      ``sqrt(a)``, so the first, near-pure-noise sampling step cannot amplify the prediction
      error (with noise prediction it is ~300x at ``a = 1e-5``), whatever the action scale.
    * ``flow_matching``: rectified flow, ``x_t = (1 - t) x_0 + t x_1``, velocity ``x_1 - x_0``
      (Lipman et al., 2023; Liu et al., 2023), Euler sampling (pi0).

    Args:
        config: dual-system settings.
        latent_dim: width of the projected latent tokens (= ``config.system1_dim``).
        img_size: camera resolution.
        state_dim: proprioception size.
        chunk: actions per chunk.
        action_dim: action size.
    """

    def __init__(self, config: DualSystemConfig, img_size: int, state_dim: int, chunk: int, action_dim: int) -> None:
        super().__init__()
        d = config.system1_dim
        self.config = config
        self.policy = config.policy
        self.chunk, self.action_dim = chunk, action_dim
        self.obs_encoder = FastObservationEncoder(d, img_size)
        self.resampler = (PerceiverResampler(d, config.obs_resampler_tokens, config.system1_heads)
                          if config.obs_resampler_tokens else None)
        self.proprio = nn.Sequential(nn.Linear(state_dim, d), nn.GELU(), nn.Linear(d, d))
        self.type_embed = nn.Parameter(torch.zeros(3, d))  # latent / observation / proprioception
        self.head = ActionTransformer(d, config.system1_layers, config.system1_heads, chunk, action_dim)
        self.strategy = SYSTEM1_POLICIES.get(config.policy)(config)

    @property
    def alpha_bar(self) -> torch.Tensor:
        """Cumulative noise schedule of the diffusion policy.

        Raises:
            AttributeError: for policies without a diffusion schedule.
        """
        try:
            return self.strategy.alpha_bar
        except AttributeError as err:
            raise AttributeError(f"policy {self.policy!r} has no alpha_bar") from err

    # ------------------------------------------------------------------ conditioning
    def condition(self, latent: torch.Tensor, latent_mask: torch.Tensor, image: torch.Tensor,
                  state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``[latent | observation | proprioception]`` tokens and their mask.

        Args:
            latent: projected System 2 latent ``(B, K, D)``.
            latent_mask: its mask ``(B, K)``.
            image: current camera frame ``(B, 3, H, W)``.
            state: proprioception ``(B, state_dim)``.

        Returns:
            ``(tokens (B, K + N + 1, D), mask)``.
        """
        obs = self.obs_encoder(image)
        if self.resampler is not None:
            obs = self.resampler(obs)
        prop = self.proprio(state).unsqueeze(1)
        tokens = torch.cat([latent + self.type_embed[0], obs + self.type_embed[1], prop + self.type_embed[2]], 1)
        ones = torch.ones(obs.size(0), obs.size(1) + 1, device=obs.device, dtype=latent_mask.dtype)
        return tokens, torch.cat([latent_mask, ones], 1)

    # ------------------------------------------------------------------ training
    def loss(self, cond: torch.Tensor, cond_mask: torch.Tensor, actions: torch.Tensor,
             action_mask: torch.Tensor) -> torch.Tensor:
        """Policy loss on a ``(B, chunk, action_dim)`` target with ``(B, chunk)`` validity mask.

        Args:
            cond: conditioning tokens from :meth:`condition`.
            cond_mask: their mask.
            actions: target chunk.
            action_mask: validity of each step.

        Returns:
            A scalar loss.
        """
        weight = action_mask.unsqueeze(-1).to(actions.dtype)
        return self.strategy.loss(self.head, cond, cond_mask, actions, weight)

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def sample(self, cond: torch.Tensor, cond_mask: torch.Tensor, steps: int | None = None) -> torch.Tensor:
        """An action chunk ``(B, chunk, action_dim)``.

        Args:
            cond: conditioning tokens from :meth:`condition`.
            cond_mask: their mask.
            steps: sampling steps (default ``config.sample_steps``).
        """
        return self.strategy.sample(self.head, cond, cond_mask, self.chunk, self.action_dim,
                                    steps or self.config.sample_steps)
