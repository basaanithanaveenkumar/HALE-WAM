"""System 1: the fast, reactive action policy of a dual-system VLA."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dual_system.config import DualSystemConfig


class FastObservationEncoder(nn.Module):
    """A light convolutional encoder for the *current* camera frame (runs at control rate).

    Three strided convolutions reduce ``(B, 3, H, W)`` to an ``H/16 x W/16`` grid of ``dim``-wide
    tokens with learned 2D positions - cheap enough to run every control step, unlike System 2.
    """

    def __init__(self, dim: int, img_size: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=4), nn.GELU(),
            nn.Conv2d(32, 64, 2, stride=2), nn.GELU(),
            nn.Conv2d(64, dim, 2, stride=2),
        )
        grid = max(1, img_size // 16)
        self.pos = nn.Parameter(torch.zeros(1, grid * grid, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        tokens = self.stem(image).flatten(2).transpose(1, 2)  # (B, N, dim)
        return tokens + self.pos[:, : tokens.size(1)]


class PerceiverResampler(nn.Module):
    """Compress a variable token set to ``num_latents`` learned queries (Alayrac et al., 2022).

    RoboDual uses it to distil the observation features before the diffusion policy.
    """

    def __init__(self, dim: int, num_latents: int, heads: int) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_latents, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        q = self.latents.expand(tokens.size(0), -1, -1)
        x = q + self.attn(q, self.norm(tokens), self.norm(tokens))[0]
        return x + self.ff(x)


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of a continuous / integer timestep ``t`` (``(B,)`` -> ``(B, dim)``)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return F.pad(emb, (0, dim % 2))


class ActionTransformer(nn.Module):
    """Action-chunk tokens that self-attend and cross-attend to the conditioning tokens.

    Shared by all three policy heads: for ``regression`` the input tokens are learned queries;
    for ``diffusion`` / ``flow_matching`` they are the embedded noisy actions plus a timestep
    embedding. Cross-attention to ``[latent | observation | proprioception]`` tokens is how the
    slow latent and the fast observation are fused (as in RoboDual's DiT and 3D Diffuser Actor).
    """

    def __init__(self, dim: int, layers: int, heads: int, chunk: int, action_dim: int) -> None:
        super().__init__()
        self.action_in = nn.Linear(action_dim, dim)
        self.queries = nn.Parameter(torch.randn(1, chunk, dim) * 0.02)
        self.time = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.pos = nn.Parameter(torch.zeros(1, chunk, dim))
        layer = nn.TransformerDecoderLayer(dim, heads, 4 * dim, dropout=0.0, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers)
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, action_dim))
        self.dim = dim

    def forward(self, cond: torch.Tensor, cond_mask: torch.Tensor, noisy: torch.Tensor | None = None,
                t: torch.Tensor | None = None) -> torch.Tensor:
        b = cond.size(0)
        x = self.queries.expand(b, -1, -1) if noisy is None else self.action_in(noisy)
        if t is not None:
            x = x + self.time(timestep_embedding(t, self.dim)).unsqueeze(1)
        x = x + self.pos
        return self.out(self.decoder(x, cond, memory_key_padding_mask=cond_mask == 0))


class System1Policy(nn.Module):
    """Fast policy: current frame + proprioception + System 2 latent -> action chunk.

    Heads (``DualSystemConfig.policy``):

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
        if self.policy == "diffusion":
            steps = config.diffusion_train_steps
            s = 0.008  # cosine schedule (Nichol & Dhariwal, 2021)
            x = torch.linspace(0, steps, steps + 1) / steps
            alpha_bar = torch.cos((x + s) / (1 + s) * math.pi / 2) ** 2
            alpha_bar = (alpha_bar / alpha_bar[0]).clamp(1e-5, 1.0)
            self.register_buffer("alpha_bar", alpha_bar[1:], persistent=False)

    # ------------------------------------------------------------------ conditioning
    def condition(self, latent: torch.Tensor, latent_mask: torch.Tensor, image: torch.Tensor,
                  state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``[latent | observation | proprioception]`` tokens and their mask."""
        obs = self.obs_encoder(image)
        if self.resampler is not None:
            obs = self.resampler(obs)
        prop = self.proprio(state).unsqueeze(1)
        tokens = torch.cat([latent + self.type_embed[0], obs + self.type_embed[1], prop + self.type_embed[2]], 1)
        ones = torch.ones(obs.size(0), obs.size(1) + 1, device=obs.device, dtype=latent_mask.dtype)
        return tokens, torch.cat([latent_mask, ones], 1)

    # ------------------------------------------------------------------ training
    def loss(self, cond, cond_mask, actions: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        """Policy loss on a ``(B, chunk, action_dim)`` target with ``(B, chunk)`` validity mask."""
        b = actions.size(0)
        weight = action_mask.unsqueeze(-1).to(actions.dtype)
        if self.policy == "regression":
            pred = self.head(cond, cond_mask)
            return ((pred - actions).abs() * weight).sum() / (weight.sum() * self.action_dim).clamp_min(1.0)
        noise = torch.randn_like(actions)
        if self.policy == "diffusion":
            k = torch.randint(0, self.alpha_bar.numel(), (b,), device=actions.device)
            ab = self.alpha_bar[k].view(b, 1, 1)
            noisy = ab.sqrt() * actions + (1 - ab).sqrt() * noise
            target = ab.sqrt() * noise - (1 - ab).sqrt() * actions  # v-prediction
            pred = self.head(cond, cond_mask, noisy, k)
        else:  # flow matching
            t = torch.rand(b, device=actions.device)
            noisy = (1 - t.view(b, 1, 1)) * noise + t.view(b, 1, 1) * actions
            target = actions - noise
            pred = self.head(cond, cond_mask, noisy, t * 1000)
        return (((pred - target) ** 2) * weight).sum() / (weight.sum() * self.action_dim).clamp_min(1.0)

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def sample(self, cond: torch.Tensor, cond_mask: torch.Tensor, steps: int | None = None) -> torch.Tensor:
        """An action chunk ``(B, chunk, action_dim)``."""
        if self.policy == "regression":
            return self.head(cond, cond_mask)
        steps = steps or self.config.sample_steps
        b = cond.size(0)
        x = torch.randn(b, self.chunk, self.action_dim, device=cond.device)
        if self.policy == "flow_matching":
            for i in range(steps):
                t = torch.full((b,), i / steps, device=cond.device)
                x = x + self.head(cond, cond_mask, x, t * 1000) / steps
            return x
        # DDIM (eta = 0) over an evenly spaced subset of the training noise levels
        levels = torch.linspace(self.alpha_bar.numel() - 1, 0, steps, device=cond.device).round().long()
        for i, k in enumerate(levels):
            ab = self.alpha_bar[k]
            v = self.head(cond, cond_mask, x, k.expand(b))
            x0 = ab.sqrt() * x - (1 - ab).sqrt() * v
            eps = (1 - ab).sqrt() * x + ab.sqrt() * v
            ab_prev = self.alpha_bar[levels[i + 1]] if i + 1 < len(levels) else torch.ones((), device=x.device)
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        return x
