"""FlowActionDecoder: predicts the vector field v_theta(x_t, t, condition)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .sinusoidal_time_embedding import SinusoidalTimeEmbedding


class FlowActionDecoder(nn.Module):
    """Predicts the vector field v_theta(x_t, t, condition) of a conditional flow-matching policy.

    Args:
        action_dim_flat: flattened action-chunk size.
        obs_dim: width of the observation / condition vector.
        hidden_dim: MLP hidden width.
        time_embed_dim: width of the sinusoidal time embedding.
    """

    def __init__(
        self, action_dim_flat: int, obs_dim: int, hidden_dim: int = 1024, time_embed_dim: int = 128
    ) -> None:
        super().__init__()
        self.action_dim_flat = action_dim_flat
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

        # Sinusoidal time embedding: scalar t → (batch, time_embed_dim)
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

        # Input: concatenate [x_t, time_emb, cond] → total dimension
        self.net = nn.Sequential(
            nn.Linear(action_dim_flat + time_embed_dim + obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim_flat),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Predict the vector field.

        Args:
            x_t: noisy flattened action chunk ``(batch, action_dim_flat)``.
            t: time in ``[0, 1]``, ``(batch,)`` or scalar.
            cond: observation condition ``(batch, obs_dim)``.

        Returns:
            ``(batch, action_dim_flat)``.
        """
        # Sinusoidal embedding of t: (batch,) → (batch, time_embed_dim)
        t_emb = self.time_embed(t)
        # Concatenate along feature dimension
        inp = torch.cat(
            [x_t, t_emb, cond], dim=1
        )  # (batch, action_dim_flat+time_embed_dim+obs_dim)
        return self.net(inp)
