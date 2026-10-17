"""FlowActionDecoder: predicts the vector field v_theta(x_t, t, condition)."""

import torch
import torch.nn as nn

from .sinusoidal_time_embedding import SinusoidalTimeEmbedding


class FlowActionDecoder(nn.Module):
    """
    Predicts the vector field v_theta(x_t, t, condition).
    Inputs:
        x_t : (batch, action_chunk_size)   noisy action chunk at time t (flattened)
        t   : (batch,) or scalar           current time in [0,1]
        cond: (batch, obs_dim)             observation / condition
    Output:
        v   : (batch, action_chunk_size)   predicted vector field
    """
    def __init__(self, action_dim_flat, obs_dim, hidden_dim=1024, time_embed_dim=128):
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
            nn.Linear(hidden_dim, action_dim_flat)
        )

    def forward(self, x_t, t, cond):
        # Sinusoidal embedding of t: (batch,) → (batch, time_embed_dim)
        t_emb = self.time_embed(t)
        # Concatenate along feature dimension
        inp = torch.cat([x_t, t_emb, cond], dim=1)  # (batch, action_dim_flat+time_embed_dim+obs_dim)
        return self.net(inp)
