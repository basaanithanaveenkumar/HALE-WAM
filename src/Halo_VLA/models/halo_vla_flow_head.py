"""Flow-matching action head of HaloVLM: training loss and Euler ODE sampling."""

from __future__ import annotations

import torch


class FlowActionHeadMixin:
    """Flow-matching action head of HaloVLM: training loss and Euler ODE sampling.

    Mixed into :class:`HaloVLM`; relies on the attributes its ``__init__`` sets.
    """

    # ------------------------------------------------------------------ #
    # Flow Matching — training loss
    #
    # Inspired by the flow matching formulation presented in:
    #   https://www.youtube.com/live/sW75XtmutfE?si=_ZKNQtiKPz4BxxZG
    #
    # The core idea follows the Conditional Flow Matching (CFM) framework,
    # but we added *conditioning on the transformer's hidden states* so
    # that the velocity field is aware of the full visual-language context
    # when predicting robot actions.
    # ------------------------------------------------------------------ #
    def compute_flow_loss(self, action_hiddens: torch.Tensor | None, target_actions: torch.Tensor,
                          action_mask_seq: torch.Tensor) -> torch.Tensor:
        """
        Conditional Flow Matching (CFM) loss for action prediction.

        Given transformer hidden states at <halo_action> positions (conditioning)
        and ground-truth target actions, this method:
          1. Flattens each action token's target chunk into x_1.
          2. Samples t ~ U(0,1) and Gaussian noise x_0 ~ N(0, I).
          3. Constructs the linear interpolant  x_t = (1-t)*x_0 + t*x_1.
          4. Predicts the velocity field  v_theta(x_t, t, cond).
          5. Returns MSE loss between v_theta and the true velocity (x_1 - x_0).

        Args:
            action_hiddens : [B, n_act, emb_dim]  — conditioning from transformer
            target_actions : [B, T, action_dim]    — ground-truth action sequence
            action_mask_seq: [B, T]                — 1 for real timesteps, 0 for pad
        Returns:
            loss : scalar tensor — flow matching MSE loss
        """
        if action_hiddens is None:
            return torch.tensor(0.0, device=target_actions.device)

        B, n_act, D = action_hiddens.shape
        device = action_hiddens.device
        chunk = self.action_chunk_size
        act_dim = self.action_dim
        flat_dim = chunk * act_dim  # total dims per action token

        # --- Build x_1: target actions reshaped to [B*n_act, flat_dim] ---
        T_avail = target_actions.size(1)
        T_need = n_act * chunk
        # Truncate or pad target actions to match n_act * chunk_size
        if T_avail >= T_need:
            targets = target_actions[:, :T_need, :]   # [B, T_need, act_dim]
        else:
            pad = torch.zeros(B, T_need - T_avail, act_dim, device=device)
            targets = torch.cat([target_actions, pad], dim=1)
        # Reshape: [B, n_act, chunk*act_dim]
        x_1 = targets.view(B, n_act, flat_dim).view(B * n_act, flat_dim)

        # --- Build loss mask from action_mask_seq: [B*n_act, flat_dim] ---
        # Truncate or pad mask to T_need timesteps, then expand to flat_dim.
        T_mask = action_mask_seq.size(1)
        if T_mask >= T_need:
            mask_seq = action_mask_seq[:, :T_need].to(device=device, dtype=x_1.dtype)
        else:
            mask_pad = torch.zeros(B, T_need - T_mask, device=device, dtype=x_1.dtype)
            mask_seq = torch.cat([action_mask_seq.to(device=device, dtype=x_1.dtype), mask_pad], dim=1)
        # [B, T_need] → [B, n_act, chunk] → [B*n_act, chunk] → [B*n_act, flat_dim]
        loss_mask = (
            mask_seq.view(B, n_act, chunk)
                    .reshape(B * n_act, chunk)
                    .unsqueeze(-1)
                    .expand(-1, -1, act_dim)
                    .reshape(B * n_act, flat_dim)
        )

        # --- Condition vector: flatten to [B*n_act, emb_dim] ---
        # Each <halo_action> token's transformer hidden state becomes an
        # independent conditioning vector for the flow network.  By reshaping
        # [B, n_act, D] → [B*n_act, D] we treat every action-token position
        # as its own sample, so the flow decoder sees one (x_t, t, cond)
        # triple at a time.
        cond = action_hiddens.view(B * n_act, D)

        # --- Sample t ~ Uniform(0, 1) per element ---
        # Each of the B*n_act samples gets its own random "time" along the
        # flow path.  t=0 corresponds to pure noise, t=1 to clean data.
        # Drawing independent t values per sample provides diverse gradient
        # signal across the full [0,1] interval in a single batch.
        t = torch.rand(B * n_act, device=device)

        # --- Sample noise x_0 ~ N(0, I) ---
        # x_0 is the source distribution (isotropic Gaussian).  At inference
        # time we will also start from this distribution and integrate the
        # learned velocity field forward to t=1 to recover clean actions.
        x_0 = torch.randn_like(x_1)

        # --- Optimal-transport linear interpolant: x_t = (1-t)*x_0 + t*x_1 ---
        # This is the *conditional* OT path between x_0 (noise) and x_1 (data).
        # At t=0 we are at pure noise x_0; at t=1 we arrive at the ground-truth
        # action x_1.  The straight-line interpolation is the optimal transport
        # displacement map for Gaussian → point-mass, and its derivative w.r.t. t
        # is the constant velocity  v* = x_1 - x_0.
        t_expand = t.unsqueeze(-1)                     # [B*n_act, 1]
        x_t = (1.0 - t_expand) * x_0 + t_expand * x_1  # [B*n_act, flat_dim]

        # --- Predict velocity field v_theta(x_t, t, cond) ---
        # The flow decoder receives the noisy action x_t, the scalar time t
        # (embedded via sinusoidal frequencies inside the decoder), and the
        # transformer conditioning vector.  It outputs a velocity vector of
        # the same dimensionality as x_t, representing its estimate of the
        # direction and magnitude needed to move from noise toward data.
        v_pred = self.flow_decoder(x_t, t, cond)       # [B*n_act, flat_dim]

        # --- Ground-truth velocity: dx/dt = x_1 - x_0 (constant for OT path) ---
        # Because the interpolant is linear in t, its time-derivative is just
        # the displacement vector (x_1 - x_0), constant for all t.  This is
        # what we supervise the network to predict.
        v_target = x_1 - x_0

        # --- Masked MSE loss ---
        # Exclude padded action timesteps (loss_mask == 0) so padding does not
        # corrupt the velocity-field gradients.
        denom = loss_mask.sum().clamp(min=1.0)
        loss = ((v_pred - v_target).pow(2) * loss_mask).sum() / denom
        return loss

    # ------------------------------------------------------------------ #
    # Flow Matching — Euler ODE sampling (inference)
    #
    # During training we taught the flow decoder to predict the velocity
    # field that moves probability mass from Gaussian noise toward the
    # real action distribution.  At inference we *reverse-engineer* the
    # actual actions by starting from noise and following that learned
    # velocity field forward in time, like tracing a river from its
    # source to its mouth.
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample_actions(self, action_hiddens: torch.Tensor | None, num_steps: int | None = None) -> torch.Tensor | None:
        """
        Generate actions by integrating the learned velocity field from t=0 to t=1
        using the Euler method.

        Starting from Gaussian noise x_0 ~ N(0, I), we iteratively update:
            x_{t+dt} = x_t + v_theta(x_t, t, cond) * dt
        until we reach t=1, which gives the predicted clean action chunk.

        Args:
            action_hiddens : [B, n_act, emb_dim] — conditioning from transformer
            num_steps      : int — number of Euler integration steps (default
                             from config.flow_num_ode_steps)
        Returns:
            action_preds : [B, n_act, chunk_size, action_dim]
        """
        # If the transformer didn't produce any <halo_action> hidden states
        # (e.g. the prompt has no action tokens), there's nothing to decode.
        if action_hiddens is None:
            return None

        # Fall back to the default number of ODE integration steps from
        # config if the caller didn't specify one.  More steps = more
        # accurate but slower; 20 is a good default for Euler.
        if num_steps is None:
            num_steps = self.flow_num_ode_steps

        B, n_act, D = action_hiddens.shape
        device = action_hiddens.device
        # flat_dim is the total number of scalars in one action chunk,
        # e.g. 16 timesteps * 7 DOF = 112.  The flow network works in
        # this flattened space and we reshape back at the end.
        flat_dim = self.action_chunk_size * self.action_dim

        # Each <halo_action> token position is treated as an independent
        # sample.  We flatten [B, n_act, emb_dim] → [B*n_act, emb_dim]
        # so every action-token gets its own conditioning vector — the
        # same way we handled it during training.
        cond = action_hiddens.view(B * n_act, D)

        # This is where generation begins: pure random noise, the same
        # distribution we used as x_0 during training.  Think of it as
        # a blank canvas that the velocity field will sculpt into a
        # meaningful action trajectory over the next num_steps iterations.
        x = torch.randn(B * n_act, flat_dim, device=device)

        # We integrate the ODE  dx/dt = v_theta(x, t, cond)  from t=0
        # to t=1 using the simplest possible solver: forward Euler.
        # dt is the time increment per step — with 20 steps, each step
        # advances time by 0.05.
        dt = 1.0 / num_steps
        for step_i in range(num_steps):
            # Current time along the flow.  At step 0 we're at t=0
            # (pure noise); by the last step we're near t=1 (clean data).
            t_val = step_i * dt
            # Broadcast the scalar time to every sample in the batch
            t = torch.full((B * n_act,), t_val, device=device)

            # Ask the flow decoder: "given this partially-denoised action
            # x at time t, and this transformer context, which direction
            # and how far should I move?"  The sinusoidal time embedding
            # inside the decoder tells it *where* it is on the noise→data
            # continuum, so it knows how aggressively to push.
            v = self.flow_decoder(x, t, cond)

            # Take one Euler step: move x a little bit in the predicted
            # velocity direction.  After all num_steps of these small
            # nudges, x will have traveled from noise to (approximately)
            # the clean action that the model thinks the robot should take.
            x = x + v * dt

        # The integration is done — x is now our best estimate of the
        # clean action chunk in flattened form [B*n_act, flat_dim].
        # Reshape it back to the structured [B, n_act, chunk_size, action_dim]
        # so downstream code can interpret each timestep × DOF independently.
        action_preds = x.view(B, n_act, self.action_chunk_size, self.action_dim)
        return action_preds
