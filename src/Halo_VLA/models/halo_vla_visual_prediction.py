"""Visual DiT world-model head of HaloVLM: frame-prediction loss and future-frame sampling."""

import torch
from exceptions import ModelInputError


class VisualPredictionMixin:
    """Visual DiT world-model head of HaloVLM: frame-prediction loss and future-frame sampling.

    Mixed into :class:`HaloVLM`; relies on the attributes its ``__init__`` sets.
    """

    # ------------------------------------------------------------------ #
    # Visual DiT — conditional flow matching on frames (see dit_frame_prediction)
    # ------------------------------------------------------------------ #
    def compute_visual_prediction_loss(
        self,
        images: torch.Tensor,
        image_mask: torch.Tensor | None = None,
        visual_context_emb: torch.Tensor | None = None,
        world_video_query_hiddens: torch.Tensor | None = None,
        future_frames: torch.Tensor | None = None,
        perceptual_weight: float = 0.1,
        ssim_weight: float = 0.1,
        temporal_weight: float = 0.05,
    ) -> tuple[torch.Tensor, dict]:
        """
        Train the DiT video-prediction head to predict frames beyond the observed context.

        Returns:
            (total_loss, sub_losses_dict) where sub_losses_dict contains
            loss_rgb_cfm, loss_rgb_perceptual, loss_rgb_ssim, loss_rgb_temporal.
        """
        zero = torch.tensor(0.0, device=images.device)
        empty: dict = {}

        if self.visual_predictor is None or future_frames is None:
            return zero, empty

        if visual_context_emb is None:
            raise ModelInputError(
                "visual_context_emb is required by compute_visual_prediction_loss."
            )

        B, P, _, H, W = future_frames.shape
        if world_video_query_hiddens is not None:
            n_wv = world_video_query_hiddens.size(1)
        else:
            n_wv = self.visual_predictor.num_predict_frames
        p_predict = min(n_wv, P, self.visual_predictor.num_predict_frames)
        future_targets = future_frames[:, :p_predict]

        # Last context frame = final frame of the observed window (images[:, -1] or
        # the last real frame when image_mask is provided).
        if image_mask is not None:
            last_idx = (image_mask.sum(dim=1) - 1).clamp(min=0).long()  # [B]
            last_ctx = images[torch.arange(B, device=images.device), last_idx]
        else:
            last_ctx = images[:, -1]  # [B, 3, H, W]

        out = self.visual_predictor.compute_loss(
            future_targets, visual_context_emb,
            num_frames=p_predict,
            world_video_query_hiddens=world_video_query_hiddens,
            perceptual_weight=perceptual_weight,
            ssim_weight=ssim_weight,
            temporal_weight=temporal_weight,
            last_context_frame=last_ctx,
        )
        return out["loss_rgb_total"], out

    @torch.no_grad()
    def predict_visual_future(
        self,
        past_rgb: torch.Tensor,
        visual_context_emb: torch.Tensor | None = None,
        num_frames: int = 3,
        world_video_query_hiddens: torch.Tensor | None = None,
        num_ode_steps: int = 20,
    ) -> dict[str, torch.Tensor]:
        """
        Sample future RGB frames from the DiT head autoregressively.

        Args:
            past_rgb: [B, T, 3, H, W] or [B, 3, H, W]
            visual_context_emb: [B, emb_dim] — pooled decoder hidden states (3rd return of
                ``forward``). Recomputed from ViT if None or step > 0.
            num_frames: number of future frames to generate (default: 3).
            world_video_query_hiddens: [B, n_wv, emb_dim] optional — 4th return of
                ``forward``; supplies per-frame DiT query conditioning.
            num_ode_steps: Euler integration steps (more → better quality, slower).

        Returns:
            dict with key ``"rgb"`` → [B, num_frames, 3, H, W], or {} if DiT disabled.
        """
        if self.visual_predictor is None:
            return {}

        if past_rgb.dim() == 4:
            past_rgb = past_rgb.unsqueeze(1)

        H, W = past_rgb.shape[-2], past_rgb.shape[-1]
        vp = self.visual_predictor
        current_past = past_rgb
        preds_rgb = []

        for step in range(num_frames):
            ctx_emb = (
                visual_context_emb
                if step == 0 and visual_context_emb is not None
                else self.encode_visual_context_from_past_vit(current_past)
            )

            wv_step = (
                world_video_query_hiddens[:, step : step + 1, :]
                if world_video_query_hiddens is not None
                and step < world_video_query_hiddens.size(1)
                else None
            )

            # The pixel-level context frame: last observed frame for step 0,
            # previously predicted frame for step > 0.
            context_frame = current_past[:, -1]  # [B, 3, H, W]

            # Returns [B, 1, C, H, W]; take frame 0
            pred_frame = vp.predict_future_frames(
                H, W, ctx_emb,
                num_frames=1,
                num_steps=num_ode_steps,
                world_video_query_hiddens=wv_step,
                frame_offset=step,
                context_frame=context_frame,
            )[:, 0]  # [B, 3, H, W]

            preds_rgb.append(pred_frame)
            current_past = torch.cat([current_past, pred_frame.unsqueeze(1)], dim=1)

        return {
            "rgb": torch.stack(preds_rgb, dim=1),  # [B, num_frames, 3, H, W]
        }
