"""VideoVAE: sD-style Video VAE — 8× spatial compression for robot video frames."""

from __future__ import annotations

import torch
import torch.nn as nn

from .vae_decoder import VAEDecoder
from .vae_encoder import VAEEncoder
from .vae_output import VAEOutput


class VideoVAE(nn.Module):
    """
    SD-style Video VAE — 8× spatial compression for robot video frames.

    Token count comparison for 224×224 frames with DiT patch_size=4:
      Pixel-space DiT : 56 × 56 = 3 136 tokens
      VAE latent DiT  : 28 × 28 = 784 pixels → 7 × 7 = 49 tokens   (64× fewer)

    The same DiT compute budget produces dramatically higher quality when
    the model diffuses in a compact latent space instead of raw pixels.

    Parameter count at base_ch=128 : ~66 M  (encoder ~30 M, decoder ~36 M)
    Use base_ch=64 for a lighter    : ~17 M  variant.

    Typical workflow
    ----------------
    Joint training (VAE + DiT trained together):

        # --- VAE loss ---
        vae_out  = vae(frame)
        vae_loss = F.mse_loss(vae_out.x_recon, frame) + kl_w * vae_out.kl_loss

        # --- DiT operates on stop-gradient latents for stability ---
        z_sg     = vae_out.z.detach()
        dit_loss = cfm(dit, z_sg, conditioning)

        loss = dit_loss + vae_loss

    Inference:
        z     = vae.encode(past_frame, sample=False)   # deterministic
        z_gen = euler_integrate(dit, conditioning)     # DiT generates latent
        frame = vae.decode(z_gen)                      # back to pixels
    """

    SCALE_FACTOR: float = 0.18215   # keeps latent std ≈ 1 (same as SD)
    LATENT_CH:    int   = 4

    def __init__(self, in_ch: int = 3, out_ch: int = 3, base_ch: int = 128) -> None:
        super().__init__()
        self.encoder = VAEEncoder(in_ch=in_ch,              latent_ch=self.LATENT_CH, base_ch=base_ch)
        self.decoder = VAEDecoder(latent_ch=self.LATENT_CH, out_ch=out_ch,            base_ch=base_ch)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _split(h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split 8-channel encoder output into mean and clamped logvar."""
        mean, logvar = h.chunk(2, dim=1)
        return mean, logvar.clamp(-30.0, 20.0)

    @staticmethod
    def reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = mean + std * ε,  ε ~ N(0,I)."""
        return mean + (0.5 * logvar).exp() * torch.randn_like(mean)

    @staticmethod
    def kl_divergence(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL( N(mean, exp(logvar)) ‖ N(0,I) ) averaged over all elements."""
        return (-0.5 * (1.0 + logvar - mean.pow(2) - logvar.exp())).mean()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def encode(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """
        Encode RGB pixels to scaled latents.

        Args:
            x:      [B, 3, H, W]   input frames in model normalisation range.
            sample: True  → sample  z ~ q(z|x)   (training / stochastic)
                    False → use posterior mean    (deterministic inference)
        Returns:
            [B, 4, H/8, W/8] scaled by SCALE_FACTOR.
        """
        h              = self.encoder(x)
        mean, logvar   = self._split(h)
        z              = self.reparameterize(mean, logvar) if sample else mean
        return z * self.SCALE_FACTOR

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode scaled latents back to RGB pixels.

        Args:
            z: [B, 4, H/8, W/8]  scaled by SCALE_FACTOR.
        Returns:
            [B, 3, H, W] reconstructed / generated frames.
        """
        return self.decoder(z / self.SCALE_FACTOR)

    def forward(self, x: torch.Tensor) -> VAEOutput:
        """
        Full encode → reparameterise → decode pass (for VAE training).

        Args:
            x: [B, 3, H, W] input frames.
        Returns:
            VAEOutput containing reconstruction, latent, and KL loss.
        """
        h              = self.encoder(x)
        mean, logvar   = self._split(h)
        z              = self.reparameterize(mean, logvar)
        x_recon        = self.decoder(z)
        return VAEOutput(
            x_recon = x_recon,
            z       = z * self.SCALE_FACTOR,
            mean    = mean,
            logvar  = logvar,
            kl_loss = self.kl_divergence(mean, logvar),
        )
