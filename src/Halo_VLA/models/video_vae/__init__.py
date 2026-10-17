"""Video VAE: SD-style convolutional autoencoder with 8× spatial compression.

Compresses RGB frames [B, 3, H, W] → latents [B, 4, H/8, W/8].
Architecture mirrors Stable Diffusion VAE (f=8):

  Encoder  channels [128, 256, 512, 512]  3 × 2× strided-conv downsampling
  Decoder  channels [512, 512, 256, 128]  3 × nearest+conv upsampling
  Bottleneck (at 1/8 resolution): ResBlock → Self-Attention → ResBlock
  Latent: 4-channel diagonal Gaussian (encoder outputs 8 ch = mean ‖ logvar)

Latent scaling: latents are multiplied by SCALE_FACTOR = 0.18215 (SD
convention) so the DiT sees approximately unit-variance inputs.

Why VAE + DiT instead of pixel-space DiT?
  At 224×224 with patch_size=4 the pixel-space DiT sees 56×56 = 3 136 tokens.
  After 8× VAE compression the latent is 28×28 with patch_size=4 → 49 tokens.
  Same compute budget → 64× fewer attention operations → higher quality.

Training losses (returned by VideoVAE.forward):
  L_recon  MSE( decode(encode(x)), x )      pixel reconstruction
  L_kl     KL( q(z|x) || N(0,I) )           posterior regularisation

Usage:
    vae = VideoVAE()

    # Training
    out = vae(frame)                          # VAEOutput
    loss = F.mse_loss(out.x_recon, frame) + 1e-4 * out.kl_loss

    # Encode for DiT (returns scaled latent)
    z = vae.encode(frame)                     # [B, 4, H/8, W/8]

    # Decode DiT output back to pixels
    frame = vae.decode(z)                     # [B, 3, H, W]"""

from .attention_block import AttentionBlock  # noqa: F401
from .decoder_block import DecoderBlock  # noqa: F401
from .downsample import Downsample  # noqa: F401
from .encoder_block import EncoderBlock  # noqa: F401
from .mid_block import MidBlock  # noqa: F401
from .res_block import ResBlock  # noqa: F401
from .upsample import Upsample  # noqa: F401
from .vae_decoder import VAEDecoder  # noqa: F401
from .vae_encoder import VAEEncoder  # noqa: F401
from .vae_output import VAEOutput  # noqa: F401
from .video_vae import VideoVAE  # noqa: F401

__all__ = [
    "AttentionBlock",
    "DecoderBlock",
    "Downsample",
    "EncoderBlock",
    "MidBlock",
    "ResBlock",
    "Upsample",
    "VAEDecoder",
    "VAEEncoder",
    "VAEOutput",
    "VideoVAE",
]
