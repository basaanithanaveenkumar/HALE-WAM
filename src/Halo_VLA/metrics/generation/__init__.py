"""Video and image generation quality metrics for HALE-WAM.

HALE-WAM uses a DiT-based visual predictor (``dit_frame_prediction.py``) to
generate future frames from robot observations.  These metrics evaluate the
quality of that generation.

Metrics
-------
PSNRMetric    — pixel-level fidelity (in dB); quick and deterministic.
SSIMMetric    — structural/perceptual similarity; better than PSNR for blur.
FVDProxyMetric— lightweight proxy for Fréchet Video Distance; captures
               temporal distribution mismatch without the I3D network."""

from .fvd_proxy_metric import FVDProxyMetric  # noqa: F401
from .psnr_metric import PSNRMetric  # noqa: F401
from .ssim_metric import SSIMMetric, _gaussian_kernel, _ssim_map  # noqa: F401

__all__ = [
    "FVDProxyMetric",
    "PSNRMetric",
    "SSIMMetric",
]
