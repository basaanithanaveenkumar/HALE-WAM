"""PyTorch dataset for `airoa-org/airoa-moma` (HSR teleoperation, RGB video + metadata).

Designed for Halo-VLA training: **language + flow actions + multi-frame visuals**
(including future-frame / DiT losses when ``num_sample_frames >= 2``).

The public Hugging Face repo is gated and video-heavy; this loader targets a **local
clone** after ``git lfs pull`` (see the dataset card). Expected layout::

    <data_root>/
      episodes.jsonl
      videos/chunk-XXX/observation.image.hand/episode_YYYYYY.mp4
      videos/chunk-XXX/observation.image.head/episode_YYYYYY.mp4

Optional proprio (if you export arrays next to the clone)::

    <data_root>/states/episode_YYYYYY.npy   # float32 [T, state_dim_raw]

If ``states/*.npy`` is missing, states/actions are padded zeros with masks so the
language + vision branches still train; add exports to enable full action supervision.

Dataset card: https://huggingface.co/datasets/airoa-org/airoa-moma"""

from .airoa_moma_config import AiroaMomaConfig  # noqa: F401
from .airoa_moma_dataset import AiroaMomaDataset  # noqa: F401
from .common import _import_cv2, build_airoa_moma_dataloader  # noqa: F401

__all__ = [
    "AiroaMomaConfig",
    "AiroaMomaDataset",
    "build_airoa_moma_dataloader",
]
