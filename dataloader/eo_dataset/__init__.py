"""EO-Data1.5M Dataloader for Halo-VLA.

Supports the IPEC-COMMUNITY/EO-Data1.5M dataset from HuggingFace,
including all 17 subsets (5 interleaved + 12 QA).

Dataset schema per sample:
    source       : str   — source dataset identifier
    conversation : list  — multi-turn dicts [{"from": ..., "value": ...}, ...]
    image        : PIL.Image or list[PIL.Image] — visual observations
    action       : list/ndarray or None — robot action chunks (continuous)
    state        : list/ndarray or None — robot proprioceptive state

Reference: https://huggingface.co/datasets/IPEC-COMMUNITY/EO-Data1.5M"""

from .common import (  # noqa: F401
    ALL_SUBSETS,
    INTERLEAVE_SUBSETS,
    QA_SUBSETS,
    build_eo_dataloader,
    eo_collate_fn,
)
from .eo_dataset import (  # noqa: F401
    ACTION_TOKEN,
    IMAGE_TOKEN,
    STATE_TOKEN,
    WORLD_VIDEO_TOKEN,
    EODataset,
    build_image_transform,
)
from .eo_dataset_config import EODatasetConfig  # noqa: F401

__all__ = [
    "ACTION_TOKEN",
    "ALL_SUBSETS",
    "EODataset",
    "EODatasetConfig",
    "IMAGE_TOKEN",
    "INTERLEAVE_SUBSETS",
    "QA_SUBSETS",
    "STATE_TOKEN",
    "WORLD_VIDEO_TOKEN",
    "build_eo_dataloader",
    "build_image_transform",
    "eo_collate_fn",
]
