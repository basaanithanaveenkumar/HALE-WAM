"""EODatasetConfig: configuration for the EO-Dataset dataloader."""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class EODatasetConfig:
    """Configuration for the EO-Dataset dataloader."""

    dataset_name: str = "IPEC-COMMUNITY/EO-Data1.5M"
    subset: str = "interleave-temporal"
    split: str = "train"
    img_size: int = 224
    max_seq_len: int = 77
    max_action_len: int = 64
    action_dim: int = 32  # EO-Data uses 32-dim actions (varies by embodiment)
    state_dim: int = 32
    max_images: int = 8  # max images per sample for interleaved data
    tokenizer_name: str = "HuggingFaceTB/cosmo2-tokenizer"
    max_samples: int | None = None  # limit dataset size (None = all)
    streaming: bool = False
    cache_dir: str | None = None
    # Image normalisation (ImageNet defaults)
    img_mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    img_std: tuple[float, ...] = (0.229, 0.224, 0.225)
    # Conversation formatting
    system_prefix: str = "<|im_start|>system\n"
    human_prefix: str = "<|im_start|>user\n"
    assistant_prefix: str = "<|im_start|>assistant\n"
    turn_end: str = "<|im_end|>\n"
    # Number of future frames the DiT head predicts; one <halo_world_video> token
    # is inserted per frame so each frame gets its own transformer hidden state.
    num_predict_frames: int = 1
    # Special token IDs (set after tokenizer init)
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    image_token_id: int = 3
    action_token_id: int = 4
    state_token_id: int = 5
