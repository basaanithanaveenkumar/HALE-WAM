"""DROIDConfig: dROIDConfig."""

from __future__ import annotations

from dataclasses import dataclass

# DROID action space: 7-DoF delta end-effector control + gripper
DROID_ACTION_DIM = 7


# DROID state: 7 joint positions + 7 Cartesian EE pose (xyz + rpy + gripper)
DROID_STATE_DIM = 14


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class DROIDConfig:
    """Hyper-parameters for DROID → Halo-VLA batches."""

    data_root: str  # local clone root

    tokenizer_name: str = "HuggingFaceTB/cosmo2-tokenizer"
    img_size: int = 224
    img_mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    img_std: tuple[float, ...] = (0.229, 0.224, 0.225)

    max_seq_len: int = 512
    max_action_len: int = 256
    action_dim: int = DROID_ACTION_DIM
    state_dim: int = DROID_STATE_DIM

    # Which camera view to use for the main image stream.
    # Must be a key present under videos/<camera_key>/
    camera: str = "observation.images.exterior_image_1_left"

    # Window sampling
    num_sample_frames: int = 5  # context frames fed as model input
    num_predict_frames: int = 5  # future frames returned as DiT targets
    frame_stride: int = 6  # temporal stride between sampled frames
    min_episode_length: int = 12  # skip very short episodes

    # Episodes per chunk directory (default matches lerobot droid_100)
    episodes_per_chunk: int = 1000

    # Conversation template (ChatML, matches other loaders)
    system_prefix: str = "<|im_start|>system\n"
    human_prefix: str = "<|im_start|>user\n"
    assistant_prefix: str = "<|im_start|>assistant\n"
    turn_end: str = "<|im_end|>\n"

    max_samples: int | None = None
    seed: int = 42

    # Internal — populated from meta/info.json
    fps: float = 15.0
