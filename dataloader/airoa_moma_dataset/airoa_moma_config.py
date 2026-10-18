"""AiroaMomaConfig: hyper-parameters for MoMa → Halo-VLA batches."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AiroaMomaConfig:
    """Hyper-parameters for MoMa → Halo-VLA batches."""

    data_root: str  # directory containing episodes.jsonl and videos/

    tokenizer_name: str = "HuggingFaceTB/cosmo2-tokenizer"
    img_size: int = 224
    img_mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    img_std: tuple[float, ...] = (0.229, 0.224, 0.225)

    max_seq_len: int = 512
    max_action_len: int = 256  # subsample / truncate trajectory steps
    action_dim: int = 32
    state_dim: int = 32

    # Video sampling: num_sample_frames context frames + num_predict_frames future targets
    camera: str = "head"  # "hand" | "head"
    num_sample_frames: int = 8
    num_predict_frames: int = 1  # future frames returned as "future_frames" in batch
    frame_stride: int = 14  # consecutive windows: t, t+stride, ...
    min_episode_length: int = 32  # skip shorter episodes (metadata or probe)

    # Path templates relative to data_root
    episodes_jsonl: str = "episodes.jsonl"
    # chunk index from episode_index (dataset uses chunk-000, chunk-001, …)
    episodes_per_chunk: int = 1000
    video_rel_template: str = (
        "videos/observation.image.{camera}/chunk-{chunk:03d}/file-{file_num:03d}.mp4"
    )
    state_npy_subdir: str = "states"  # optional episode_XXXXXX.npy

    # Conversation template (ChatML-style, matches EO dataloader)
    system_prefix: str = "<|im_start|>system\n"
    human_prefix: str = "<|im_start|>user\n"
    assistant_prefix: str = "<|im_start|>assistant\n"
    turn_end: str = "<|im_end|>\n"

    max_samples: int | None = None
    seed: int = 42
