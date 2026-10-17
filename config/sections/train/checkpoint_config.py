"""CheckpointConfig: checkpointConfig."""

from __future__ import annotations

from config.sections.base import VLASection


class CheckpointConfig(VLASection):
    ARGS = {"dir": "ckpt_dir", "every_n_steps": "save_every", "keep_last": "keep_ckpts"}

    dir: str = "checkpoints"
    every_n_steps: int = 100
    keep_last: int = 3                     # 0 = keep all
