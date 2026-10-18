"""CheckpointConfig: where and how often checkpoints are written."""

from __future__ import annotations

from config.sections.base import VLASection


class CheckpointConfig(VLASection):
    """Checkpoint cadence.

    Attributes:
        dir: output directory.
        every_n_steps: epochs between checkpoints (``--save_every``).
        keep_last: newest checkpoints to keep (0 keeps all).
    """

    ARGS = {"dir": "ckpt_dir", "every_n_steps": "save_every", "keep_last": "keep_ckpts"}

    dir: str = "checkpoints"
    every_n_steps: int = 100
    keep_last: int = 3                     # 0 = keep all
