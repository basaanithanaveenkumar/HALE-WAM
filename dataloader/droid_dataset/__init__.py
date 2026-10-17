"""PyTorch dataset for ``lerobot/droid_100`` (and the full DROID corpus) stored in
the **LeRobot v2 parquet + video format**.

DROID is a large-scale robot manipulation dataset collected on a Franka arm
with three cameras (two exterior, one wrist). The lerobot release stores
tabular data (actions, states, timestamps) in per-episode parquet files and
video frames in MP4 clips.

Expected local layout after ``huggingface-cli download lerobot/droid_100``::

    <data_root>/
      meta/
        info.json          # dataset-level metadata (fps, features, camera keys)
        episodes.parquet   # episode-level rows: episode_index, length, task_index
        tasks.parquet      # task_index → task string
      data/
        chunk-000/
          episode_000000.parquet   # per-frame rows: action, observation.state, …
          episode_000001.parquet
          …
      videos/
        chunk-000/
          observation.images.exterior_image_1_left/
            episode_000000.mp4
          observation.images.exterior_image_2_left/
            episode_000000.mp4
          observation.images.wrist_image_left/
            episode_000000.mp4
          …

Each training item is one **sliding window** of frames from one episode:
  - ``num_sample_frames`` context frames  → ``images`` [n, 3, H, W]
  - ``num_predict_frames`` future targets → ``future_frames`` [p, 3, H, W]
  - Per-frame actions from the parquet    → ``actions`` [T, action_dim]
  - Proprioceptive state                  → ``states``  [T, state_dim]

The output dict is fully compatible with ``eo_collate_fn`` and ``scripts/train.py``.

Reference: https://huggingface.co/datasets/lerobot/droid_100"""

from .common import DROID_CAMERA_KEYS, build_droid_dataloader  # noqa: F401
from .droid_config import DROID_ACTION_DIM, DROID_STATE_DIM, DROIDConfig  # noqa: F401
from .droid_dataset import DROIDDataset  # noqa: F401

__all__ = [
    "DROIDConfig",
    "DROIDDataset",
    "DROID_ACTION_DIM",
    "DROID_CAMERA_KEYS",
    "DROID_STATE_DIM",
    "build_droid_dataloader",
]
