"""DROID (LeRobot v2) metadata: info.json, task map, episode list and file paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger


class DroidEpisodeIndexMixin:
    """DROID (LeRobot v2) metadata: info.json, task map, episode list and file paths.

    Mixed into :class:`DROIDDataset`; relies on the attributes its ``__init__`` sets.
    """

    def _load_meta(self) -> None:
        """Read meta/info.json, meta/episodes.parquet, meta/tasks.parquet."""
        meta_dir = self.root / "meta"

        # info.json — fps + feature list
        info_path = meta_dir / "info.json"
        if info_path.is_file():
            with open(info_path) as f:
                info = json.load(f)
            self.cfg.fps = float(info.get("fps", self.cfg.fps))
            # Discover camera keys from info if config key is not explicit
            cam_keys = info.get("camera_keys", []) or info.get("video_keys", [])
            if cam_keys and self.cfg.camera not in cam_keys:
                # Prefer the configured key; fall back to the first available
                logger.warning(
                    "Configured camera '{}' not in info.json camera_keys {}; "
                    "using first available.",
                    self.cfg.camera,
                    cam_keys,
                )
                self.cfg.camera = cam_keys[0]
        else:
            logger.warning("meta/info.json not found — using defaults (fps={}).", self.cfg.fps)

        # tasks.parquet — task_index → task string
        self._task_map: dict[int, str] = self._load_task_map(meta_dir)

        # episodes.parquet — episode_index, length, task_index
        self._episode_meta: list[dict[str, Any]] = self._load_episodes_parquet(meta_dir)

    def _load_task_map(self, meta_dir: Path) -> dict[int, str]:
        tasks_pq = meta_dir / "tasks.parquet"
        if not tasks_pq.is_file():
            return {}
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(tasks_pq)
            names = table.schema.names
            idx_col = next((c for c in ("task_index", "index") if c in names), None)
            txt_col = next(
                (c for c in ("task", "task_description", "short_horizon_task") if c in names),
                None,
            )
            if not idx_col or not txt_col:
                return {}
            return {
                int(table[idx_col][i].as_py()): str(table[txt_col][i].as_py())
                for i in range(table.num_rows)
            }
        except Exception as exc:
            logger.warning("Could not load tasks.parquet: {}", exc)
            return {}

    def _load_episodes_parquet(self, meta_dir: Path) -> list[dict[str, Any]]:
        eps_pq = meta_dir / "episodes.parquet"
        if not eps_pq.is_file():
            logger.warning("meta/episodes.parquet not found — will discover from data/.")
            return []
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(eps_pq)
            names = table.schema.names
            rows = []
            for i in range(table.num_rows):
                rec: dict[str, Any] = {col: table[col][i].as_py() for col in names}
                if "episode_index" not in rec:
                    rec["episode_index"] = rec.get("index", i)
                rows.append(rec)
            rows.sort(key=lambda r: int(r.get("episode_index", 0)))
            return rows
        except Exception as exc:
            logger.warning("Could not load episodes.parquet: {}", exc)
            return []

    def _build_episode_list(self) -> list[dict[str, Any]]:
        """
        Merge parquet metadata with filesystem discovery.

        Returns list of episode dicts each guaranteed to have:
          episode_index, length, task (str), data_path (Path), video_path (Path).
        """
        rows = list(self._episode_meta)

        # If parquet gave us nothing, scan the data/ directory
        if not rows:
            rows = self._scan_data_dir()

        episodes = []
        for rec in rows:
            ep_idx = int(rec.get("episode_index", 0))

            data_path = self._data_path(ep_idx)
            video_path = self._video_path(ep_idx)

            # Resolve task text
            task_idx = rec.get("task_index")
            task = self._task_map.get(int(task_idx), "") if task_idx is not None else ""
            if not task:
                task = str(rec.get("task", rec.get("short_horizon_task", "robot manipulation")))

            # Prefer parquet-reported length; fall back to zero (probed on first access)
            length = int(rec.get("length", 0) or 0)

            if not data_path.is_file() and not video_path.is_file():
                continue  # skip episodes with no data at all

            if length > 0 and length < self.cfg.min_episode_length:
                continue

            episodes.append(
                {
                    "episode_index": ep_idx,
                    "length": length,
                    "task": task,
                    "data_path": data_path,
                    "video_path": video_path,
                }
            )

        logger.info("DROIDDataset: {} usable episodes after filtering.", len(episodes))
        return episodes

    def _scan_data_dir(self) -> list[dict[str, Any]]:
        """Fallback: discover episodes by scanning data/chunk-*/ parquet files."""
        import re

        data_root = self.root / "data"
        if not data_root.is_dir():
            return []
        rows = []
        for ep_pq in sorted(data_root.rglob("episode_*.parquet")):
            m = re.search(r"episode_(\d+)", ep_pq.stem)
            if not m:
                continue
            ep_idx = int(m.group(1))
            rows.append({"episode_index": ep_idx, "length": 0, "task": ""})
        return rows

    def _data_path(self, episode_index: int) -> Path:
        """Path to the per-episode parquet file under data/chunk-XXX/."""
        chunk = episode_index // self.cfg.episodes_per_chunk
        return self.root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"

    def _video_path(self, episode_index: int) -> Path:
        """Path to the MP4 for the configured camera view."""
        chunk = episode_index // self.cfg.episodes_per_chunk
        return (
            self.root
            / "videos"
            / f"chunk-{chunk:03d}"
            / self.cfg.camera
            / f"episode_{episode_index:06d}.mp4"
        )
