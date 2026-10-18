"""AIRoA MoMa metadata: episode parquet/jsonl, task map, video files and filtering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger


class MomaEpisodeIndexMixin:
    """AIRoA MoMa metadata: episode parquet/jsonl, task map, video files and filtering.

    Mixed into :class:`AiroaMomaDataset`; relies on the attributes its ``__init__`` sets.
    """

    def _load_episodes(self) -> list[dict[str, Any]]:
        """Return a list of episode dicts, trying sources in priority order."""
        # 1. Try lerobot-format parquet under meta/episodes/
        parquet_dir = self.root / "meta" / "episodes"
        if parquet_dir.is_dir():
            rows = self._load_episodes_parquet(parquet_dir)
            if rows:
                logger.info("Loaded {} episodes from meta/episodes/ parquet.", len(rows))
                return rows

        # 2. Fallback: scan video files from disk (pure stdlib, no deps)
        rows = self._scan_video_files()
        if rows:
            logger.info(
                "Loaded {} episodes by scanning video directory (no parquet found).",
                len(rows),
            )
            return rows

        # 3. Legacy: try episodes.jsonl
        jsonl_path = self.root / self.cfg.episodes_jsonl
        if jsonl_path.is_file():
            rows = self._load_episodes_jsonl(jsonl_path)
            # episodes.jsonl may be the global stats file (1 line); discard if so
            if rows and "episode_index" in rows[0]:
                logger.info("Loaded {} episodes from episodes.jsonl.", len(rows))
                return rows

        logger.warning(
            "No episode metadata found under {}. "
            "Expected meta/episodes/ parquet or video files at "
            "videos/observation.image.{}/chunk-*/file-*.mp4",
            self.root,
            self.cfg.camera,
        )
        return []

    def _load_episodes_parquet(self, parquet_dir: Path) -> list[dict[str, Any]]:
        """Read all parquet files under meta/episodes/ using pyarrow (no pandas)."""
        try:
            import pyarrow.parquet as pq  # type: ignore
        except ImportError:
            logger.warning("pyarrow not available; skipping parquet load.")
            return []

        # Load optional task index → task string mapping
        task_map: dict[int, str] = self._load_task_map()

        rows: list[dict[str, Any]] = []
        for pq_file in sorted(parquet_dir.rglob("*.parquet")):
            try:
                table = pq.read_table(pq_file)
            except Exception as exc:
                logger.warning("Could not read {}: {}", pq_file, exc)
                continue
            names = table.schema.names
            for i in range(table.num_rows):
                rec: dict[str, Any] = {}
                for col in names:
                    val = table[col][i].as_py()
                    rec[col] = val
                # Normalise episode_index field name
                if "episode_index" not in rec:
                    # lerobot v2 may call it 'index'
                    rec["episode_index"] = rec.get("index", i)
                # Attach video_rel_path so _video_path() can resolve it
                ep_idx = int(rec["episode_index"])
                chunk = ep_idx // self.cfg.episodes_per_chunk
                file_num = ep_idx % self.cfg.episodes_per_chunk
                rec["video_rel_path"] = (
                    f"videos/observation.image.{self.cfg.camera}"
                    f"/chunk-{chunk:03d}/file-{file_num:03d}.mp4"
                )
                # Attach task text
                task_idx = rec.get("task_index", None)
                if task_idx is not None and task_idx in task_map:
                    rec["tasks"] = [task_map[int(task_idx)]]
                elif "tasks" not in rec:
                    rec["tasks"] = [rec.get("short_horizon_task", "manipulation")]
                rows.append(rec)

        rows.sort(key=lambda r: int(r.get("episode_index", 0)))
        return rows

    def _load_task_map(self) -> dict[int, str]:
        """Load meta/tasks.parquet → {task_index: task_string}. Best-effort."""
        tasks_pq = self.root / "meta" / "tasks.parquet"
        if not tasks_pq.is_file():
            return {}
        try:
            import pyarrow.parquet as pq  # type: ignore

            table = pq.read_table(tasks_pq)
            task_map: dict[int, str] = {}
            names = table.schema.names
            idx_col = "task_index" if "task_index" in names else (names[0] if names else None)
            txt_col = (
                "task"
                if "task" in names
                else "short_horizon_task"
                if "short_horizon_task" in names
                else (names[1] if len(names) > 1 else None)
            )
            if idx_col and txt_col:
                for i in range(table.num_rows):
                    k = table[idx_col][i].as_py()
                    v = table[txt_col][i].as_py()
                    if k is not None:
                        task_map[int(k)] = str(v)
            return task_map
        except Exception as exc:
            logger.warning("Could not load tasks.parquet: {}", exc)
            return {}

    def _scan_video_files(self) -> list[dict[str, Any]]:
        """Pure-stdlib fallback: discover episodes from video files on disk."""
        import re

        videos_root = self.root / "videos" / f"observation.image.{self.cfg.camera}"
        if not videos_root.is_dir():
            return []
        rows: list[dict[str, Any]] = []
        for chunk_dir in sorted(videos_root.glob("chunk-*")):
            m = re.search(r"chunk-(\d+)", chunk_dir.name)
            if not m:
                continue
            chunk_idx = int(m.group(1))
            for mp4 in sorted(chunk_dir.glob("file-*.mp4")):
                fm = re.search(r"file-(\d+)", mp4.stem)
                if not fm:
                    continue
                file_num = int(fm.group(1))
                ep_idx = chunk_idx * self.cfg.episodes_per_chunk + file_num
                rows.append(
                    {
                        "episode_index": ep_idx,
                        "length": 0,
                        "tasks": ["manipulation"],
                        "video_rel_path": str(mp4.relative_to(self.root)),
                    }
                )
        rows.sort(key=lambda r: r["episode_index"])
        return rows

    def _load_episodes_jsonl(self, path: Path) -> list[dict[str, Any]]:
        """Legacy loader for episodes.jsonl."""
        rows: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        rows.sort(key=lambda r: int(r.get("episode_index", 0)))
        return rows

    def _filter_episodes(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ok = []
        for m in rows:
            ep = int(m.get("episode_index", -1))
            if ep < 0:
                continue
            vid = self._video_path(ep, meta=m)
            if vid.is_file():
                ok.append(m)
        if not ok:
            logger.warning(
                "No episodes found — video files not matched. "
                "Run scripts/generate_episodes_jsonl.py to rebuild episodes.jsonl."
            )
        else:
            logger.info(f"Loaded {len(ok)} episodes with verified video paths.")
        return ok

    def _video_path(self, episode_index: int, meta: dict[str, Any] | None = None) -> Path:
        # Prefer the explicit relative path written by generate_episodes_jsonl.py
        if meta is not None and meta.get("video_rel_path"):
            return self.root / meta["video_rel_path"]
        # Fallback: reconstruct from episode_index using the template
        chunk = int(episode_index) // self.cfg.episodes_per_chunk
        file_num = int(episode_index) % self.cfg.episodes_per_chunk  # 0–999 per chunk
        rel = self.cfg.video_rel_template.format(
            chunk=chunk,
            camera=self.cfg.camera,
            ep=int(episode_index),
            file_num=file_num,
        )
        return self.root / rel
