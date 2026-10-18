"""DROIDDataset: one item = one temporal window from one DROID episode."""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset

from dataloader.droid_dataset.droid_episode_index import DroidEpisodeIndexMixin
from dataloader.droid_dataset.droid_prompt import DroidPromptMixin
from dataloader.eo_dataset import build_image_transform
from exceptions import DatasetNotFoundError

from .droid_config import DROIDConfig


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DROIDDataset(DroidPromptMixin, DroidEpisodeIndexMixin, Dataset):
    """
    One item = one temporal window from one DROID episode.

    ``images``        [n, 3, H, W]  — ``num_sample_frames`` context frames
    ``future_frames`` [p, 3, H, W]  — ``num_predict_frames`` ground-truth targets
    ``actions``       [T, action_dim]
    ``states``        [T, state_dim]
    """

    def __init__(
        self,
        config: DROIDConfig | None = None,
        tokenizer: Any = None,
        split: str = "train",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if config is None:
            config = DROIDConfig(**kwargs)
        self.cfg = config
        self.split = split
        self.root = Path(config.data_root).expanduser().resolve()
        if not self.root.is_dir():
            raise DatasetNotFoundError(f"data_root not found: {self.root}")

        self._setup_tokenizer(tokenizer)
        self.img_transform = build_image_transform(
            self.cfg.img_size, self.cfg.img_mean, self.cfg.img_std
        )

        self._load_meta()
        self.episodes = self._build_episode_list()

        if self.cfg.max_samples is not None:
            self.episodes = self.episodes[: self.cfg.max_samples]

        logger.info(
            "DROIDDataset: {} episodes  camera={}  frames={}/+{}  stride={}",
            len(self.episodes),
            self.cfg.camera,
            self.cfg.num_sample_frames,
            self.cfg.num_predict_frames,
            self.cfg.frame_stride,
        )

    # ------------------------------------------------------------------
    # Metadata loading
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Episode list construction
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Episode data loading
    # ------------------------------------------------------------------

    def _load_episode_parquet(self, data_path: Path) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Parse a per-episode parquet file.

        Returns:
            actions  : float32 [T, action_dim]
            states   : float32 [T, state_dim]
            length   : int  (number of frames / rows)
        """
        try:
            import pyarrow.parquet as pq
        except ImportError:
            logger.warning("pyarrow not available; returning dummy actions/states.")
            return (
                np.zeros((0, self.cfg.action_dim), np.float32),
                np.zeros((0, self.cfg.state_dim), np.float32),
                0,
            )

        if not data_path.is_file():
            return (
                np.zeros((0, self.cfg.action_dim), np.float32),
                np.zeros((0, self.cfg.state_dim), np.float32),
                0,
            )

        try:
            table = pq.read_table(data_path)
        except Exception as exc:
            logger.warning("Could not read {}: {}", data_path, exc)
            return (
                np.zeros((0, self.cfg.action_dim), np.float32),
                np.zeros((0, self.cfg.state_dim), np.float32),
                0,
            )

        T = table.num_rows
        names = set(table.schema.names)

        # ── Actions ─────────────────────────────────────────────────────────
        action_col = next(
            (c for c in ("action", "action.cartesian_velocity", "actions") if c in names),
            None,
        )
        if action_col:
            raw_act = table[action_col].to_pylist()
            acts = self._nested_to_array(raw_act, T)
        else:
            acts = np.zeros((T, 0), np.float32)

        D_a = self.cfg.action_dim
        if acts.shape[1] >= D_a:
            acts = acts[:, :D_a]
        else:
            pad = np.zeros((T, D_a - acts.shape[1]), np.float32)
            acts = np.concatenate([acts, pad], axis=1)

        # ── States ───────────────────────────────────────────────────────────
        state_col = next(
            (c for c in ("observation.state", "state", "observation.robot_state") if c in names),
            None,
        )
        if state_col:
            raw_st = table[state_col].to_pylist()
            sts = self._nested_to_array(raw_st, T)
        else:
            sts = np.zeros((T, 0), np.float32)

        D_s = self.cfg.state_dim
        if sts.shape[1] >= D_s:
            sts = sts[:, :D_s]
        else:
            pad = np.zeros((T, D_s - sts.shape[1]), np.float32)
            sts = np.concatenate([sts, pad], axis=1)

        return acts.astype(np.float32), sts.astype(np.float32), T

    @staticmethod
    def _nested_to_array(rows: list[Any], T: int) -> np.ndarray:
        """
        Convert a list of rows (each a float or list[float]) to float32 [T, D].
        Handles both scalar and vector action/state formats.
        """
        if not rows:
            return np.zeros((T, 1), np.float32)
        first = rows[0]
        if isinstance(first, (int, float)):
            return np.array(rows, dtype=np.float32).reshape(T, 1)
        try:
            arr = np.array(rows, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(T, 1)
            return arr
        except (ValueError, TypeError):
            # Ragged rows — pad each to max width
            maxd = max((len(r) if isinstance(r, (list, tuple)) else 1) for r in rows)
            out = np.zeros((T, maxd), np.float32)
            for i, r in enumerate(rows):
                if isinstance(r, (list, tuple)):
                    d = min(len(r), maxd)
                    out[i, :d] = r[:d]
                else:
                    out[i, 0] = float(r)
            return out

    # ------------------------------------------------------------------
    # Video frame reading (PyAV, same approach as AiroaMomaDataset)
    # ------------------------------------------------------------------

    def _get_video_length(self, video_path: Path) -> int:
        """Return frame count without full decode."""
        try:
            import av

            container = av.open(str(video_path))
            stream = container.streams.video[0]
            if stream.frames and stream.frames > 0:
                container.close()
                return int(stream.frames)
            if stream.duration and stream.average_rate:
                dur_s = float(stream.duration) * stream.time_base
                fps = float(stream.average_rate)
                if dur_s > 0 and fps > 0:
                    container.close()
                    return max(1, round(dur_s * fps))
            count = sum(1 for p in container.demux(stream) if p.size > 0)
            container.close()
            return max(count, 1)
        except Exception as exc:
            logger.warning("Could not get length of {}: {}", video_path, exc)
            return 0

    def _read_frames_at_indices(
        self, video_path: Path, indices: Sequence[int]
    ) -> list[torch.Tensor]:
        """Decode specific frame indices using PyAV with a sequential counter."""
        try:
            import av
        except ImportError as exc:
            from hale_core.exceptions import MissingDependencyError

            raise MissingDependencyError("av", "PyAV decodes DROID videos: pip install av") from exc

        blank = torch.zeros(3, self.cfg.img_size, self.cfg.img_size)
        if not video_path.is_file():
            return [blank.clone() for _ in indices]

        try:
            container = av.open(str(video_path))
        except Exception as exc:
            logger.warning("Cannot open {}: {}", video_path, exc)
            return [blank.clone() for _ in indices]

        stream = container.streams.video[0]
        required = set(int(i) for i in indices)
        frame_map: dict[int, torch.Tensor] = {}
        counter = 0

        for packet in container.demux(stream):
            for frame in packet.decode():
                if counter in required:
                    img = frame.to_image().convert("RGB")
                    frame_map[counter] = self.img_transform(img)
                    required.discard(counter)
                    if not required:
                        break
                counter += 1
            if not required:
                break

        container.close()

        return [frame_map.get(int(i), blank.clone()) for i in indices]

    def _dummy_frames(self, n: int) -> torch.Tensor:
        return torch.zeros(n, 3, self.cfg.img_size, self.cfg.img_size)

    # ------------------------------------------------------------------
    # Conversation / tokenisation
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Window sampling helpers
    # ------------------------------------------------------------------

    def _sample_window(self, T: int, idx: int) -> tuple[list[int], list[int]]:
        """
        Sample ``num_sample_frames + num_predict_frames`` indices with ``frame_stride``.

        Training: per-item seeded random start for diversity while remaining
        reproducible.  Val/test: always starts at 0.
        """
        n = self.cfg.num_sample_frames
        p = self.cfg.num_predict_frames
        total = n + p
        stride = self.cfg.frame_stride
        span = (total - 1) * stride + 1

        if T < span:
            # Shrink stride to fit
            stride = max(1, (T - 1) // max(total - 1, 1))
            span = (total - 1) * stride + 1

        t0_max = max(0, T - span)
        if self.split != "train" or t0_max == 0:
            t0 = 0
        else:
            rng = random.Random(self.cfg.seed + idx)
            t0 = rng.randint(0, t0_max)

        all_idx = [t0 + j * stride for j in range(total)]
        all_idx = [min(i, T - 1) for i in all_idx]

        return all_idx[:n], all_idx[n:]

    # ------------------------------------------------------------------
    # __len__ / __getitem__
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ep = self.episodes[idx]
        task = ep["task"] or "robot manipulation"
        data_path = ep["data_path"]
        video_path = ep["video_path"]

        # ── Resolve episode length ──────────────────────────────────────────
        T = int(ep["length"] or 0)
        if T == 0:
            # Probe from parquet row count (fast, no video decode)
            if data_path.is_file():
                try:
                    import pyarrow.parquet as pq

                    T = pq.read_metadata(data_path).num_rows
                except (ImportError, OSError, ValueError) as exc:
                    # pyarrow missing or unreadable parquet: fall back to the video length below
                    logger.debug("cannot read row count of {}: {}", data_path, exc)
            if T == 0 and video_path.is_file():
                T = self._get_video_length(video_path)
            if T == 0:
                T = self.cfg.num_sample_frames + self.cfg.num_predict_frames

        # ── Sample frame indices ────────────────────────────────────────────
        n = self.cfg.num_sample_frames
        p = self.cfg.num_predict_frames
        ctx_indices, fut_indices = self._sample_window(T, idx)

        # ── Load actions + states from parquet ─────────────────────────────
        acts_all, sts_all, T_pq = self._load_episode_parquet(data_path)

        # Slice the action/state window aligned to context frames
        actions, action_mask, states, state_mask = self._slice_actions_states(
            acts_all, sts_all, ctx_indices
        )

        # ── Load video frames ───────────────────────────────────────────────
        all_frame_indices = ctx_indices + fut_indices
        if video_path.is_file():
            frame_tensors = self._read_frames_at_indices(video_path, all_frame_indices)
            images = torch.stack(frame_tensors[:n], dim=0)
            future_frames = torch.stack(frame_tensors[n:], dim=0)
        else:
            logger.warning("Video missing: {} — using dummy frames", video_path)
            images = self._dummy_frames(n)
            future_frames = self._dummy_frames(p)

        # ── Conversation tokens ─────────────────────────────────────────────
        user_body, asst_body = self._build_text(task, num_images=n)
        input_ids, attention_mask_tok, labels = self._tokenise(user_body, asst_body)

        # Pool states: one <state> token in the prompt → single vector [1, state_dim]
        if states.numel() > 0:
            states_in = states.mean(dim=0, keepdim=True)  # [1, state_dim]
            state_mask_in = torch.ones(1, dtype=torch.long)
        else:
            states_in = torch.zeros(1, self.cfg.state_dim)
            state_mask_in = torch.zeros(1, dtype=torch.long)

        return {
            "images": images,  # [n, 3, H, W]
            "future_frames": future_frames,  # [p, 3, H, W]
            "input_ids": input_ids,  # [seq_len]
            "attention_mask": attention_mask_tok,  # [seq_len]
            "labels": labels,  # [seq_len]
            "actions": actions,  # [T, action_dim]
            "action_mask": action_mask,  # [T]
            "states": states_in,  # [1, state_dim]
            "state_mask": state_mask_in,  # [1]
            "num_images": images.size(0),
            "pad_token_id": self.pad_token_id,
        }

    def _slice_actions_states(
        self,
        acts: np.ndarray,
        sts: np.ndarray,
        frame_indices: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract and truncate action + state rows aligned to ``frame_indices``.

        DROID parquet already has one action per frame, so we just index-select
        the rows that correspond to the sampled context window.
        """
        D_a = self.cfg.action_dim
        D_s = self.cfg.state_dim

        if acts.shape[0] == 0:
            # No parquet data — return empty tensors (training continues
            # with the language/vision branches only).
            return (
                torch.zeros(0, D_a),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, D_s),
                torch.zeros(0, dtype=torch.long),
            )

        T = acts.shape[0]
        idx_clamped = [min(max(0, i), T - 1) for i in frame_indices]

        sel_acts = acts[idx_clamped]  # [n, D_a]
        sel_sts = (
            sts[idx_clamped] if sts.shape[0] > 0 else np.zeros((len(idx_clamped), D_s), np.float32)
        )

        # Consecutive action differences give a delta-action representation.
        # When the parquet already stores delta actions (DROID default), the
        # raw values are used directly and this diff is a second-order delta —
        # acceptable for imitation learning.  Override this behaviour by
        # setting action_dim to the raw parquet width if you need absolute actions.
        if sel_acts.shape[0] > 1:
            diff = np.diff(sel_acts, axis=0)  # [n-1, D_a]
        else:
            diff = sel_acts  # [1, D_a] when single frame

        max_t = self.cfg.max_action_len
        diff = diff[:max_t]
        action_mask = torch.ones(diff.shape[0], dtype=torch.long)

        return (
            torch.from_numpy(diff),
            action_mask,
            torch.from_numpy(sel_sts.astype(np.float32)),
            torch.ones(sel_sts.shape[0], dtype=torch.long),
        )
