"""AiroaMomaDataset: one training item = one **window** of frames from one episode + task text + state/action slice."""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset

from dataloader.airoa_moma_dataset.moma_episode_index import MomaEpisodeIndexMixin
from dataloader.airoa_moma_dataset.moma_prompt import MomaPromptMixin
from dataloader.eo_dataset import build_image_transform
from exceptions import DatasetNotFoundError

from .airoa_moma_config import AiroaMomaConfig


class AiroaMomaDataset(MomaPromptMixin, MomaEpisodeIndexMixin, Dataset):
    """
    One training item = one **window** of frames from one episode + task text + state/action slice.

    ``images`` are ordered in time; the **last** image is the "future" target for the
    visual DiT head when ``num_sample_frames >= 2`` (matches ``compute_visual_prediction_loss``).
    """

    def __init__(
        self,
        config: AiroaMomaConfig | None = None,
        tokenizer: Any = None,
        split: str = "train",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if config is None:
            config = AiroaMomaConfig(**kwargs)
        self.cfg = config
        self.split = split
        self.root = Path(config.data_root).expanduser().resolve()
        if not self.root.is_dir():
            raise DatasetNotFoundError(f"data_root is not a directory: {self.root}")

        self._setup_tokenizer(tokenizer)
        self.img_transform = build_image_transform(
            self.cfg.img_size, self.cfg.img_mean, self.cfg.img_std
        )

        # Load episodes: prefer parquet metadata, fall back to video-file scan.
        raw = self._load_episodes()
        self.episodes = self._filter_episodes(raw)
        if self.cfg.max_samples is not None:
            self.episodes = self.episodes[: self.cfg.max_samples]
        # _rng is only used as a fallback; per-item sampling uses a per-index seed.

        logger.info(
            "AiroaMomaDataset: {} episodes under {} (camera={}, frames={}, stride={})",
            len(self.episodes),
            self.root,
            self.cfg.camera,
            self.cfg.num_sample_frames,
            self.cfg.frame_stride,
        )

    # ------------------------------------------------------------------
    # Episode discovery: parquet → video scan → jsonl (priority order)
    # ------------------------------------------------------------------

    # def _probe_num_frames(self, video_path: Path) -> int:
    #     cv2 = _import_cv2()
    #     cap = cv2.VideoCapture(str(video_path))
    #     if not cap.isOpened():
    #         return 0
    #     n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #     cap.release()
    #     return max(n, 0)
    def _get_video_frame_count(self, video_path: Path) -> int:
        """
        Return frame count without decoding the video.

        Priority:
          1. stream.frames — set by most MP4 muxers, instant.
          2. Estimate from duration × fps — fast, off by ±1 at worst.
          3. Count demux packets (no decode) — much faster than full decode.
        Full-decode fallback is intentionally removed: it could take 30-40s on
        long robot episodes and was called on every __getitem__ call.
        """
        import av

        try:
            container = av.open(str(video_path))
            stream = container.streams.video[0]

            # Fast path 1: muxer wrote frame count into the stream header.
            if stream.frames and stream.frames > 0:
                container.close()
                return int(stream.frames)

            # Fast path 2: duration × average frame rate (no decoding).
            if stream.duration and stream.average_rate:
                dur_s = float(stream.duration) * stream.time_base
                fps = float(stream.average_rate)
                if dur_s > 0 and fps > 0:
                    container.close()
                    return max(1, round(dur_s * fps))

            # Fast path 3: count packets (mux-level, no pixel decoding).
            count = sum(1 for p in container.demux(stream) if p.size > 0)
            container.close()
            return max(count, 1)
        except Exception as e:
            logger.warning(f"Failed to get frame count for {video_path}: {e}")
            return 0

    # def _read_frames_at_indices(
    #     self, video_path: Path, indices: Sequence[int]
    # ) -> List[torch.Tensor]:
    #     """Return list of [3,H,W] float tensors (ImageNet-normalised), order = ``indices``."""
    #     from PIL import Image

    #     cv2 = _import_cv2()
    #     cap = cv2.VideoCapture(str(video_path))
    #     if not cap.isOpened():
    #         raise FileNotFoundError(f"Cannot open video: {video_path}")

    #     want = sorted(set(int(i) for i in indices))
    #     frame_map: Dict[int, torch.Tensor] = {}
    #     pos = 0
    #     wi = 0
    #     while wi < len(want):
    #         target = want[wi]
    #         while pos < target:
    #             if not cap.read()[0]:
    #                 break
    #             pos += 1
    #         if pos != target:
    #             break
    #         ret, bgr = cap.read()
    #         if not ret:
    #             break
    #         rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    #         pil = Image.fromarray(rgb)
    #         frame_map[target] = self.img_transform(pil)
    #         pos += 1
    #         wi += 1
    #     cap.release()

    #     blank = torch.zeros(3, self.cfg.img_size, self.cfg.img_size)
    #     out: List[torch.Tensor] = []
    #     for i in indices:
    #         ii = int(i)
    #         if ii in frame_map:
    #             out.append(frame_map[ii])
    #         else:
    #             logger.warning(
    #                 "Frame index {} missing in {}; using blank", ii, video_path.name
    #             )
    #             out.append(blank.clone())
    #     return out
    # Add at top of file

    # Replace _read_frames_at_indices with:
    # def _read_frames_at_indices(self, video_path, indices):
    #     from decord import VideoReader, cpu
    #     vr = VideoReader(str(video_path), ctx=cpu(0))
    #     # decord indices are 0-based; ensure they're within range
    #     valid_idx = [i for i in indices if i < len(vr)]
    #     frames = vr.get_batch(valid_idx).asnumpy()  # [T, H, W, C] (RGB)
    #     # Convert to torch tensor and apply transform
    #     transformed = []
    #     j = 0
    #     for i in indices:
    #         if i in valid_idx:
    #             pil = Image.fromarray(frames[j])
    #             transformed.append(self.img_transform(pil))
    #             j += 1
    #         else:
    #             transformed.append(torch.zeros(3, self.cfg.img_size, self.cfg.img_size))
    #     return transformed
    # def _read_frames_at_indices(self, video_path: Path, indices: Sequence[int]) -> List[torch.Tensor]:
    #     """Read specific frame indices from a video using PyAV."""
    #     import av
    #     from PIL import Image

    #     container = av.open(str(video_path))
    #     # Assume first video stream
    #     stream = container.streams.video[0]

    #     # Build a set of requested indices for fast lookup
    #     required = set(int(i) for i in indices)
    #     frame_map = {}

    #     # Decode frames sequentially (fast enough for short videos / small frame requests)
    #     for packet in container.demux(stream):
    #         for frame in packet.decode():
    #             pts = int(frame.pts)  # presentation timestamp (frame index)
    #             if pts in required:
    #                 # Convert frame to PIL RGB image
    #                 img = frame.to_image().convert('RGB')
    #                 frame_map[pts] = self.img_transform(img)
    #                 required.remove(pts)
    #                 if not required:
    #                     break
    #         if not required:
    #             break

    #     container.close()

    #     # Build output list: if a requested index wasn't found, insert a blank
    #     blank = torch.zeros(3, self.cfg.img_size, self.cfg.img_size)
    #     out = []
    #     for i in indices:
    #         ii = int(i)
    #         if ii in frame_map:
    #             out.append(frame_map[ii])
    #         else:
    #             logger.warning(f"Frame index {ii} missing in {video_path.name}; using blank")
    #             out.append(blank.clone())
    #     return out
    def _read_frames_at_indices(
        self, video_path: Path, indices: Sequence[int]
    ) -> list[torch.Tensor]:
        """Read specific frame indices using PyAV with a simple counter."""
        import av

        container = av.open(str(video_path))
        stream = container.streams.video[0]
        required = set(int(i) for i in indices)
        frame_map = {}
        frame_counter = 0

        for packet in container.demux(stream):
            for frame in packet.decode():
                if frame_counter in required:
                    img = frame.to_image().convert("RGB")
                    frame_map[frame_counter] = self.img_transform(img)
                    required.remove(frame_counter)
                    if not required:
                        break
                frame_counter += 1
            if not required:
                break

        container.close()

        blank = torch.zeros(3, self.cfg.img_size, self.cfg.img_size)
        out = []
        for i in indices:
            ii = int(i)
            if ii in frame_map:
                out.append(frame_map[ii])
            else:
                # This warning should now be rare because indices are clamped
                logger.warning(f"Frame index {ii} missing in {video_path.name}; using blank")
                out.append(blank.clone())
        return out

    def _load_state_trajectory(self, episode_index: int) -> np.ndarray | None:
        sub = self.root / self.cfg.state_npy_subdir / f"episode_{int(episode_index):06d}.npy"
        if not sub.is_file():
            return None
        arr = np.load(sub)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr.astype(np.float32)

    def _generate_dummy_frames(self, num_frames: int) -> torch.Tensor:
        """Return [num_frames, 3, H, W] black images."""
        shape = (num_frames, 3, self.cfg.img_size, self.cfg.img_size)
        return torch.zeros(shape, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        meta = self.episodes[idx]
        ep = int(meta.get("episode_index", idx))
        vid_path = self._video_path(ep, meta=meta)

        # Use parquet/jsonl length when available — avoids opening the video.
        # Only fall back to probing the file when length is missing or zero.
        T_meta = int(meta.get("length", 0) or 0)
        if T_meta > 0:
            T = T_meta
        elif vid_path.is_file():
            T = self._get_video_frame_count(vid_path)
        else:
            T = 0

        n = self.cfg.num_sample_frames
        p = self.cfg.num_predict_frames  # future frames to predict
        total = n + p
        stride = self.cfg.frame_stride
        span = (total - 1) * stride + 1
        if T < span:
            stride = max(1, (T - 1) // max(total - 1, 1))
            span = (total - 1) * stride + 1
        t0_max = max(0, T - span)
        if self.split != "train" or t0_max == 0:
            # Val/test and single-window episodes are always deterministic.
            t0 = 0
        else:
            # Per-index seed: same idx → same window every epoch, different
            # indices → different windows.  Supports overfit testing with any
            # value of max_samples (not just 1).
            rng = random.Random(self.cfg.seed + idx)
            t0 = rng.randint(0, t0_max)
        all_indices = [t0 + j * stride for j in range(total)]
        frame_indices = all_indices[:n]  # context/input frames
        future_indices = all_indices[n:]  # prediction target frames

        # Clamp indices to valid range (0 .. T-1) if video exists
        if vid_path.is_file() and T > 0:
            frame_indices = [min(i, T - 1) for i in frame_indices]
            future_indices = [min(i, T - 1) for i in future_indices]

        if not vid_path.is_file():
            logger.warning(f"Video missing: {vid_path} – using dummy frames")
            images = self._generate_dummy_frames(n)
            future_frames = self._generate_dummy_frames(p)
            frame_indices = list(range(n))
        else:
            frame_tensors = self._read_frames_at_indices(vid_path, frame_indices + future_indices)
            images = torch.stack(frame_tensors[:n], dim=0)  # [n, 3, H, W]
            future_frames = torch.stack(frame_tensors[n:], dim=0)  # [p, 3, H, W]

        user_body, assistant_body = self._build_text(meta, num_images=n)
        input_ids, attention_mask, labels = self._tokenise_turns(user_body, assistant_body)

        state_traj = self._load_state_trajectory(ep)
        actions, action_mask, states_per_frame, _ = self._align_actions_states(
            state_traj, frame_indices
        )
        # One <state> token in the prompt → single pooled vector [1, state_dim]
        if states_per_frame.numel() > 0:
            states = states_per_frame.mean(dim=0, keepdim=True)
            state_mask = torch.ones(1, dtype=torch.long)
        else:
            states = torch.ones(1, self.cfg.state_dim)
            state_mask = torch.zeros(1, dtype=torch.long)

        return {
            "images": images,
            "future_frames": future_frames,  # [p, 3, H, W] — prediction targets
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "actions": actions,
            "action_mask": action_mask,
            "states": states,
            "state_mask": state_mask,
            "num_images": images.size(0),
            "pad_token_id": self.pad_token_id,
        }

    def _align_actions_states(
        self,
        state_traj: np.ndarray | None,
        frame_indices: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Align proprio to frame indices. Actions = consecutive differences in state.
        If no ``states/*.npy``, return empty tensors (masked).
        """
        n = len(frame_indices)
        D_a = self.cfg.action_dim
        D_s = self.cfg.state_dim
        if state_traj is None:
            # Use fixed seed so dummy tensors are identical every call (important when
            # max_samples=1 and there are no .npy state files).
            gen = torch.Generator().manual_seed(42)
            dummy_states = torch.randn(n, D_s, generator=gen)
            dummy_actions = torch.randn(max(0, n - 1), D_a, generator=gen)
            state_mask = torch.ones(n, dtype=torch.long)
            action_mask = torch.ones(dummy_actions.shape[0], dtype=torch.long)
            # truncation
            if dummy_actions.shape[0] > self.cfg.max_action_len:
                dummy_actions = dummy_actions[: self.cfg.max_action_len]
                action_mask = action_mask[: self.cfg.max_action_len]
            return dummy_actions, action_mask, dummy_states, state_mask

        T = state_traj.shape[0]
        idx = [min(max(0, i), T - 1) for i in frame_indices]
        sel = state_traj[idx]  # [n, d_raw]
        d_raw = sel.shape[1]

        # States: one row per sampled frame, padded/truncated to state_dim
        states = np.zeros((sel.shape[0], D_s), dtype=np.float32)
        d_use = min(d_raw, D_s)
        states[:, :d_use] = sel[:, :d_use]
        state_mask = torch.ones(states.shape[0], dtype=torch.long)

        # Actions: finite differences along time within the window, length n-1
        diff = np.diff(sel, axis=0)  # [n-1, d_raw]
        if diff.size == 0:
            return (
                torch.zeros(0, D_a),
                torch.zeros(0, dtype=torch.long),
                torch.from_numpy(states),
                state_mask,
            )
        actions = np.zeros((diff.shape[0], D_a), dtype=np.float32)
        d_act = min(d_raw, D_a)
        actions[:, :d_act] = diff[:, :d_act]
        action_mask = torch.ones(actions.shape[0], dtype=torch.long)

        # Truncate to max_action_len
        max_t = self.cfg.max_action_len
        if actions.shape[0] > max_t:
            actions = actions[:max_t]
            action_mask = action_mask[:max_t]

        return (
            torch.from_numpy(actions),
            action_mask,
            torch.from_numpy(states),
            state_mask,
        )
