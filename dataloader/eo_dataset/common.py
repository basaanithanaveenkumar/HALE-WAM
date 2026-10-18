"""Helpers shared by the ``eo_dataset`` package."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from .eo_dataset import EODataset
from .eo_dataset_config import EODatasetConfig

# ---------------------------------------------------------------------------
# All available subsets
# ---------------------------------------------------------------------------
INTERLEAVE_SUBSETS = [
    "interleave-free_chat",
    "interleave-random_qa",
    "interleave-temporal",
    "interleave-trajectory",
    "interleave-video_caption",
]


QA_SUBSETS = [
    "qa-affordance_qa",
    "qa-episode_caption",
    "qa-failure_detection",
    "qa-multiview_qa",
    "qa-object_referring_qa",
    "qa-physical_common_sense",
    "qa-points_qa",
    "qa-process_verification",
    "qa-relation_reasoning",
    "qa-subtask_qa",
    "qa-task_planning",
    "qa-trajectory_qa",
]


ALL_SUBSETS = INTERLEAVE_SUBSETS + QA_SUBSETS


# ---------------------------------------------------------------------------
# Collate function for variable-length image sequences
# ---------------------------------------------------------------------------
def eo_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Custom collate that pads images, actions, and states to the
    per-batch maximum (no fixed padding).

    Returns:
        images        : [B, max_N, 3, H, W]
        image_mask    : [B, max_N]            (1 for real images, 0 for pad)
        input_ids     : [B, seq_len]
        attention_mask : [B, seq_len]
        labels        : [B, seq_len]
        actions       : [B, max_T_act, action_dim]
        action_mask   : [B, max_T_act]
        states        : [B, max_T_state, state_dim]
        state_mask    : [B, max_T_state]
    """
    B = len(batch)

    # --- Images (variable count per sample) ---
    max_n_images = max(b["num_images"] for b in batch)
    C = batch[0]["images"].size(1)
    H = batch[0]["images"].size(2)
    W = batch[0]["images"].size(3) if batch[0]["images"].dim() == 4 else batch[0]["images"].size(2)
    padded_images = torch.zeros(B, max_n_images, C, H, W)
    image_mask = torch.zeros(B, max_n_images, dtype=torch.long)

    # --- Actions (variable timesteps per sample) ---
    action_dim = batch[0]["actions"].size(-1) if batch[0]["actions"].dim() == 2 else 0
    max_t_act = max(b["actions"].size(0) for b in batch)
    max_t_act = max(max_t_act, 1)  # at least 1 to avoid empty tensors
    padded_actions = torch.zeros(B, max_t_act, action_dim)
    padded_action_mask = torch.zeros(B, max_t_act, dtype=torch.long)

    # --- States (variable timesteps per sample) ---
    state_dim = batch[0]["states"].size(-1) if batch[0]["states"].dim() == 2 else 0
    max_t_state = max(b["states"].size(0) for b in batch)
    max_t_state = max(max_t_state, 1)
    padded_states = torch.zeros(B, max_t_state, state_dim)
    padded_state_mask = torch.zeros(B, max_t_state, dtype=torch.long)

    input_ids = []
    attention_masks = []
    labels_list = []

    for i, b in enumerate(batch):
        # Images
        n = b["num_images"]
        padded_images[i, :n] = b["images"][:n]
        image_mask[i, :n] = 1

        # Text
        input_ids.append(b["input_ids"])
        attention_masks.append(b["attention_mask"])
        labels_list.append(b["labels"])

        # Actions
        t_act = b["actions"].size(0)
        if t_act > 0:
            padded_actions[i, :t_act] = b["actions"]
            padded_action_mask[i, :t_act] = b["action_mask"]

        # States
        t_state = b["states"].size(0)
        if t_state > 0:
            padded_states[i, :t_state] = b["states"]
            padded_state_mask[i, :t_state] = b["state_mask"]

    # --- Text (dynamic padding to per-batch max) ---
    pad_token_id = batch[0].get("pad_token_id", 0)
    max_text_len = max(ids.size(0) for ids in input_ids)
    padded_input_ids = torch.full((B, max_text_len), pad_token_id, dtype=torch.long)
    padded_attention_mask = torch.zeros(B, max_text_len, dtype=torch.long)
    padded_labels = torch.full((B, max_text_len), -100, dtype=torch.long)
    for i in range(B):
        L = input_ids[i].size(0)
        padded_input_ids[i, :L] = input_ids[i]
        padded_attention_mask[i, :L] = attention_masks[i]
        padded_labels[i, :L] = labels_list[i]

    out = {
        "images": padded_images,  # [B, max_N, 3, H, W]
        "image_mask": image_mask,  # [B, max_N]
        "input_ids": padded_input_ids,  # [B, max_text_len]
        "attention_mask": padded_attention_mask,  # [B, max_text_len]
        "labels": padded_labels,  # [B, max_text_len]
        "actions": padded_actions,  # [B, max_T_act, action_dim]
        "action_mask": padded_action_mask,  # [B, max_T_act]
        "states": padded_states,  # [B, max_T_state, state_dim]
        "state_mask": padded_state_mask,  # [B, max_T_state]
    }

    # Optional: future_frames from datasets that provide ground-truth prediction targets.
    # Only stack when ALL items in the batch have future_frames to avoid shape mismatches.
    items_with_future = [b["future_frames"] for b in batch if "future_frames" in b]
    if len(items_with_future) == B:
        out["future_frames"] = torch.stack(items_with_future, dim=0)

    return out


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------
def build_eo_dataloader(
    subset: str = "interleave-temporal",
    split: str = "train",
    batch_size: int = 4,
    num_workers: int = 4,
    tokenizer: Any = None,
    img_size: int = 224,
    max_seq_len: int = 512,
    max_action_len: int = 64,
    action_dim: int = 32,
    state_dim: int = 32,
    num_predict_frames: int = 1,
    streaming: bool = False,
    cache_dir: str | None = None,
    shuffle: bool = True,
    pin_memory: bool = True,
    **kwargs: Any,
) -> DataLoader:
    """
    Build a ready-to-use DataLoader for EO-Data1.5M.

    Example usage::

        from data import build_eo_dataloader

        loader = build_eo_dataloader(
            subset="interleave-temporal",
            batch_size=8,
            num_workers=4,
        )
        for batch in loader:
            images = batch["images"]           # [B, N, 3, 224, 224]
            input_ids = batch["input_ids"]     # [B, 512]
            actions = batch["actions"]         # [B, 64, 7]
            ...
    """
    cfg = EODatasetConfig(
        subset=subset,
        split=split,
        img_size=img_size,
        max_seq_len=max_seq_len,
        max_action_len=max_action_len,
        action_dim=action_dim,
        state_dim=state_dim,
        num_predict_frames=num_predict_frames,
        streaming=streaming,
        cache_dir=cache_dir,
        max_samples=kwargs.pop("max_samples", None),
    )

    dataset = EODataset(config=cfg, tokenizer=tokenizer, **kwargs)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and not streaming,
        num_workers=num_workers,
        collate_fn=eo_collate_fn,
        pin_memory=pin_memory,
        drop_last=True,
    )
