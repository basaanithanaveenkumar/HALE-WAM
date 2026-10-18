"""Tiny stand-ins for HaloVLM and its dataloader, used to run ``scripts/train.py``'s real
``train(args)`` end to end (single process, DDP, WebDataset, Ray) in seconds on a CPU.

They implement exactly the surface the training loop touches: the 4-tuple ``forward``,
``compute_flow_loss`` / ``compute_visual_prediction_loss`` (which, like the real model, use
parameters outside ``forward``) and ``dataset.tokenizer``.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
# Same import roots as scripts/train.py (the Halo_VLA sources are not an installed package).
for _path in (ROOT, ROOT / "src" / "Halo_VLA"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
VOCAB = 50
SEQ_LEN = 12
ACTION_DIM = 4
STATE_DIM = 3


class TinyTokenizer:
    """The two tokenizer members the training loop's debug print uses."""

    pad_token_id = 0

    @staticmethod
    def decode(ids, skip_special_tokens: bool = False) -> str:
        return " ".join(str(int(i)) for i in ids)


class TinyVLADataset(Dataset):
    """Deterministic samples with the keys and shapes of the real collated VLA batch."""

    tokenizer = TinyTokenizer()

    def __init__(self, length: int = 16) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        g = torch.Generator().manual_seed(index)
        ids = torch.randint(1, VOCAB, (SEQ_LEN,), generator=g)
        return {
            "index": torch.tensor(index),  # lets tests check which rank saw which sample
            "images": torch.rand(2, 3, 8, 8, generator=g),
            "input_ids": ids,
            "attention_mask": torch.ones(SEQ_LEN, dtype=torch.long),
            "labels": ids.clone(),
            "actions": torch.randn(5, ACTION_DIM, generator=g),
            "action_mask": torch.ones(5),
            "states": torch.randn(1, STATE_DIM, generator=g),
        }


class TinyHaloVLM(nn.Module):
    """``HaloVLM``-shaped model: forward returns (logits, action_hiddens, visual_ctx, world)."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(VOCAB, 16)
        self.image_proj = nn.Linear(3 * 8 * 8, 16)
        self.lm_head = nn.Linear(16, VOCAB)
        # Used only inside the loss methods, like the real action decoder and DiT.
        self.action_head = nn.Linear(16, ACTION_DIM)
        self.visual_head = nn.Linear(16, 3 * 8 * 8)
        self.seen_indices: list[int] = []

    def forward(self, images, input_ids, attention_mask, states, image_mask=None):
        img = self.image_proj(images.flatten(2)).mean(1, keepdim=True)  # (B, 1, 16)
        hidden = self.embed(input_ids) + img
        return self.lm_head(hidden), hidden, img, None

    def compute_flow_loss(self, action_hiddens, actions, action_mask):
        pred = self.action_head(action_hiddens.mean(1, keepdim=True)).expand_as(actions)
        return ((pred - actions) ** 2 * action_mask[..., None]).mean()

    def sample_actions(self, action_hiddens, num_steps=None):
        """``(B, n, 16)`` -> ``(B, 1, 5, ACTION_DIM)`` deterministic "chunks" for the evaluator."""
        pred = self.action_head(action_hiddens.mean(1))
        return pred[:, None, None, :].expand(-1, 1, 5, -1)

    def compute_visual_prediction_loss(self, images, image_mask, visual_context_emb, **_kwargs):
        recon = self.visual_head(visual_context_emb[:, 0]).view(images.shape[0], 3, 8, 8)
        loss = F.mse_loss(recon, images[:, -1])
        return loss, {"loss_rgb_cfm": loss.detach()}


def build_tiny_loader(batch_size: int = 2, num_workers: int = 0, max_samples: int | None = 16, **_kw):
    """Drop-in for ``build_eo_dataloader`` (ignores the dataset-specific arguments)."""
    return DataLoader(
        TinyVLADataset(max_samples or 16), batch_size=batch_size, shuffle=True, num_workers=num_workers
    )


def load_train_script() -> ModuleType:
    """Import ``scripts/train.py`` and swap in the tiny model and dataloader."""
    spec = importlib.util.spec_from_file_location("halo_train_script", ROOT / "scripts" / "train.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.HaloVLM = TinyHaloVLM
    from training.train_datasets import TRAIN_DATASETS

    # The "eo" dataset builder stands in for EO-Data1.5M with the tiny in-memory loader.
    TRAIN_DATASETS.register_module(name="eo", force=True)(
        lambda args, config, tokenizer: build_tiny_loader(
            batch_size=args.batch_size, num_workers=args.num_workers, max_samples=args.max_samples)
    )
    module.log_module_parameters = lambda *a, **k: None
    return module


def train_tiny(args: argparse.Namespace) -> None:
    """Ray entry point (``tests.support.training_support:train_tiny``): the real ``train`` on tiny fakes."""
    load_train_script().train(args)


def tiny_args(module: ModuleType, ckpt_dir: Path, *extra: str) -> argparse.Namespace:
    """Parse the script's real CLI with settings small enough for a CPU test."""
    argv = [
        "train.py", "--device", "cpu", "--epochs", "2", "--batch_size", "2", "--num_workers", "0",
        "--max_samples", "16", "--save_every", "1", "--keep_ckpts", "0", "--vis_every", "0",
        "--tensorboard_dir", "", "--ckpt_dir", str(ckpt_dir), "--log_every", "1000", *extra,
    ]
    old, sys.argv = sys.argv, argv
    try:
        return module.parse_args()
    finally:
        sys.argv = old
