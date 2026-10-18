"""``scripts/train.py --dual_system openhelix`` end to end on a tiny real HaloVLM (CPU)."""

from __future__ import annotations

import functools
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "src" / "Halo_VLA"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

pytest.importorskip("hale_core")

from config import HaloVLMConfig  # noqa: E402
from models.halo_vla import HaloVLM  # noqa: E402


def test_train_script_with_dual_system(tmp_path: Path, monkeypatch) -> None:
    from tests.support.training_support import load_train_script, tiny_args

    module = load_train_script()
    module.HaloVLM = HaloVLM  # the real model: System 2 needs encode_sequence
    import training.model_config_builder as config_builder
    from training.train_datasets import TRAIN_DATASETS

    monkeypatch.setattr(
        config_builder,
        "HaloVLMConfig",
        functools.partial(
            HaloVLMConfig,
            img_size=32,
            patch_size=16,
            emb_dim=32,
            vit_num_layers=1,
            vit_num_heads=4,
            vit_mlp_dim=32,
            dec_num_layers=1,
            dec_num_heads=4,
            dec_mlp_dim=32,
            use_moe=False,
            enable_visual_dit=False,
            flow_hidden_dim=16,
        ),
    )

    class Rows(list):
        """Map-style dataset with the ``tokenizer`` attribute train.py reads."""

        tokenizer = type(
            "Tok", (), {"pad_token_id": 0, "decode": staticmethod(lambda *a, **k: "")}
        )()

    def build_loader(**_kw):
        g = torch.Generator().manual_seed(0)
        rows = Rows(
            {
                "images": torch.rand(2, 3, 32, 32, generator=g),
                "input_ids": torch.randint(1, 50, (8,), generator=g),
                "attention_mask": torch.ones(8, dtype=torch.long),
                "labels": torch.randint(1, 50, (8,), generator=g),
                "actions": torch.randn(4, 4, generator=g),
                "action_mask": torch.ones(4),
                "states": torch.randn(1, 3, generator=g),
            }
            for _ in range(8)
        )
        return DataLoader(rows, batch_size=2)

    monkeypatch.setitem(TRAIN_DATASETS.items, "eo", lambda args, config, tokenizer: build_loader())
    args = tiny_args(
        module,
        tmp_path,
        "--dual_system",
        "openhelix",
        "--action_dim",
        "4",
        "--state_dim",
        "3",
        "--action_chunk_size",
        "2",
        "--dual_system_option",
        "prealign_steps=2",
        "--dual_system_option",
        "system1_dim=32",
        "--dual_system_option",
        "system1_heads=4",
        "--dual_system_option",
        "system1_layers=1",
        "--val_fraction",
        "0.25",
    )
    module.train(args)
    ckpt = torch.load(tmp_path / "halo_vla_epoch2.pt", weights_only=False)
    assert any(k.startswith("system1.") for k in ckpt["model_state_dict"])
    assert ckpt["config"].dual_system == "openhelix"
