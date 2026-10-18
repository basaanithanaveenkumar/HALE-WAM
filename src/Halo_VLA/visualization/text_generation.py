"""Load a Halo-VLA checkpoint and generate/compare answers for visualisation."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from loguru import logger

from config import HaloVLMConfig
from visualization.action_plots import (  # noqa: F401 - moved; re-exported for callers
    action_comparison_image,
    tensor_to_display_image,
    unnormalise_image,
)
from visualization.drawing import (  # noqa: F401 - moved; re-exported for callers
    _FONT_CACHE,
    _FONT_CANDIDATES_BOLD,
    _FONT_CANDIDATES_REGULAR,
    _PRAI_BAR_H,
    _PRAI_PHASES,
    _bgr_to_pil,
    _import_cv2,
    _import_matplotlib,
    _import_pil,
    _pil_to_bgr,
    add_divider,
    compose_frame,
    draw_correctness_badge,
    draw_header,
    draw_panel,
    draw_prai_bar,
    get_font,
    wrap_text,
)


# ---------------------------------------------------------------------------
# Checkpoint / model helpers (optional — used when --checkpoint is given)
# ---------------------------------------------------------------------------
def load_model(ckpt_path: str, device: torch.device) -> tuple[nn.Module, HaloVLMConfig]:
    """Load a training checkpoint into a ``HaloVLM`` in eval mode.

    Args:
        ckpt_path: ``halo_vla_epoch*.pt`` written by ``scripts/train.py``.
        device: where to put the model.

    Returns:
        ``(model, config)``; missing keys are logged and left at their random initialisation.
    """
    from models.halo_vla import HaloVLM

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", HaloVLMConfig())
    model = HaloVLM(config=config).to(device)
    miss, _ = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if miss:
        logger.warning(
            "{} missing keys after load (random init), e.g. {}",
            len(miss),
            miss[:5],
        )
    model.eval()
    logger.info("Loaded checkpoint {}  (epoch {})", ckpt_path, ckpt.get("epoch", -1))
    return model, config


@torch.no_grad()
def generate_text(
    model: nn.Module,
    tokenizer: Any,
    images: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    states: torch.Tensor,
    max_new_tokens: int = 128,
    device: torch.device | str = "cpu",
) -> tuple[str, torch.Tensor | None]:
    """Greedy auto-regressive generation. Returns decoded string + action preds."""
    eos_id = tokenizer.eos_token_id
    action_preds = None
    original_len = input_ids.size(1)

    for _ in range(max_new_tokens):
        logits, action_hiddens, _, _ = model(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            states=states,
        )
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        token_id = next_token.item()
        if token_id == eos_id:
            break
        input_ids = torch.cat([input_ids, next_token], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones(1, 1, dtype=torch.long, device=device)],
            dim=1,
        )
        if action_hiddens is not None:
            act = model.sample_actions(action_hiddens)
        else:
            act = None
        if act is not None:
            action_preds = act

    gen_ids = input_ids[0, original_len:].cpu().tolist()
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text, action_preds


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------
def extract_question_answer(sample: dict) -> tuple[str, str]:
    """Pull the first user question and assistant answer from raw conversation."""
    conv = sample.get("conversation", [])
    question, answer = "", ""
    for turn in conv:
        role = turn.get("from", turn.get("role", "")).lower()
        value = turn.get("value", turn.get("content", ""))
        if role in ("human", "user") and not question:
            question = value.strip()
        elif role not in ("human", "user", "system") and question and not answer:
            answer = value.strip()
            break
    return question, answer


def simple_match(gt: str, pred: str, threshold: float = 0.5) -> bool:
    """Rough word-overlap correctness heuristic."""
    gt_words = set(gt.lower().split())
    pred_words = set(pred.lower().split())
    if not gt_words:
        return True
    overlap = len(gt_words & pred_words) / len(gt_words)
    return overlap >= threshold
