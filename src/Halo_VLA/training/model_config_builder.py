"""Build the :class:`HaloVLMConfig` of a training run from parsed command-line arguments."""

from __future__ import annotations

import argparse

from loguru import logger

from config import HaloVLMConfig
from config.overrides import apply_model_overrides, parse_model_options
from dual_system.resolution import parse_overrides


def build_model_config(args: argparse.Namespace) -> HaloVLMConfig:
    """Map ``scripts/train.py`` arguments (CLI flags, YAML defaults, overrides) onto a config.

    Order of precedence for architecture fields without a flag: the YAML's ``model_overrides``
    (set by ``--config``), then ``--model_option KEY=VALUE``.

    Args:
        args: the parsed arguments of :func:`training.train_arg_parser.parse_args`.

    Returns:
        The validated model configuration.

    Raises:
        HaloVLAConfigError: for an unknown or ill-typed override.
    """
    config = HaloVLMConfig(
        action_dim=args.action_dim,
        state_dim=args.state_dim,
        action_chunk_size=args.action_chunk_size,
        num_visual_predict_frames=args.num_predict_frames,
        proj_type=args.projector_type,
    )
    if args.num_query_tokens is not None:
        config.qformer_num_queries = args.num_query_tokens
        config.gated_xattn_num_latents = args.num_query_tokens
    if args.visual_loss_weight is not None:
        config.visual_loss_weight = args.visual_loss_weight
    # Vision encoder: built-in ViT, SigLIP / CLIP, or a HaleBlocks backbone (--vision_encoder).
    config.vision_encoder = args.vision_encoder
    config.vision_model_id = args.vision_model_id
    config.vision_backbone = args.vision_backbone
    config.vision_freeze = args.vision_freeze
    config.vision_freeze_up_to_stage = args.vision_freeze_up_to_stage
    config.vision_pretrained_path = args.vision_pretrained_path
    # Optional pretrained, fine-tunable LLM in place of the built-in decoder (--llm_backbone).
    config.llm_backbone = args.llm_backbone
    config.llm_model_id = args.llm_model_id
    config.llm_checkpoint = args.llm_checkpoint
    config.llm_finetune = args.llm_finetune
    config.llm_lora_r, config.llm_lora_alpha = args.llm_lora_r, args.llm_lora_alpha
    config.llm_lora_dropout = args.llm_lora_dropout
    config.llm_lora_train_embeddings = not args.llm_freeze_embeddings
    config.llm_dtype = args.llm_dtype
    config.llm_gradient_checkpointing = args.llm_gradient_checkpointing
    # Dual-system VLA (--dual_system): HaloVLM becomes System 2 of a slow/fast pair.
    config.dual_system = args.dual_system
    config.dual_system_options = parse_overrides(args.dual_system_option) or None
    config.llm_attn_implementation = args.llm_attn_implementation
    model_overrides = {
        **getattr(args, "model_overrides", {}),
        **parse_model_options(args.model_option),
    }
    if model_overrides:
        apply_model_overrides(config, model_overrides)
        logger.info("HaloVLMConfig overrides: {}", model_overrides)
    return config
