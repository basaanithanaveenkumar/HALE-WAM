"""HaloVLM: the Halo-VLA model - vision-language decoder with flow-matching action and DiT video heads."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from exceptions import ModelStateError
from loguru import logger

from config import HaloVLMConfig
from models.dit_frame_prediction import VisualDiTPredictor

# --- Flow matching action decoder replaces MLP ActionDecoder ---
from models.flow_action_decoder import FlowActionDecoder
from models.halo_vla_flow_head import FlowActionHeadMixin
from models.halo_vla_visual_prediction import VisualPredictionMixin
from models.image_proj import build_image_projector
from models.language_decoder import PretrainedLanguageDecoder
from models.lm_head import LMHead
from models.state_encoder import StateEncoder
from models.transformer import DecoderTransformer
from models.vision_encoders import build_vision_encoder


def pool_past_frame_embeddings_for_visual_dit(
    frame_stack: torch.Tensor,
    image_mask: torch.Tensor | None,
) -> torch.Tensor:
    """
    Combine per-frame pooled ViT embeddings into one DiT conditioning vector per sample.

    ``frame_stack`` holds one row per video frame (same order as ``images[:, i]``).
    The **future** frame is the last *valid* slot per row; we average all **strictly
    earlier** valid slots as "past context" (matches ``compute_visual_prediction_loss``).

    Args:
        frame_stack: [B, N, emb_dim] — typically ``proj_i.mean(dim=1)`` per frame.
        image_mask:  [B, N] or longer; only ``[:, :N]`` is used. None = all frames valid.

    Returns:
        [B, emb_dim] — zero rows where there is no past (e.g. a single valid frame).
    """
    B, N, D = frame_stack.shape
    device = frame_stack.device
    dtype = frame_stack.dtype
    if image_mask is not None:
        mask = image_mask[:, :N].to(device=device, dtype=dtype)
    else:
        mask = torch.ones(B, N, device=device, dtype=dtype)

    n_slots = mask.sum(dim=1).long()
    rev = mask.long().flip(dims=[1])
    from_end = rev.argmax(dim=1)
    last_idx = (N - 1 - from_end).clamp(0, N - 1)
    last_idx = torch.where(n_slots > 0, last_idx, torch.zeros_like(last_idx))

    idx = torch.arange(N, device=device, dtype=torch.long).view(1, N).expand(B, N)
    past_mask = mask * (idx < last_idx.unsqueeze(1))
    denom = past_mask.sum(dim=1, keepdim=True).clamp(min=1e-6)
    out = (frame_stack * past_mask.unsqueeze(-1)).sum(dim=1) / denom
    zero_rows = (past_mask.sum(dim=1) < 1).unsqueeze(-1)
    out = out.masked_fill(zero_rows, 0.0)
    return out


class HaloVLM(VisualPredictionMixin, FlowActionHeadMixin, nn.Module):
    """Halo-VLA: one causal decoder reading images, proprioception and text, with three task heads.

    Images (built-in ViT, SigLIP / CLIP or a HaleBlocks backbone) are projected to the decoder
    width and spliced in at ``<image>`` slots, states at ``<state>`` slots. The decoder's states at
    ``<halo_action>`` slots condition a conditional flow-matching action head; those at
    ``<halo_world_video>`` slots condition a DiT that predicts future frames. The decoder is either
    the built-in :class:`DecoderTransformer` or a pretrained LLM (``llm_backbone``). The heads live in
    :class:`FlowActionHeadMixin` and :class:`VisualPredictionMixin`.

    Args:
        config: model settings; ``None`` builds one from ``kwargs``.
        **kwargs: ``HaloVLMConfig`` fields used when ``config`` is ``None``.

    Example:
        >>> model = HaloVLM(HaloVLMConfig(emb_dim=64))  # doctest: +SKIP
        >>> out = model(images, input_ids, attention_mask, states)  # doctest: +SKIP
    """

    def __init__(self, config: HaloVLMConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        if config is None:
            config = HaloVLMConfig(**kwargs)
        self.config = config
        network_config = config

        # Vision encoder (built-in ViT, SigLIP / CLIP, or a HaleBlocks backbone). Its token width
        # feeds the image projector and its token count the language-loss offsets.
        self.vis_enc, vision = build_vision_encoder(network_config)
        if network_config.vision_encoder != "vit":
            network_config.proj_vision_dim = vision.output_dim
        network_config.vision_num_tokens = vision.num_tokens
        # Language decoder: a registered pretrained LLM (config.llm_backbone), or the built-in one.
        # The pretrained path is built first: it rewrites the special-token ids / vocab size in
        # ``config`` from the LLM's tokenizer, which every other component then reads.
        self.language = PretrainedLanguageDecoder(network_config) if network_config.uses_pretrained_llm else None
        self.decoder_transformer = None if self.language is not None else DecoderTransformer(
            num_layers=network_config.dec_num_layers,
            emb_dim=network_config.emb_dim,
            num_heads=network_config.dec_num_heads,
            mlp_dim=network_config.dec_mlp_dim,
            drop_fact=network_config.dec_drop,
            use_moe=network_config.use_moe,
            moe_hid_scale=network_config.moe_hid_scale,
            moe_num_routed_experts=network_config.moe_num_routed_experts,
            moe_top_k=network_config.moe_top_k,
            moe_num_shared_experts=network_config.moe_num_shared_experts,
            gradient_checkpointing=getattr(network_config, "gradient_checkpointing", True),
        )
        if self.language is None:
            self.token_emb = nn.Embedding(network_config.vocab_size, network_config.emb_dim)
            self.pos_embed = nn.Embedding(network_config.max_position_embeddings, network_config.emb_dim)
            self.lm_head = LMHead(hidden_size=network_config.emb_dim, vocab_size=network_config.vocab_size)
        self.image_projector = build_image_projector(network_config)
        self.state_encoder = StateEncoder(config=network_config)

        # --- Flow matching decoder: replaces the MLP ActionDecoder ---
        # action_dim_flat = chunk_size * action_dim  (the flow network operates on
        # the full flattened action chunk for each <halo_action> token position)
        action_dim_flat = network_config.action_chunk_size * network_config.action_dim
        self.flow_decoder = FlowActionDecoder(
            action_dim_flat=action_dim_flat,
            obs_dim=network_config.emb_dim,
            hidden_dim=network_config.flow_hidden_dim,
            time_embed_dim=network_config.flow_time_embed_dim,
        )
        # Store action shape info for reshape during sampling
        self.action_chunk_size = network_config.action_chunk_size
        self.action_dim = network_config.action_dim
        self.flow_num_ode_steps = network_config.flow_num_ode_steps

        # --- DiT visual prediction (future RGB, depth, optical flow) ---
        self.enable_visual_dit = getattr(network_config, "enable_visual_dit", True)
        if self.enable_visual_dit:
            n_pred = getattr(network_config, "num_visual_predict_frames", 1)
            self.visual_predictor = VisualDiTPredictor(network_config, num_predict_frames=n_pred)
        else:
            self.visual_predictor = None

    def encode_visual_context_from_past_vit(self, past_rgb: torch.Tensor) -> torch.Tensor:
        """
        Inference helper when no full ``forward`` is run: pool ViT + projector over
        **past** frames only [B, emb_dim].

        ``forward`` instead uses decoder hidden states at the same patch **positions**
        (after blocks, before ``lm_head``); use the third return value of ``forward``
        when training or when language context is available.
        """
        if past_rgb.dim() == 4:
            x = past_rgb.unsqueeze(1)
        else:
            x = past_rgb
        B, T, _, H, W = x.shape
        pooled_frames: list[torch.Tensor] = []
        for t in range(T):
            feat = self.vis_enc(x[:, t])
            proj = self.image_projector(feat)
            pooled_frames.append(proj.mean(dim=1))
        return torch.stack(pooled_frames, dim=1).mean(dim=1)

    def _embed_inputs(self, images, input_ids, states):
        """Steps 1-5 of :meth:`forward`: the decoder's input embeddings before positions.

        Returns ``(combined_embeds [B, N_img*P + L, W], num_prepended, N_img)`` where ``W`` is the
        decoder width (``emb_dim``, or the pretrained LLM's hidden size).
        """
        B = input_ids.size(0)
        device = input_ids.device
        network_config = self.config
        image_token_id = network_config.image_token_id
        state_token_id = network_config.state_token_id

        # ------------------------------------------------------------------ #
        # 1. IMAGES — count <image> tokens, encode each image through ViT,
        #    project into LLM space, and concatenate all patch features.
        #    img_proj → [B, N_img * num_patches, emb_dim]
        # ------------------------------------------------------------------ #
        num_images = (input_ids == image_token_id).sum(dim=1)  # [B]
        N_img = num_images.max().item()                        # scalar

        all_img_proj = []
        for i in range(N_img):
            img_i = images[:, i]                               # [B, 3, H, W]
            feat_i = self.vis_enc(img_i)                       # [B, num_patches, emb_dim]
            proj_i = self.image_projector(feat_i)              # [B, num_patches, emb_dim]
            all_img_proj.append(proj_i)

        if all_img_proj:
            img_proj = torch.cat(all_img_proj, dim=1)          # [B, N_img*num_patches, emb_dim]
        else:
            img_proj = torch.empty(B, 0, network_config.emb_dim, device=device)
        num_prepended = img_proj.size(1)                       # image tokens before text

        # ------------------------------------------------------------------ #
        # 2. STATES — count <state> tokens, encode each state vector through
        #    the StateEncoder MLP to get a single embedding per state.
        #    state_embeds → [B, N_state, emb_dim]
        # ------------------------------------------------------------------ #
        num_states = (input_ids == state_token_id).sum(dim=1)  # [B]
        N_state = num_states.max().item()                      # scalar

        all_state_embeds = []
        for i in range(N_state):
            state_i = states[:, i]                             # [B, state_dim]
            emb_i = self.state_encoder(state_i)                # [B, emb_dim]
            all_state_embeds.append(emb_i.unsqueeze(1))        # [B, 1, emb_dim]

        if all_state_embeds:
            state_embeds = torch.cat(all_state_embeds, dim=1)  # [B, N_state, emb_dim]
        else:
            state_embeds = torch.empty(B, 0, network_config.emb_dim, device=device)

        # ------------------------------------------------------------------ #
        # 3. TEXT EMBEDDINGS — embed all tokens, then zero out positions that
        #    correspond to <image> or <state> placeholders (their actual
        #    information comes from the encoders above).
        # ------------------------------------------------------------------ #
        # Zero out image and state placeholder positions; action and world_video tokens
        # keep their vocab embeddings so the transformer sees a meaningful signal there.
        special_mask = (
            (input_ids != image_token_id)
            & (input_ids != state_token_id)
        )
        if self.language is not None:
            # LLM space: text from the LLM's embeddings; images / states projected into it.
            text_embeds = self.language.embed_text(input_ids).to(img_proj.dtype)  # [B, seq_len, H]
            img_proj = self.language.project_in(img_proj)
            state_embeds = self.language.project_in(state_embeds)
        else:
            text_embeds = self.token_emb(input_ids)            # [B, seq_len, emb_dim]
        text_embeds = text_embeds * special_mask.unsqueeze(-1)  # zero out placeholders

        # ------------------------------------------------------------------ #
        # 4. INJECT STATE EMBEDDINGS — replace the zeroed-out <state> token
        #    positions in text_embeds with the encoded state embeddings.
        #    This keeps states at their original position in the sequence
        #    (rather than prepending like images).
        # ------------------------------------------------------------------ #
        for b in range(B):
            state_positions = (input_ids[b] == state_token_id).nonzero(as_tuple=True)[0]
            for idx, pos in enumerate(state_positions):
                if idx < N_state:
                    text_embeds[b, pos] = state_embeds[b, idx]

        # ------------------------------------------------------------------ #
        # 5. CONCATENATE — prepend image patch features before the text+state
        #    sequence so the decoder can attend to visual context first.
        #    combined_embeds → [B, N_img*num_patches + seq_len, emb_dim]
        # ------------------------------------------------------------------ #
        combined_embeds = torch.cat([img_proj, text_embeds], dim=1)
        return combined_embeds, num_prepended, N_img

    # ------------------------------------------------------------------ #
    # Sequence encoding for the dual-system VLA (System 2)
    # ------------------------------------------------------------------ #
    @property
    def layer_width(self) -> int:
        """Width of the decoder's hidden states (``emb_dim`` or the LLM's hidden size)."""
        return self.language.hidden_size if self.language is not None else self.config.emb_dim

    def sequence_token_ids(self, input_ids: torch.Tensor, num_prefix: int, num_extra: int = 0) -> torch.Tensor:
        """Token ids aligned with the decoder sequence ``[image prefix, text/states, extra]``.

        Image slots carry the image token id and appended queries the pad id. A pretrained LLM
        whose layers read token identity (Gemma 4's per-layer embeddings) needs them alongside
        the spliced embeddings.
        """
        B = input_ids.size(0)
        parts = [input_ids.new_full((B, num_prefix), self.config.image_token_id), input_ids]
        if num_extra:
            pad = getattr(self.language.tokenizer, "pad_token_id", None) if self.language is not None else None
            parts.append(input_ids.new_full((B, num_extra), pad or 0))
        return torch.cat(parts, dim=1)

    def decoder_layers(self) -> nn.ModuleList:
        """The decoder's transformer blocks (built-in or pretrained LLM), in order."""
        if self.language is None:
            return self.decoder_transformer.layers
        candidates = [m for n, m in self.language.llm.named_modules()
                      if isinstance(m, nn.ModuleList) and n.endswith("layers") and len(m) > 0]
        if not candidates:
            raise ModelStateError("could not locate the LLM backbone's decoder layers")
        return max(candidates, key=len)

    def encode_sequence(self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                        states: torch.Tensor, *, extra_embeds: torch.Tensor | None = None,
                        capture_layer: int | None = None) -> dict[str, Any]:
        """Run the vision-language decoder and return its hidden states (no task heads).

        Args:
            images, input_ids, attention_mask, states: as for :meth:`forward`.
            extra_embeds: optional ``(B, K, emb_dim)`` embeddings appended after the text
                (e.g. learnable ``<ACT>`` queries); being last, they attend to everything.
            capture_layer: optional decoder-layer index whose output is also returned.

        Returns:
            dict with ``hidden`` ``(B, T, emb_dim)`` final states, ``mask`` ``(B, T)`` (1 = real),
            ``logits`` ``(B, T, V)``, ``num_prefix`` (image tokens before the text),
            ``text_len``, ``num_extra``, ``input_embeds`` (decoder inputs before positions) and
            ``layer_hidden`` ``(B, T, layer_width)`` when ``capture_layer`` is set.
        """
        B, device = input_ids.size(0), input_ids.device
        combined, num_prefix, _ = self._embed_inputs(images, input_ids, states)
        mask = torch.cat([torch.ones(B, num_prefix, device=device, dtype=attention_mask.dtype),
                          attention_mask], dim=1)
        num_extra = 0
        if extra_embeds is not None:
            extra = self.language.project_in(extra_embeds) if self.language is not None else extra_embeds
            combined = torch.cat([combined, extra.to(combined.dtype)], dim=1)
            mask = torch.cat([mask, torch.ones(B, extra.size(1), device=device, dtype=mask.dtype)], dim=1)
            num_extra = extra.size(1)
        input_embeds = combined
        if self.language is None:
            positions = torch.arange(combined.size(1), device=device)
            combined = combined + self.pos_embed(positions).unsqueeze(0)

        captured = {}
        handle = None
        if capture_layer is not None:
            def grab(_module: nn.Module, _inputs: tuple, output: Any) -> None:
                """Forward hook: remember the captured layer's output."""
                captured["h"] = output[0] if isinstance(output, tuple) else output
            handle = self.decoder_layers()[capture_layer].register_forward_hook(grab)
        try:
            if self.language is not None:
                logits, hidden = self.language(combined, mask, self.sequence_token_ids(input_ids, num_prefix, num_extra))
            else:
                hidden = self.decoder_transformer(combined, pad_mask=mask)
                logits = self.lm_head(hidden)
        finally:
            if handle is not None:
                handle.remove()
        return {
            "hidden": hidden, "mask": mask, "logits": logits, "num_prefix": num_prefix,
            "text_len": input_ids.size(1), "num_extra": num_extra, "input_embeds": input_embeds,
            "layer_hidden": captured.get("h"),
        }

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                states: torch.Tensor, image_mask: torch.Tensor | None = None,
                ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """
        Forward pass with interleaved multi-image, state-injection, and
        action-decoding support.

        The input_ids sequence contains four kinds of special tokens:
          • <image>  (image_token_id)  — replaced by ViT patch embeddings
          • <state>  (state_token_id)  — replaced by StateEncoder output
          • <halo_action> (action_token_id) — used *after* the transformer to
            extract hidden states for the flow matching action decoder
          • <halo_world_video> (world_video_token_id) — used *after* the transformer
            to extract hidden states for conditioning the DiT video predictor

        Args:
            images:         [B, N_img, 3, H, W]  — N_img images per sample.
            input_ids:      [B, seq_len]          — token ids with <image>, <state>,
                            <halo_action>, and <halo_world_video> placeholders.
            attention_mask: [B, seq_len]          — 1 for real tokens, 0 pad.
            states:         [B, N_state, state_dim] — proprioceptive states,
                            one per <state> token.
            image_mask:     [B, N_img] optional — 1 for real image slots (padding 0).
                            Used when pooling **past** frames for DiT conditioning from
                            decoder states at image-patch positions (same order as ``images``).

        Returns:
            logits:                    [B, total_len, vocab_size] — next-token logits.
            action_hiddens:            [B, n_action_tokens, emb_dim]
                                       — transformer hidden states at <halo_action>
                                       positions; conditioning for the flow decoder.
                                       None if no <halo_action> tokens are present.
            visual_context_emb:        [B, emb_dim] or None — pooled decoder hidden
                                       states at image-patch positions (post-transformer,
                                       pre-lm_head); past-frame mean for DiT conditioning.
                                       None if visual DiT is disabled.
            world_video_query_hiddens: [B, n_world_video_tokens, emb_dim] or None
                                       — transformer hidden states at <halo_world_video>
                                       positions; used as query conditioning for the DiT
                                       video predictor. None if no such tokens present.
        """

        B = input_ids.size(0)
        device = input_ids.device
        network_config = self.config
        action_token_id = network_config.action_token_id        # e.g. 151666
        world_video_token_id = network_config.world_video_token_id  # e.g. 151668

        # Steps 1-5 (images, states, text, state injection, concatenation): shared with
        # encode_sequence(), which the dual-system VLA uses.
        combined_embeds, num_prepended, N_img = self._embed_inputs(images, input_ids, states)

        # ------------------------------------------------------------------ #
        # 6. POSITIONAL EMBEDDINGS — add learned positions over the full
        #    combined sequence.
        # ------------------------------------------------------------------ #
        if self.language is None:  # a pretrained LLM adds its own (rotary / learned) positions
            total_len = combined_embeds.size(1)
            pos_emb = self.pos_embed(torch.arange(total_len, device=device)).unsqueeze(0)
            combined_embeds = combined_embeds + pos_emb

        # ------------------------------------------------------------------ #
        # 7. DECODER TRANSFORMER — causal self-attention + MoE FFN.
        # ------------------------------------------------------------------ #
        # Build full padding mask covering prepended image patches + text tokens.
        # Image patch slots are always valid (ones); text mask comes from the caller.
        img_pad = torch.ones(B, num_prepended, device=device, dtype=attention_mask.dtype)
        pad_mask_full = torch.cat([img_pad, attention_mask], dim=1)  # [B, total_len]

        if self.language is not None:
            # logits from the LLM's head; hidden states mapped back to emb_dim for the heads below.
            logits, transformer_out = self.language(
                combined_embeds, pad_mask_full, self.sequence_token_ids(input_ids, num_prepended))
        else:
            transformer_out = self.decoder_transformer(combined_embeds, pad_mask=pad_mask_full)

        # --- Visual DiT conditioning: same **positions** as prepended image patches,
        #     but **after** transformer blocks (and final LN), **before** lm_head.
        visual_context_emb = None
        if self.visual_predictor is not None and N_img >= 1 and num_prepended > 0:
            patches_per_img = num_prepended // N_img
            img_dec = transformer_out[:, :num_prepended, :]  # [B, N_img * P, D]
            frame_stack = img_dec.view(B, N_img, patches_per_img, -1).mean(dim=2)
            visual_context_emb = pool_past_frame_embeddings_for_visual_dit(
                frame_stack, image_mask
            )

        # ------------------------------------------------------------------ #
        # 8. LANGUAGE HEAD — project every position to vocab logits.
        #    logits → [B, total_len, vocab_size]
        # ------------------------------------------------------------------ #
        if self.language is None:
            logits = self.lm_head(transformer_out)

        # ------------------------------------------------------------------ #
        # 9. ACTION HIDDEN STATES — find <halo_action> token positions in the
        #    *original* input_ids (offset by the prepended image patches).
        #    Extract the transformer hidden states at those positions and
        #    return them as conditioning for the flow matching decoder.
        #
        #    action_hiddens → [B, n_action_tokens, emb_dim]
        #    Returns None if no <halo_action> tokens are present.
        # ------------------------------------------------------------------ #
        action_mask = (input_ids == action_token_id)                # [B, seq_len]
        n_action_tokens = action_mask.sum(dim=1).max().item()  # scalar

        # ------------------------------------------------------------------ #
        # 9b. EXTRACT ACTION HIDDEN STATES — collect transformer hidden states
        #     at <halo_action> positions.  Instead of decoding actions directly
        #     with an MLP, we return these hidden states as *conditioning*
        #     for the flow matching decoder.  Flow matching loss is computed
        #     separately via  compute_flow_loss()  and actions are sampled
        #     at inference via  sample_actions().
        # ------------------------------------------------------------------ #
        action_hiddens = None
        if n_action_tokens > 0:
            action_hidden_list = []
            for b in range(B):
                action_positions = action_mask[b].nonzero(as_tuple=True)[0]  # in input_ids
                # Offset by prepended image patches to index into transformer_out
                action_positions = action_positions + num_prepended
                hiddens = transformer_out[b, action_positions]              # [n_act, emb_dim]
                # Pad if this sample has fewer <halo_action> tokens than the batch max
                if hiddens.size(0) < n_action_tokens:
                    pad = torch.zeros(
                        n_action_tokens - hiddens.size(0), network_config.emb_dim,
                        device=device,
                    )
                    hiddens = torch.cat([hiddens, pad], dim=0)
                action_hidden_list.append(hiddens)

            # action_hiddens: [B, n_action_tokens, emb_dim]
            # These serve as conditioning vectors for the flow decoder.
            action_hiddens = torch.stack(action_hidden_list, dim=0)
        else:
            logger.warning("action_hiddens is None: no <halo_action> tokens found in input_ids")

        # ------------------------------------------------------------------ #
        # 10. WORLD VIDEO QUERY HIDDEN STATES — find <halo_world_video> token
        #     positions in the *original* input_ids (offset by the prepended
        #     image patches) and extract the transformer hidden states there.
        #     These serve as query embeddings for conditioning the DiT video
        #     predictor, analogous to action_hiddens for the flow decoder.
        #
        #     world_video_query_hiddens → [B, n_world_video_tokens, emb_dim]
        #     Returns None if no <halo_world_video> tokens are present.
        # ------------------------------------------------------------------ #
        world_video_mask = (input_ids == world_video_token_id)      # [B, seq_len]
        n_world_video_tokens = world_video_mask.sum(dim=1).max().item()

        world_video_query_hiddens = None
        if n_world_video_tokens > 0:
            wv_hidden_list = []
            for b in range(B):
                wv_positions = world_video_mask[b].nonzero(as_tuple=True)[0]
                wv_positions = wv_positions + num_prepended
                hiddens = transformer_out[b, wv_positions]          # [n_wv, emb_dim]
                if hiddens.size(0) < n_world_video_tokens:
                    pad = torch.zeros(
                        n_world_video_tokens - hiddens.size(0), network_config.emb_dim,
                        device=device,
                    )
                    hiddens = torch.cat([hiddens, pad], dim=0)
                wv_hidden_list.append(hiddens)

            # world_video_query_hiddens: [B, n_world_video_tokens, emb_dim]
            world_video_query_hiddens = torch.stack(wv_hidden_list, dim=0)

        return logits, action_hiddens, visual_context_emb, world_video_query_hiddens




