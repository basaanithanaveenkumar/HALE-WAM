---
name: hale-wam-world-model
description: Modify or extend the HALE-WAM DiT world model (future-frame prediction) or the flow-matching action head — conditioning, samplers (Euler/Heun), classifier-free guidance, auxiliary losses, adding depth/optical-flow heads or moving the DiT to VideoVAE latents. Use for any change to dit_frame_prediction.py, DiT.py or flow_action_decoder.py.
---

# Working on the world model and flow head

Read `docs/world_model.md` first: it is the authoritative write-up, including six
past bug fixes you should not reintroduce.

## Conditioning path for frame i (`RGBFramePredictor._create_all_frame_contexts`)

1. Query = `world_video_query_hiddens[:, i]` (decoder hidden state at the i-th
   `<halo_world_video>` token, through `wv_query_proj`) or the learnable
   `world_action_tokens[frame_offset + i]`.
2. Cross-attention: Q = query, K/V = the context embedding, then residual + LN + FFN + LN.
3. `+ frame_pos_proj(sinusoidal(frame_offset + i))` gives an unambiguous frame identity.
4. `+ context_frame_enc(last_observed_frame)` adds a pixel-level scene anchor (dropout 0.2).
5. CFG: with p = 0.1 in training, replace the result with the learned `cfg_null_context`.
6. In the DiT: `c = sigmoid(W_gate·ctx) ⊙ t_emb + W_ctx·ctx` drives adaLN-Zero in every block.

Keep all six when refactoring. Removing step 3 or 4 is the usual cause of identical
frames or scene drift.

## Invariants

- Flow convention: `x_t = (1−t)·x₀ + t·x₁`, target `v* = x₁ − x₀`, `t ~ U(0,1)`, and
  sampling goes from t=0 (noise) to t=1 (data). Actions and frames use the same convention.
- The Heun sampler must never evaluate the model at exactly `t = 1` (clamp to `1 − 1e-5`).
- `x1_hat = x_t + (1−t)·v_pred` is the clean-frame estimate used by the SSIM/VGG losses.
  Clamp it to `[0, 1]`.
- Don't name a local variable `F` inside modules that import `torch.nn.functional as F`
  (bug #4 in the docs).
- New tensors created inside `forward` must use the input's `device`/`dtype` (bugs #1 and #3).

## Recipes

- **Add a depth or optical-flow head.** The config already has `depth_channels`,
  `flow_channels`, `dit_depth_in_channels` and loss weights. Build a second
  `DiT(make_dit_config(config, 1, 1))`, reuse the same per-frame context, and add its
  CFM loss to `compute_visual_prediction_loss`'s returned dict.
- **Latent DiT.** Encode frames with `VideoVAE.encode` (8× down, 4 channels), set
  `dit_max_resolution=28`, `dit_patch_size=2` and `dit_in_channels=4`, and decode samples
  with `VideoVAE.decode`. Pretrain or freeze the VAE first.
- **Better action sampler.** `sample_actions` is Euler. A Heun variant mirrors
  `predict_future_frames`. Keep `num_steps` configurable via `flow_num_ode_steps`.
- **Stronger flow head.** `FlowActionDecoder` is a 4-layer ReLU MLP on
  `[x_t, t_emb, cond]`. A DiT-style or cross-attention head must keep the call
  `flow_decoder(x_t[B, chunk·dim], t[B], cond[B, emb_dim]) -> [B, chunk·dim]`.

## Test

```bash
PYTHONPATH=.:src/Halo_VLA pytest tests/test_visual_prediction_multi_step.py -q
```
