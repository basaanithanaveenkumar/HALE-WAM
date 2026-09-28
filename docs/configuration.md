# Configuration

`HaloVLMConfig` (`config/model_config.py`) holds every model hyper-parameter.

## Backbone

| Field | Default | Meaning |
|---|---|---|
| `vocab_size` | 49156 | cosmo2 tokenizer (49,152) + 4 special tokens |
| `image_token_id` / `action_token_id` / `state_token_id` / `world_video_token_id` | 49152 / 49153 / 49154 / 49155 | `<image>`, `<halo_action>`, `<state>`, `<halo_world_video>` |
| `emb_dim` | 512 | model width |
| `emb_factor_dim` | 128 | reserved for factored embeddings, **not implemented yet** |
| `img_size`, `patch_size` | 224, 16 | 196 patches per frame |
| `vit_num_layers`, `vit_num_heads` | 6, 16 | dense ViT |
| `dec_num_layers`, `dec_num_heads` | 8, 16 | causal decoder |
| `use_moe`, `moe_num_routed_experts`, `moe_top_k`, `moe_num_shared_experts`, `moe_hid_scale` | True, 6, 2, 2, 1.2 | DeepSeekMoE FFN |
| `max_position_embeddings` | 2000 | learned positions |
| `gradient_checkpointing` | True | recompute decoder activations |

## Action (flow matching)

| Field | Default | Meaning |
|---|---|---|
| `action_dim` | 7 | per-step action size (the training CLI defaults to 32) |
| `action_chunk_size` | 16 | steps per `<halo_action>` token |
| `flow_hidden_dim` | 1024 | velocity MLP width |
| `flow_time_embed_dim` | 128 | sinusoidal time embedding |
| `flow_num_ode_steps` | 24 | Euler steps at inference |
| `state_dim`, `state_hidden_dims` | 32, (256, 512) | state encoder |

## World model (DiT)

| Field | Default | Meaning |
|---|---|---|
| `enable_visual_dit` | True | build the DiT head |
| `num_visual_predict_frames` | 5 | future frames (one world-action token each) |
| `dit_patch_size` | 8 | 784 tokens at 224 px |
| `dit_hidden_size`, `dit_depth`, `dit_num_heads`, `dit_mlp_ratio` | 512, 8, 8, 4.0 | DiT size |
| `dit_time_freq_dim` | 512 | timestep frequency embedding |
| `dit_max_resolution` | 224 | pixel resolution |
| `dit_num_sample_steps` | 75 | sampling steps |
| `visual_loss_weight` | 8.0 | λ_v in the total loss |
| `depth_channels`, `flow_channels`, `dit_depth_*`, `dit_flow_*` | — | reserved for depth / optical-flow heads |

## Training flags (`scripts/train.py`)

| Flag | Default | Meaning |
|---|---|---|
| `--dataset` | `eo` | `eo` or `moma` |
| `--subset` | `interleave-temporal` | EO-Data1.5M subset |
| `--moma_data_root`, `--moma_camera` | —, `head` | MoMa clone and camera |
| `--moma_num_frames`, `--num_predict_frames`, `--moma_frame_stride` | 5, 5, 25 | context / future frames, stride |
| `--batch_size`, `--grad_accum_steps` | 1, 1 | effective batch = product |
| `--max_seq_len`, `--max_samples` | 256, 2000 | |
| `--action_dim`, `--state_dim`, `--action_chunk_size` | 32, 32, 16 | |
| `--epochs`, `--lr`, `--weight_decay`, `--grad_clip` | 5, 1e-4, 0.01, 1.0 | AdamW |
| `--action_loss_weight`, `--visual_loss_weight` | 1.0, config | loss weights |
| `--perceptual_weight`, `--ssim_weight`, `--temporal_weight` | 0.1, 0.4, 0.2 | anti-blur losses (0 = off) |
| `--resume` | — | checkpoint path |
| `--save_every`, `--keep_ckpts`, `--ckpt_dir` | 100, 3, `checkpoints` | |
| `--tensorboard_dir`, `--vis_every` | `runs`, 10 | logging |
