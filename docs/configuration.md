# Configuration

A run is configured in two layers:

* **YAML run configs** (`configs/*.yaml`, schema `config/run_config.py::VLARunConfig`). They use
  the same layout and loader as Hale-VLM: `model.vision`, `model.llm`, `model.dual_system`, ...,
  with `inherits:`. `scripts/train.py --config <yaml>` reads one, and command-line flags override it.
* **`HaloVLMConfig`** (`config/model_config.py`), the model dataclass every run builds. YAML keys
  without a CLI flag (architecture sizes) are set on it directly, as is `--model_option KEY=VALUE`.

## YAML run configs

```bash
python scripts/train.py --config configs/base.yaml                      # every default, spelled out
python scripts/train.py --config configs/overfit.yaml                   # CPU smoke test
python scripts/train.py --config configs/dual_system/openhelix.yaml --dual_system_option async_steps=60
python scripts/train.py --config configs/examples/llm_qwen3_5_4b_base.yaml --epochs 20
```

```yaml
inherits: base.yaml               # up to two levels of inheritance (child -> parent -> grandparent)
variant: my_run
device: cuda
model:
  emb_dim: 512
  vision:   {encoder: siglip, projector_type: qformer, qformer: {num_queries: 32}}
  llm:      {backbone: gemma4-e4b, use_lora: true, lora_r: 16, attn_implementation: eager}
  action:   {action_dim: 7, state_dim: 8, chunk_size: 16}
  world_model: {enabled: true, num_predict_frames: 5}
  dual_system: {preset: openhelix, async_steps: 10}
train:  {epochs: 20, batch_size: 8, lr: 1.0e-4, checkpoint: {dir: checkpoints/run1}, ray: {enabled: false}}
data:   {dataset: droid, droid: {data_root: /data/droid_100}, webdataset: {enabled: false}}
eval:   {val_fraction: 0.05}
logging: {tensorboard_dir: runs}
viz:    {every_n_steps: 100}
```

| section | contents | reaches |
|---|---|---|
| `model` | `emb_dim`, `decoder`, `moe`, `vision` (`vit` / `qformer` / `gated_cross_attention` / `local_backbone`), `llm`, `action` (+ `flow`), `world_model` (+ `dit`), `dual_system` | CLI flags + `HaloVLMConfig` |
| `train` | epochs, batch size, optimiser, `resume`, `checkpoint`, `ray` | CLI flags |
| `data` | `dataset` (`eo` / `moma` / `droid`), `droid`, `moma`, `webdataset` | CLI flags |
| `eval`, `logging`, `viz` | validation split, TensorBoard, training GIFs | CLI flags |

Rules:

* **Validation.** Every key is validated: unknown keys, unknown LLM backbones, bad encoders and
  invalid dual-system combinations fail at load time.
* **Precedence.** Only keys present in the YAML chain are forwarded. Anything else keeps the
  script's or the dataclass's own default, and command-line flags override the YAML.
* **No drift.** `configs/base.yaml` equals the CLI defaults exactly, and the schema's defaults are
  tested against both the CLI and `HaloVLMConfig`.
* **`model.llm`.** Same fields as Hale-VLM's (`backbone`, `use_lora` > `freeze_llm` > full,
  `lora_*`, `checkpoint`). `backbone: null` keeps the built-in decoder.
* **`model.dual_system`.** `preset` (`none`, `basic`, `lcb`, `dp_vla`, `hirt`, `robodual`,
  `dexvla`, `openhelix`, `custom`) plus any preset field as an override. `--dual_system_option`
  flags merge on top. See `docs/dual_system.md`.
* **Provenance.** The resolved YAML (inheritance applied, plus the command line) is saved as
  `<ckpt_dir>/run_config.yaml`.

Shipped configs:

| file | what |
|---|---|
| `configs/base.yaml` | every field with its default |
| `configs/overfit.yaml` | small model, 16 samples, no DiT: smoke test |
| `configs/halo_vla_qformer.yaml`, `halo_vla_gated_xattn.yaml` | Q-Former / gated cross-attention projectors |
| `configs/halo_vla_flow_matching.yaml` | larger flow head, longer schedule, validation |
| `configs/dual_system/{basic,lcb,dp_vla,hirt,robodual,dexvla,openhelix,custom_middle_layer}.yaml` | dual-system presets |
| `configs/examples/llm_qwen3_5_4b_base.yaml`, `llm_gemma4_e4b.yaml`, `llm_cosmos_reason2_2b.yaml` | 2026 LLM backbones (multimodal text towers) |
| `configs/examples/ray_webdataset.yaml` | Ray Train + WebDataset shards |
| `configs/roadmap/` | specifications whose code paths do not exist yet (stage datasets, simulator eval, deployment, GWM); not loadable, see its README |

Needs HaleBlocks: `uv sync --extra config`.

## `HaloVLMConfig`

### Backbone

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

### Action (flow matching)

| Field | Default | Meaning |
|---|---|---|
| `action_dim` | 7 | per-step action size (the training CLI defaults to 32) |
| `action_chunk_size` | 16 | steps per `<halo_action>` token |
| `flow_hidden_dim` | 1024 | velocity MLP width |
| `flow_time_embed_dim` | 128 | sinusoidal time embedding |
| `flow_num_ode_steps` | 24 | Euler steps at inference |
| `state_dim`, `state_hidden_dims` | 32, (256, 512) | state encoder |

### World model (DiT)

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
| `--config` | — | YAML run config (above); flags override it |
| `--model_option KEY=VALUE` | — | set any `HaloVLMConfig` field (repeatable) |
| `--vision_encoder`, `--projector_type`, `--num_query_tokens` | `vit`, `mlp`, — | vision tower / projector |
| `--llm_backbone`, `--llm_finetune`, `--llm_lora_r`, `--llm_attn_implementation` | —, `lora`, 16, `sdpa` | pretrained LLM decoder (`docs/llm_backbone.md`) |
| `--dual_system`, `--dual_system_option KEY=VALUE` | `none` | dual-system VLA (`docs/dual_system.md`) |
| `--val_fraction`, `--eval_every` | 0, 1 | validation |
| `--ray*`, `--webdataset*` | off | distributed training (`docs/distributed.md`) |
