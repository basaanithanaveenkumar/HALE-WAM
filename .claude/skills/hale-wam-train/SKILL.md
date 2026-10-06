---
name: hale-wam-train
description: Train, resume, visualise and debug HALE-WAM on EO-Data1.5M or AIRoA MoMa, including gradient accumulation, AMP, loss weights and TensorBoard. Use when asked to run training, tune loss weights, fix NaNs or OOM, or render GIFs of predicted actions and future frames.
---

# Training HALE-WAM

## Objective

```
total = CE(assistant text)
      + action_loss_weight · FlowMatchingMSE(v_θ(x_t, t, h_action), x₁ − x₀)
      + visual_loss_weight · [ CFM_rgb + perceptual·VGG + ssim·(1−SSIM) + temporal·smooth ]
```

`visual_loss_weight` defaults to the config value (8.0) when `--visual_loss_weight` is unset.

## Datasets

| `--dataset` | Source | Needs |
|---|---|---|
| `eo` (default) | `IPEC-COMMUNITY/EO-Data1.5M` via the HF Hub, `--subset` picks 1 of 17 subsets | internet / HF login |
| `moma` | local clone of `airoa-org/airoa-moma` (`episodes.jsonl` + MP4s) | `--moma_data_root`, `git lfs pull` |

MoMa samples `--moma_num_frames` context frames plus `--num_predict_frames` future frames with
stride `--moma_frame_stride`; without `states/*.npy` the action and state tensors are zero-padded with masks.

## Recipes

```bash
# Overfit one clip (sanity check: GIFs should converge within a few thousand steps)
python scripts/train.py --dataset moma --moma_data_root $MOMA --max_samples 1 \
  --epochs 5000 --batch_size 1 --lr 1e-4 --vis_every 100

# Full run
python scripts/train.py --dataset moma --moma_data_root $MOMA --batch_size 4 --epochs 100 \
  --lr 3e-4 --moma_num_frames 5 --num_predict_frames 5 \
  --ckpt_dir checkpoints/run1 --tensorboard_dir runs/run1

# 8 GB GPU: effective batch 16
python scripts/train.py --dataset moma --moma_data_root $MOMA --batch_size 1 --grad_accum_steps 16 --lr 4e-4

# Resume (restores model, optimiser, scheduler and AMP scaler)
python scripts/train.py --dataset moma --moma_data_root $MOMA --resume checkpoints/run1/halo_vla_epoch10.pt --epochs 50

# Visualise a checkpoint
python scripts/visualize.py --checkpoint checkpoints/run1/model_best.pt --moma_data_root $MOMA \
  --output_dir vis_output --diffusion_steps 50
```

Monitor with `tensorboard --logdir runs`. `--keep_ckpts N` keeps only the last N checkpoints.

## Debugging

| Symptom | Likely cause / fix |
|---|---|
| Blurry future frames | raise `--ssim_weight` / `--perceptual_weight`; more `dit_num_sample_steps`; use CFG `guidance_scale > 1` at inference |
| All predicted frames identical | world-action tokens collapsed: check `frame_pos_proj` is used and `num_predict_frames` matches the data |
| `act_loss` flat at ~2 | `action_dim`/`action_chunk_size` don't match the data, or `action_mask` is all zeros (MoMa without `states/*.npy`) |
| OOM | lower `--batch_size` and use `--grad_accum_steps`; keep `gradient_checkpointing=True`; fewer `--moma_num_frames` (each adds 196 tokens) |
| dtype errors in VGG loss under AMP | see "Bug fixes" in `docs/world_model.md` |
| Warning "action_hiddens is None" | the text contains no `<halo_action>` token, so the flow loss is 0 for that batch |
