# Architecture

All diagrams are Mermaid and render on GitHub. They match Section 3 of the
[paper](../paper/main.tex) and the [project page](../project-page/index.html). For
implementation detail on the world model, see [world_model.md](world_model.md).

## 1. System overview: perceive, reason, act, imagine

```mermaid
flowchart TB
  F["N context frames<br/>[B, N, 3, 224, 224]"] --> VIT["ViT (6 dense layers, patch 16)<br/>+ MLP projector"]
  T["ChatML tokens<br/>&lt;image&gt; &lt;state&gt; &lt;halo_action&gt; &lt;halo_world_video&gt;"] --> EMB["token embedding 49,156 × 512"]
  S["state [B, N_s, 32]"] --> SE["StateEncoder MLP"]
  VIT --> SEQ["[ N·196 patches ; text with state embeddings in place ] + positions"]
  EMB --> SEQ
  SE --> SEQ
  SEQ --> DEC["Causal decoder × 8<br/>MHA (padding mask) + DeepSeekMoE (6 routed + 2 shared, top-2)<br/>gradient checkpointing"]
  DEC -->|"every position"| LM["LM head → text"]
  DEC -->|"h at &lt;halo_action&gt;"| FM["Flow-matching head<br/>v(x_t, t, h) → 16 × action_dim"]
  DEC -->|"h at &lt;halo_world_video&gt;"| DIT["DiT world model"]
  DEC -->|"mean of patch states (past frames)"| DIT
  F -->|"last observed frame"| DIT
  DIT --> FUT["future frames<br/>[B, F, 3, 224, 224]"]
```

## 2. Flow-matching action head

```mermaid
flowchart LR
  subgraph Training
    X0["x₀ ~ N(0, I)"] --> XT["x_t = (1−t)·x₀ + t·x₁"]
    X1["x₁ = GT chunk (16 × d_a)"] --> XT
    TT["t ~ U(0,1)"] --> XT
    XT --> V["v_θ = MLP([x_t, sin(t), h_act])<br/>4 layers × 1024, ReLU"]
    V --> L["MSE(v_θ, x₁ − x₀)"]
  end
  subgraph Inference
    N0["x ← noise"] --> LOOP["repeat 24×: x ← x + v_θ(x, t, h_act)·Δt"]
    LOOP --> A["action chunk"]
  end
```

## 3. Per-frame conditioning for the DiT world model

```mermaid
flowchart TB
  Q["query_i = W·h_wv,i  or  learned world_action_token[f+i]"] --> CA["Cross-attention<br/>Q = query_i, K/V = visual context"]
  CA --> FFN["residual + LN → FFN → residual + LN"]
  FFN --> ADD1["+ W_p · sinusoidal(f + i)<br/>(frame identity)"]
  ADD1 --> ADD2["+ Dropout₀.₂( PixelEnc(last frame) )<br/>(scene anchor)"]
  ADD2 --> CFG{"training and<br/>rand &lt; 0.1?"}
  CFG -->|yes| NULL["learned null context"]
  CFG -->|no| CI["c_i"]
  NULL --> GATE
  CI --> GATE["c̃ = σ(W_g·c_i) ⊙ emb(t) + W_c·c_i"]
  GATE --> ADALN["adaLN-Zero in each of 8 DiT blocks"]
```

## 4. DiT denoiser and sampler

```mermaid
flowchart LR
  XN["x_t [3, 224, 224]"] --> PE["patchify 8×8 → 784 tokens<br/>+ positions"]
  PE --> B1["DiT block × 8<br/>(adaLN shift/scale/gate)"]
  B1 --> UP["final LN → linear → unpatchify"]
  UP --> VEL["velocity v [3, 224, 224]"]
  VEL --> HEUN["Heun step:<br/>x̃ = x + Δt·v₁;  x ← x + Δt/2·(v₁ + v₂)<br/>CFG: v = v_∅ + w·(v_c − v_∅)"]
  HEUN -->|"t &lt; 1"| XN
  HEUN -->|"t = 1"| OUT["predicted frame → appended to context<br/>for the next frame"]
```

## 5. Training step and losses

```mermaid
sequenceDiagram
  participant D as Dataloader (EO / MoMa)
  participant M as HaloVLM
  participant L as Losses
  participant O as AdamW + AMP scaler
  D->>M: images, input_ids, attention_mask, states, image_mask
  M-->>L: logits, action_hiddens, visual_context_emb, world_video_hiddens
  D->>L: labels, actions, action_mask, future_frames
  L->>L: CE(text, offset by N·196)
  L->>L: flow-matching MSE on action chunks
  L->>L: CFM(frames) + 0.1·VGG + 0.4·(1−SSIM) + 0.2·temporal on x̂₁ = x_t + (1−t)·v
  L->>O: total = CE + 1·act + 8·vis   (÷ grad_accum_steps)
  O->>M: unscale, clip 1.0, step every grad_accum_steps
```

## Parameter budget (default config, measured)

| Module | Params |
|---|---|
| ViT | 10.0M |
| Image projector | 0.23M |
| Token embedding | 25.2M |
| Position embedding | 1.0M |
| Decoder (MoE) | 68.8M |
| LM head | 25.2M |
| State encoder | 0.40M |
| Flow-matching head | 3.0M |
| DiT world model | 42.5M |
| **Total** | **176.3M** |
