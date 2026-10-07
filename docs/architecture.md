# Architecture

Mermaid diagrams (render on GitHub). They match Section 3 of the
[paper](../paper/main.tex) and the [project page](../project-page/index.html).
For the world model implementation details, see [world_model.md](world_model.md).

---

## 1. End-to-end overview

One causal decoder reads camera patches, proprioceptive state, and language tokens — then three independent heads read specific hidden states to produce text, actions, and future frames.

```mermaid
flowchart TB
  subgraph INPUTS["Inputs (per forward pass)"]
    F["Camera frames\n[B, N, 3, 224, 224]"]
    S["Robot state\n[B, N_s, 32]\n(joint angles, gripper)"]
    T["Language tokens\n(ChatML-formatted instruction)"]
  end

  subgraph ENCODE["Encoding"]
    VIT["ViT image encoder\n6 dense transformer layers, patch=16\n→ [B·N, 196, 512] patch embeddings"]
    PROJ["MLP projector\n512 → 512\nscales visual features to decoder dim"]
    SE["StateEncoder MLP\n32 → 512\nembed proprioceptive state"]
    TOK_EMB["Token embedding\n49,156 vocab × 512\n(ALBERT-factored: 512→128 bottleneck)"]
  end

  subgraph SEQUENCE["Sequence assembly"]
    SEQ["[N·196 visual patches | text tokens with state injected]\n+ sinusoidal position embeddings\ntotal length S = N·196 + L_text"]
  end

  subgraph DECODER["Causal decoder — 8 layers"]
    MHA["Multi-head attention\n8 heads, dim 512\ncausal mask (padding-aware)"]
    MOE["DeepSeekMoE FFN\n6 routed + 2 shared experts\ntop-2 routing, expert dim 2048"]
    LN["RMSNorm between sub-layers"]
  end

  subgraph HEADS["Three output heads"]
    LMH["LM head\n512 → 49,156 logits\nevery position"]
    FMH["Flow-matching head\n4-layer MLP 512 → 1024 → action_dim\nreads hidden state at &lt;halo_action&gt; token"]
    DITH["DiT world model\nreads hidden states at &lt;halo_world_video&gt; tokens\n→ future RGB frames"]
  end

  F --> VIT --> PROJ --> SEQ
  S --> SE --> SEQ
  T --> TOK_EMB --> SEQ
  SEQ --> MHA --> MOE --> LN
  LN --> LMH & FMH & DITH
```

---

## 2. Vision encoder detail (ViT)

```mermaid
flowchart LR
  IMG["Single frame\n[B, 3, 224, 224]"]
  PATCH["Patch embedding\n14×14 = 196 patches\n3×16×16 → 512"]
  PE["Learnable position\nembedding\n[196, 512]"]
  ADD["Add positions"]
  L1["Transformer layer 1\nMHA (no causal mask) + FFN"]
  L2["Transformer layer 2"]
  L3["Transformer layer 3"]
  L4["Transformer layer 4"]
  L5["Transformer layer 5"]
  L6["Transformer layer 6"]
  OUT["196 patch tokens\n[B, 196, 512]"]

  IMG --> PATCH --> ADD
  PE --> ADD
  ADD --> L1 --> L2 --> L3 --> L4 --> L5 --> L6 --> OUT

  NOTE["Run independently per frame N\nthen concatenate:\n[B, N·196, 512]"]
```

---

## 3. DeepSeekMoE expert routing

```mermaid
flowchart LR
  H["Hidden state\n[B, S, 512]"]
  GATE["Gating network\n512 → num_experts logits\nsoftmax → expert scores"]
  TOP2["Top-2 selection\nkeep highest 2 scores,\nzero the rest"]
  subgraph SHARED["2 shared experts (always active)"]
    SE1["Expert S1\n512→2048→512"]
    SE2["Expert S2\n512→2048→512"]
  end
  subgraph ROUTED["6 routed experts (top-2 selected)"]
    E1["Expert 1"]
    E2["Expert 2"]
    E3["Expert 3"]
    E4["Expert 4"]
    E5["Expert 5"]
    E6["Expert 6"]
  end
  WGT["Weighted sum\noutput = Σ score_i · expert_i(h)\n+ shared_experts(h)"]

  H --> GATE --> TOP2
  TOP2 --> E1 & E2 & E3 & E4 & E5 & E6
  H --> SE1 & SE2
  E1 & E2 & E3 & E4 & E5 & E6 --> WGT
  SE1 & SE2 --> WGT
```

---

## 4. Flow-matching action head

The velocity MLP learns a straight-line trajectory in action space from noise to the ground-truth chunk.

```mermaid
flowchart LR
  subgraph TRAIN["Training"]
    X0["x₀ ~ N(0, I)\nrandom noise\n[B, 16, action_dim]"]
    X1["x₁ = ground truth\naction chunk\n[B, 16, action_dim]"]
    T_U["t ~ U(0, 1)\nrandom time step"]
    XT["x_t = (1-t)·x₀ + t·x₁\ninterpolate along\nstraight-line path"]
    TENC["time encoding\nsinusoidal(t) → [B, 64]"]
    HACT["action hidden\nh at &lt;halo_action&gt; token\n[B, 512]"]
    CAT["concat [x_t flat | sin(t) | h_act]"]
    MLP["velocity MLP\n4 layers × 1024, ReLU\n→ [B, 16 × action_dim]"]
    TARGET["target: x₁ − x₀\n(constant velocity field)"]
    LOSS["MSE(v_θ, target)"]

    X0 & X1 --> XT
    T_U --> XT & TENC
    XT & TENC & HACT --> CAT --> MLP
    MLP & TARGET --> LOSS
  end

  subgraph INFER["Inference (ODE integration)"]
    NOISE["x = x₀ ~ N(0, I)"]
    LOOP["for i in 0 … 23:\n  dt = 1 / 24\n  v = MLP(x, t_i, h_act)\n  x = x + v · dt"]
    ACTION["action chunk\n[B, 16, action_dim]"]
    NOISE --> LOOP --> ACTION
  end
```

---

## 5. DiT world model — per-frame conditioning

Each future frame gets a unique conditioning vector built from four stacked signals, which modulates every DiT block via adaLN-Zero.

```mermaid
flowchart TB
  subgraph COND["Per-frame conditioning (frame index i)"]
    HWV["World-video hidden\nh_wv,i at &lt;halo_world_video&gt;\nor learned world_action_token[f+i]\n[B, 512]"]
    XA["Cross-attention\nQ = W·h_wv,i\nK,V = visual context from ViT\n→ [B, 512]"]
    FFN["residual + LN → FFN → residual + LN"]
    FI["+ sinusoidal(f + i)\n× W_p  [B, 512]\n(frame identity embedding)"]
    PENC["+ Dropout₀.₂( PixelEnc(last_frame) )\n[B, 512]\n(pixel-level scene anchor)"]
    CFG{"training and\nrand < 0.1?"}
    NULL["learned null context\n[512]  (used in ~10% of steps)"]
    CI["context_i\n[B, 512]"]
    GATE["gated fusion\nc̃ = σ(W_g · c_i) ⊙ emb(t) + W_c · c_i\n[B, 512]"]
  end

  subgraph DIT["DiT — 8 blocks"]
    BLK["DiT block:\nadaLN-Zero(c̃) → self-attn → adaLN-Zero(c̃) → FFN\n(same c̃ for all 8 blocks of frame i)"]
    OUT["predicted frame\n[B, 3, 224, 224]"]
  end

  HWV --> XA --> FFN --> FI --> PENC --> CFG
  CFG -->|"yes (CFG)"| NULL --> GATE
  CFG -->|"no"| CI --> GATE
  GATE --> BLK --> OUT
```

---

## 6. Heun sampler (ODE integration for world model)

```mermaid
flowchart LR
  NOISE["x₀ ~ N(0, I)\n[B, 3, 224, 224]"]
  subgraph HEUN["Heun 2nd-order integration\nT steps (default 50)"]
    PRED1["k₁ = v_θ(xₜ, t, c)\n(evaluate velocity at tₙ)"]
    STEP1["x̂ₙ₊₁ = xₙ + k₁ · Δt\n(Euler predictor step)"]
    PRED2["k₂ = v_θ(x̂ₙ₊₁, t+Δt, c)\n(evaluate at corrected point)"]
    CORRECT["xₙ₊₁ = xₙ + (k₁ + k₂)/2 · Δt\n(average — halves truncation error)"]
  end
  FRAME["predicted frame\n[B, 3, 224, 224]"]

  NOISE --> PRED1 --> STEP1 --> PRED2 --> CORRECT
  CORRECT -->|"repeat T times"| PRED1
  CORRECT --> FRAME
```

---

## 7. Training step (all losses combined)

```mermaid
sequenceDiagram
  participant D as Dataloader
  participant M as HALE-WAM
  participant L as Loss module
  participant O as AdamW + AMP scaler

  D->>M: images [B,N,3,H,W], input_ids, attention_mask, states [B,N_s,32]
  D->>M: image_mask (which token positions hold visual patches)
  M-->>L: logits [B,S,V], action_hiddens [B,512], vis_ctx [B,196,512], wv_hiddens [B,F,512]
  D->>L: labels [B,S], actions [B,16,A], action_mask, future_frames [B,F,3,H,W]

  L->>L: CE loss on text tokens (offset by N·196 patch positions)
  L->>L: flow-matching MSE on action chunk (via action_hiddens)
  L->>L: x̂₁ = xₜ + (1-t)·v_θ  (clean-frame estimate — free from flow formula)
  L->>L: visual CFM(frames) = MSE(v_θ, x₁ - x₀)
  L->>L: + 0.1 × VGG perceptual on x̂₁
  L->>L: + 0.4 × (1 - SSIM) on x̂₁
  L->>L: + 0.2 × temporal smoothness on x̂₁
  L->>L: total = 1·CE + 1·act_flow + 8·visual  (÷ grad_accum_steps)

  O->>M: unscale gradients, clip norm ≤ 1.0
  O->>M: step every grad_accum_steps micro-batches
```

---

## 8. Token sequence layout

```mermaid
flowchart LR
  subgraph SEQ["Full input sequence fed to decoder"]
    V1["Visual patches\nframe 1\n196 tokens"]
    V2["Visual patches\nframe 2\n196 tokens"]
    VN["Visual patches\nframe N\n196 tokens"]
    IM["&lt;image&gt; marker"]
    ST["&lt;state&gt; + state embed"]
    USR["user turn tokens\n(instruction)"]
    ACT["&lt;halo_action&gt;"]
    WV["&lt;halo_world_video&gt;\n× F future frames"]
    AST["assistant turn\n(generated text)"]
  end

  V1 --> V2 --> VN --> IM --> ST --> USR --> ACT --> WV --> AST
```

LM loss is applied only to `AST` tokens. The `<halo_action>` hidden state feeds the flow-matching head; `<halo_world_video>` hidden states feed the DiT.
