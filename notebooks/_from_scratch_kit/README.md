# From-scratch notebook kit

Every notebook in the Hale repositories has a **from-scratch twin** in a `from_scratch/` folder next to it
(the full list is in `notebooks/FROM_SCRATCH.md`). A twin rebuilds the same model or algorithm in
**pure Python / PyTorch**, one step at a time, for readers who are new to the topic. Each step has:

1. a plain-language explanation of **what** the step does, **why** it is there, and the shape arithmetic;
2. a **Mermaid diagram** of that step;
3. the step's code, a few lines at a time;
4. a **picture** of the tensor it produces;
5. an **animation** wherever motion helps (sliding kernels, attention rows, denoising steps …).

The twin then assembles the full model from those steps, records a picture after **every layer**
(`StepRecorder`), runs sanity checks, and finishes with a guarded cell that compares the result with the
repository's real implementation, when that implementation is importable.

## Layout

```text
notebooks/_from_scratch_kit/
├── stepviz/      # picture + animation kit (canonical copy in HaleBlocks; synced; do not edit copies)
│   ├── _core.py      # colours, Animator (embedded GIFs), detached figures
│   ├── tensors.py    # images, feature maps, kernels, heatmaps, histograms, curves
│   ├── ops.py        # hand-computed animations: conv, pooling, patchify, attention
│   ├── recorder.py   # StepRecorder: a picture after every layer of any nn.Module
│   └── diagrams.py   # tensor-journey figure, block diagrams with skip paths
├── sources/      # <original stem>.md -> ../<dir>/from_scratch/<stem>_from_scratch.ipynb
│   └── shared/   # (HaleBlocks only) notebooks that exist in several repos, e.g. backbones
└── build.py      # canonical copy in HaleBlocks; builds every repo
```

## Building

```bash
# from the Music/ folder (any Python with nbformat + nbclient)
Hale-VLM/.venv/bin/python HaleBlocks/notebooks/_from_scratch_kit/build.py            # all
Hale-VLM/.venv/bin/python HaleBlocks/notebooks/_from_scratch_kit/build.py resnet     # one source
Hale-VLM/.venv/bin/python HaleBlocks/notebooks/_from_scratch_kit/build.py --list     # plan only
```

Each notebook runs with its repository's virtualenv (falling back to Hale-VLM's) and
`PYTHONPATH=<repo>/src:HaleBlocks/src`. A failing cell, such as an `assert` that checks a hand-computed
step against PyTorch, fails the build. Animations are embedded in the notebooks, so a copied notebook
is always complete.
