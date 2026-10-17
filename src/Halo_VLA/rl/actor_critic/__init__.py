"""Actor, Critic, and the λ-return for Dreamer-style latent imagination.

The actor and critic both operate on the RSSM **feature** ``[h, z]`` — i.e.
purely in latent space.  They are trained on *imagined* trajectories rolled out
by the world model, never on real pixels, which is what makes Dreamer so
sample-efficient.

* **Actor** ``π(a | feature)`` — a diagonal Gaussian over actions.
* **Critic** ``V(feature)``     — predicts the expected λ-return from a state.
* **λ-return** — a low-variance value target that interpolates between a 1-step
  bootstrap (λ=0) and a full Monte-Carlo return (λ=1)."""

from .actor import Actor  # noqa: F401
from .common import lambda_return  # noqa: F401
from .critic import Critic  # noqa: F401

__all__ = [
    "Actor",
    "Critic",
    "lambda_return",
]
