"""An :class:`IOSignature` whose sampled ``input_ids`` follow Halo-VLA's prompt template."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from hale_core.deploy.core.io_signature import IOSignature


@dataclass(frozen=True)
class PromptIOSignature(IOSignature):
    """Random token ids are not a valid VLA prompt: without ``<image>`` / ``<state>`` /
    ``<halo_action>`` placeholders the encoder would select nothing and parity checks would
    compare zeros. This subclass writes the placeholders into every sample (reference inputs,
    parity corners and calibration batches alike), mirroring the dataloaders' template::

        [bos] <image>xN <state>xS  ...instruction text...  <halo_world_video>xF <halo_action>xK

    Attributes:
        prompt_input: name of the token-id input.
        head_tokens: ids written right after position 0.
        tail_tokens: ids written at the end of the sequence.
    """

    prompt_input: str = "input_ids"
    head_tokens: tuple[int, ...] = field(default=())
    tail_tokens: tuple[int, ...] = field(default=())

    @property
    def min_prompt_length(self) -> int:
        """Shortest sequence that holds every placeholder plus one text token."""
        return 1 + len(self.head_tokens) + len(self.tail_tokens) + 1

    def sample(self, which: str = "opt", generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
        out = super().sample(which, generator)
        ids = out[self.prompt_input].clone()
        head = torch.tensor(self.head_tokens, dtype=ids.dtype)
        tail = torch.tensor(self.tail_tokens, dtype=ids.dtype)
        if ids.size(1) < self.min_prompt_length:
            raise ValueError(f"{self.prompt_input} length {ids.size(1)} cannot hold the prompt template "
                             f"(needs >= {self.min_prompt_length})")
        if head.numel():
            ids[:, 1:1 + head.numel()] = head
        if tail.numel():
            ids[:, ids.size(1) - tail.numel():] = tail
        out[self.prompt_input] = ids
        return out
