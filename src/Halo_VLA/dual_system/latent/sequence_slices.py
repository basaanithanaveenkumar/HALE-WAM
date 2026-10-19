"""Index helpers locating the text span inside System 2's encoded sequence."""

from __future__ import annotations


def _text_slice(enc: dict) -> slice:
    """Positions of the instruction text: after the ``num_prefix`` image/state tokens."""
    return slice(enc["num_prefix"], enc["num_prefix"] + enc["text_len"])
