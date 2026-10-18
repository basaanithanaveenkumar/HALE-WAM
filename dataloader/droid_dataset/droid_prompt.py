"""DROID chat prompts: tokenizer setup, instruction text, tokenisation and loss masking."""

from __future__ import annotations

import torch

from config.tokens import get_all_custom_tokens, get_token


class DroidPromptMixin:
    """DROID chat prompts: tokenizer setup, instruction text, tokenisation and loss masking.

    Mixed into :class:`DROIDDataset`; relies on the attributes its ``__init__`` sets.
    """

    def _setup_tokenizer(self, tokenizer) -> None:
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.tokenizer_name,
                trust_remote_code=True,
            )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        extra = [tok for tok in get_all_custom_tokens() if tok not in self.tokenizer.get_vocab()]
        if extra:
            self.tokenizer.add_special_tokens({"additional_special_tokens": extra})

        self._image_tok = get_token("image_token")
        self._action_tok = get_token("action_token")
        self._state_tok = get_token("state_token")
        self._world_video_tok = get_token("world_video_token")
        self.pad_token_id = self.tokenizer.pad_token_id

    def _build_text(self, task: str, num_images: int) -> tuple[str, str]:
        """
        Build the (user, assistant) string pair for one sample.

        One ``<image>`` token per context frame, one ``<state>`` token,
        one ``<halo_world_video>`` per future frame, one ``<action>`` token.
        """
        img_tags = " ".join([self._image_tok] * num_images)
        user_body = (
            f"{img_tags} {self._state_tok} Task: {task}\n"
            "Predict the robot motion and describe the next action."
        )
        wv_tokens = self._world_video_tok * self.cfg.num_predict_frames
        assistant_body = f"Executing: {task}\n{wv_tokens}{self._action_tok}"
        return user_body, assistant_body

    def _tokenise(
        self, user_body: str, assistant_body: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from config import HaloVLMConfig

        model_cfg = HaloVLMConfig()
        system_msg = f"{self.cfg.system_prefix}{model_cfg.system_prompt}{self.cfg.turn_end}"
        human_msg = f"{self.cfg.human_prefix}{user_body}{self.cfg.turn_end}"
        asst_msg = f"{self.cfg.assistant_prefix}{assistant_body}{self.cfg.turn_end}"
        full_text = system_msg + human_msg + asst_msg

        enc = self.tokenizer(
            full_text,
            max_length=self.cfg.max_seq_len,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        # Mask system + user spans so loss only trains on the assistant reply
        parts = [system_msg, human_msg, asst_msg]
        is_asst = [False, False, True]
        labels = self._mask_non_assistant(full_text, parts, is_asst, labels)
        labels[attention_mask == 0] = -100

        return input_ids, attention_mask, labels

    def _mask_non_assistant(
        self,
        full_text: str,
        parts: list[str],
        is_asst: list[bool],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        cursor = 0
        for part, assistant in zip(parts, is_asst):
            start = full_text.find(part, cursor)
            if start == -1:
                continue
            end = start + len(part)
            if not assistant:
                prefix_ids = self.tokenizer(
                    full_text[:end],
                    max_length=self.cfg.max_seq_len,
                    truncation=True,
                    add_special_tokens=False,
                )["input_ids"]
                start_ids = self.tokenizer(
                    full_text[:start],
                    max_length=self.cfg.max_seq_len,
                    truncation=True,
                    add_special_tokens=False,
                )["input_ids"]
                tok_start = len(start_ids)
                tok_end = len(prefix_ids)
                if tok_end <= labels.numel():
                    labels[tok_start:tok_end] = -100
            cursor = end
        return labels
