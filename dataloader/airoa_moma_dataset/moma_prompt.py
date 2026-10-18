"""AIRoA MoMa chat prompts: tokenizer setup, instruction text, tokenisation and loss masking."""

from __future__ import annotations

from typing import Any

import torch

from config.tokens import get_all_custom_tokens


class MomaPromptMixin:
    """AIRoA MoMa chat prompts: tokenizer setup, instruction text, tokenisation and loss masking.

    Mixed into :class:`AiroaMomaDataset`; relies on the attributes its ``__init__`` sets.
    """

    def _setup_tokenizer(self, tokenizer):
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

        additional_special: list[str] = []
        for tok in get_all_custom_tokens():
            if tok not in self.tokenizer.get_vocab():
                additional_special.append(tok)
        if additional_special:
            self.tokenizer.add_special_tokens({"additional_special_tokens": additional_special})

        from config.tokens import get_token

        self._image_tok = get_token("image_token")
        self._action_tok = get_token("action_token")
        self._state_tok = get_token("state_token")
        self._world_video_tok = get_token("world_video_token")
        self.pad_token_id = self.tokenizer.pad_token_id

    def _build_text(self, meta: dict[str, Any], num_images: int) -> tuple[str, str]:
        """User string (with placeholders) and assistant string."""
        tasks = meta.get("tasks") or []
        task_txt = tasks[0] if tasks else meta.get("short_horizon_task", "manipulation")
        if not isinstance(task_txt, str):
            task_txt = str(task_txt)

        img_tags = " ".join([self._image_tok] * num_images)
        user_body = (
            f"{img_tags} {self._state_tok} Task: {task_txt}\n"
            "Predict the robot's motion and describe the next step."
        )
        # One <halo_world_video> token per future frame to predict, so the transformer
        # emits exactly num_predict_frames query vectors for the DiT conditioning.
        wv_tokens = self._world_video_tok * self.cfg.num_predict_frames
        assistant_body = f"I will execute the task: {task_txt}\n{wv_tokens}{self._action_tok}"
        return user_body, assistant_body

    def _tokenise_turns(self, user_body: str, assistant_body: str):
        from config import HaloVLMConfig

        model_cfg = HaloVLMConfig()
        system_msg = f"{self.cfg.system_prefix}{model_cfg.system_prompt}{self.cfg.turn_end}"
        human_msg = f"{self.cfg.human_prefix}{user_body}{self.cfg.turn_end}"
        asst_msg = f"{self.cfg.assistant_prefix}{assistant_body}{self.cfg.turn_end}"
        full_text = system_msg + human_msg + asst_msg

        encoding = self.tokenizer(
            full_text,
            max_length=self.cfg.max_seq_len,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        # Mask system + user spans (same heuristic as EODataset)
        text_parts = [system_msg, human_msg, asst_msg]
        role_is_assistant = [False, False, True]
        labels = self._mask_non_assistant(full_text, text_parts, role_is_assistant, labels)
        labels[attention_mask == 0] = -100
        return input_ids, attention_mask, labels

    def _mask_non_assistant(
        self,
        full_text: str,
        text_parts: list[str],
        role_is_assistant: list[bool],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        cursor = 0
        for part, is_assistant in zip(text_parts, role_is_assistant):
            start_char = full_text.find(part, cursor)
            if start_char == -1:
                continue
            end_char = start_char + len(part)
            if not is_assistant:
                prefix = full_text[:end_char]
                prefix_ids = self.tokenizer(
                    prefix,
                    max_length=self.cfg.max_seq_len,
                    truncation=True,
                    add_special_tokens=False,
                )["input_ids"]
                start_prefix = full_text[:start_char]
                start_ids = self.tokenizer(
                    start_prefix,
                    max_length=self.cfg.max_seq_len,
                    truncation=True,
                    add_special_tokens=False,
                )["input_ids"]
                tok_start = len(start_ids)
                tok_end = len(prefix_ids)
                if tok_end <= labels.numel():
                    labels[tok_start:tok_end] = -100
            cursor = end_char
        return labels
