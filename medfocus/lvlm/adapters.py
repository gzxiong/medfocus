"""LVLM adapters.

Encapsulate the per-family differences (image-token id, processor template,
teacher-forced input construction) so the rest of the package can speak to
"any LVLM" through a uniform interface.

The two supported families ("qwen" and "gemma") cover all six paper LVLMs.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
from PIL import Image
from transformers import PreTrainedModel, ProcessorMixin

from medfocus.lvlm.registry import ModelSpec


_GEMMA_IMAGE_TOKENS = ("<|image_pad|>", "<image>", "<image_soft_token>", "<|vision_start|>")


def _resolve_image_token_id(processor: ProcessorMixin, hint: Optional[str]) -> int:
    """Find the int id of the image-patch token.

    Tries (in order): tokenizer.image_token_id / img_token_id, the optional
    hint string, and the fallback list. Raises if nothing matches.
    """
    tok = processor.tokenizer
    for attr in ("image_token_id", "img_token_id"):
        v = getattr(tok, attr, None)
        if v is not None:
            return int(v)
    candidates = ([hint] if hint else []) + list(_GEMMA_IMAGE_TOKENS)
    for s in candidates:
        if s is None:
            continue
        t = tok.convert_tokens_to_ids(s)
        if t is not None and t != tok.unk_token_id:
            return int(t)
    raise RuntimeError("Could not determine image token id for this processor.")


class LVLMAdapter:
    """Family-agnostic LVLM wrapper.

    Holds (model, processor, spec) plus a cached image-token id, and provides
    the small set of operations attribution code actually needs.
    """

    def __init__(self, model: PreTrainedModel, processor: ProcessorMixin, spec: ModelSpec):
        self.model = model
        self.processor = processor
        self.spec = spec
        self.image_token_id = _resolve_image_token_id(processor, spec.image_token)

    # ----- prompt construction -----

    def _user_message(self, img: Image.Image, question: str) -> list[dict]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": question},
                ],
            }
        ]

    def _assistant_message(self, answer: str) -> dict:
        return {"role": "assistant", "content": [{"type": "text", "text": answer}]}

    def build_generation_inputs(self, img: Image.Image, question: str) -> dict:
        """Tokenize a user message ready for `.generate()`."""
        messages = self._user_message(img, question)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs.to(self.model.device, dtype=self.model.dtype)

    def build_teacher_forced_inputs(
        self, img: Image.Image, question: str, answer: str
    ) -> tuple[dict, dict]:
        """Build (user_inputs, full_inputs) for teacher-forced scoring.

        `user_inputs` is the prefix used to count the response start; `full_inputs`
        contains user+assistant for forward passes that need to score the answer.
        """
        user_messages = self._user_message(img, question)
        full_messages = user_messages + [self._assistant_message(answer)]

        user_inputs = self.processor.apply_chat_template(
            user_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        full_inputs = self.processor.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=self.model.dtype)
        return user_inputs, full_inputs

    # ----- token positions -----

    def image_token_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return the indices in input_ids[0] that hold image-patch tokens."""
        ids = input_ids[0]
        return (ids == self.image_token_id).nonzero(as_tuple=False).squeeze(-1)

    def target_span(self, user_inputs: dict, full_inputs: dict) -> tuple[int, int]:
        """[ts, te) span in `full_inputs.input_ids[0]` corresponding to the assistant answer."""
        ts = int(user_inputs["input_ids"].shape[1])
        te = int(full_inputs["input_ids"].shape[1])
        return ts, te
