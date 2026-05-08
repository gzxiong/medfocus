"""Generation + teacher-forced forward helpers built on top of an adapter."""

from __future__ import annotations

from typing import Optional

import torch
from PIL import Image

from medfocus.lvlm.adapters import LVLMAdapter


@torch.inference_mode()
def generate_answer(
    adapter: LVLMAdapter,
    img: Image.Image,
    question: str,
    *,
    max_new_tokens: int = 256,
    suffix: str = "",
    do_sample: bool = False,
) -> str:
    """Greedy generation of a single answer string."""
    inputs = adapter.build_generation_inputs(img, question + suffix)
    gen = adapter.model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        return_dict_in_generate=True,
    )
    prompt_len = inputs["input_ids"].shape[1]
    new_ids = gen.sequences[0, prompt_len:]
    return adapter.processor.batch_decode(
        [new_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def teacher_forced_forward(
    adapter: LVLMAdapter,
    img: Image.Image,
    question: str,
    answer: str,
    *,
    output_attentions: bool = True,
    output_hidden_states: bool = True,
):
    """Run a single teacher-forced forward pass; returns (user_inputs, full_inputs, outputs)."""
    user_inputs, full_inputs = adapter.build_teacher_forced_inputs(img, question, answer)
    with torch.inference_mode():
        outputs = adapter.model(
            **full_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_scores=True,
        )
    return user_inputs, full_inputs, outputs
