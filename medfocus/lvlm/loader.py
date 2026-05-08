"""Public entry-point for instantiating an LVLM adapter."""

from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from medfocus.lvlm.adapters import LVLMAdapter
from medfocus.lvlm.registry import ModelSpec, get_model_spec


_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "auto": "auto",
}


def load_lvlm(
    key: str,
    *,
    device: str | torch.device | None = None,
    dtype: Optional[str] = None,
    attn_implementation: str = "eager",
) -> LVLMAdapter:
    """Load model + processor from HuggingFace and wrap them in an adapter.

    `attn_implementation="eager"` is the default because attention-based
    attribution methods (Attention Head, Rollout, LRP, Gradient-weighted
    Attention) need `output_attentions=True`, which the SDPA / Flash kernels
    do not support reliably. Override to `"sdpa"` if you only need MedFocus.
    """
    spec: ModelSpec = get_model_spec(key)
    torch_dtype = _DTYPES.get(dtype or spec.dtype, "auto")

    model = AutoModelForImageTextToText.from_pretrained(
        spec.hf_id,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        device_map=device or "auto",
    )
    processor = AutoProcessor.from_pretrained(spec.hf_id)
    model.eval()
    return LVLMAdapter(model=model, processor=processor, spec=spec)
