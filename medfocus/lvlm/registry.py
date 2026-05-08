"""Hard-coded LVLM registry.

Mirrors `configs/models.yaml` so unit tests can resolve model keys without
parsing YAML. Edit the YAML for runtime overrides; this module is the
fall-back when no config is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    family: str  # "qwen" | "gemma"
    dtype: str = "bfloat16"
    image_token: Optional[str] = None


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "qwen2_5_vl_3b": ModelSpec(
        key="qwen2_5_vl_3b",
        hf_id="Qwen/Qwen2.5-VL-3B-Instruct",
        family="qwen",
        image_token="<|image_pad|>",
    ),
    "qwen2_5_vl_7b": ModelSpec(
        key="qwen2_5_vl_7b",
        hf_id="Qwen/Qwen2.5-VL-7B-Instruct",
        family="qwen",
        image_token="<|image_pad|>",
    ),
    "gemma3_4b": ModelSpec(
        key="gemma3_4b",
        hf_id="google/gemma-3-4b-it",
        family="gemma",
        image_token="<image_soft_token>",
    ),
    "gemma3_12b": ModelSpec(
        key="gemma3_12b",
        hf_id="google/gemma-3-12b-it",
        family="gemma",
        image_token="<image_soft_token>",
    ),
    "medgemma_4b": ModelSpec(
        key="medgemma_4b",
        hf_id="google/medgemma-4b-it",
        family="gemma",
        image_token="<image_soft_token>",
    ),
    "medgemma1_5_4b": ModelSpec(
        key="medgemma1_5_4b",
        hf_id="google/medgemma-1.5-4b-it",
        family="gemma",
        image_token="<image_soft_token>",
    ),
}


def get_model_spec(key: str) -> ModelSpec:
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model key {key!r}; available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key]
