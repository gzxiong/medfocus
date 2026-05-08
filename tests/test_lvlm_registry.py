"""Verify the model registry covers all 6 paper LVLMs."""

from __future__ import annotations

import pytest

from medfocus.lvlm.registry import MODEL_REGISTRY, get_model_spec


_EXPECTED = {
    "qwen2_5_vl_3b", "qwen2_5_vl_7b",
    "gemma3_4b", "gemma3_12b",
    "medgemma_4b", "medgemma1_5_4b",
}


def test_registry_has_all_paper_models():
    assert _EXPECTED.issubset(set(MODEL_REGISTRY))


@pytest.mark.parametrize("key", sorted(_EXPECTED))
def test_get_model_spec_resolves(key):
    spec = get_model_spec(key)
    assert spec.hf_id
    assert spec.family in ("qwen", "gemma")


def test_get_model_spec_rejects_unknown():
    with pytest.raises(KeyError):
        get_model_spec("not_a_real_model")
