"""LVLM loading and adapter system.

`load_lvlm("qwen2_5_vl_3b")` returns a fully wrapped :class:`LVLMAdapter`
ready for both generation and teacher-forced attribution.
"""

from medfocus.lvlm.adapters import LVLMAdapter
from medfocus.lvlm.registry import MODEL_REGISTRY, get_model_spec
from medfocus.lvlm.loader import load_lvlm

__all__ = ["load_lvlm", "LVLMAdapter", "MODEL_REGISTRY", "get_model_spec"]
