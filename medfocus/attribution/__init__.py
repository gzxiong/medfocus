"""Attribution methods + evaluation utilities."""

from medfocus.attribution.eval import (
    compute_metrics_table,
    eval_union_boxes,
    heatmap_to_bboxes_quantile,
    bboxes_rescale,
)
from medfocus.attribution.medfocus import (
    MedFocus,
    AttributionResult,
)

__all__ = [
    "MedFocus",
    "AttributionResult",
    "compute_metrics_table",
    "eval_union_boxes",
    "heatmap_to_bboxes_quantile",
    "bboxes_rescale",
]
