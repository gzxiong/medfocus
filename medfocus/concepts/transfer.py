"""Transfer anatomical concept masks from a reference CXR onto a target CXR.

Pipeline (see Algorithm 1 in the paper):
  1. Compute UOT plan from `reference_image` to `target_image` at `transfer_grid`.
  2. For each concept c with reference mask M_c:
        push M_c through the plan (top-`mass_quantile` core);
        take the tight bbox of the resulting target-mask;
        prompt MedSAM with that bbox to get a refined mask;
        report the refined tight bbox.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from PIL import Image

from medfocus.data.io import resize_pad
from medfocus.medsam.client import MedSAMClient
from medfocus.ot.mapping import compute_ot_mapping_uot, map_mask_through_ot


def _mask_tight_bbox(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def transfer_concept_masks(
    reference_image: Image.Image,
    target_image: Image.Image,
    reference_masks: dict[str, np.ndarray],
    *,
    medsam: MedSAMClient,
    image_size: int = 224,
    transfer_grid: int = 56,
    epsilon: float = 0.05,
    lambda_marginal: float = 0.1,
    mass_quantile: float = 0.75,
    sinkhorn_max_iter: int = 500,
    sinkhorn_tol: float = 1e-6,
) -> dict[str, tuple[int, int, int, int]]:
    """Return concept -> (x1, y1, x2, y2) in target-image pixel space.

    `reference_masks[c]` should be a binary mask the same size as the (resized)
    reference image at `image_size`. Concepts without a reference mask are
    silently skipped.
    """
    ref_arr = np.array(resize_pad(reference_image.convert("L"), image_size))
    tgt_arr = np.array(resize_pad(target_image.convert("L"), image_size))
    ds = max(image_size // transfer_grid, 1)

    plan = compute_ot_mapping_uot(
        ref_arr,
        tgt_arr,
        downsample=ds,
        epsilon=epsilon,
        lambda_marginal=lambda_marginal,
        max_iter=sinkhorn_max_iter,
        tol=sinkhorn_tol,
    )

    target_pil = Image.fromarray(tgt_arr)
    out: dict[str, tuple[int, int, int, int]] = {}
    for concept, mask in reference_masks.items():
        if mask is None:
            continue
        m = np.asarray(mask)
        if m.shape != ref_arr.shape:
            # Resize mask if it's at native resolution
            mp = Image.fromarray((m > 0).astype(np.uint8) * 255).convert("L")
            mp = resize_pad(mp, image_size)
            m = (np.array(mp) > 127).astype(np.uint8)
        transferred = map_mask_through_ot(m, plan, output_binary=True, mass_quantile=mass_quantile)
        bbox = _mask_tight_bbox(transferred["mapped_binary_mask"])
        if bbox is None:
            continue
        out[concept] = medsam.refine_box(target_pil, bbox)
    return out
