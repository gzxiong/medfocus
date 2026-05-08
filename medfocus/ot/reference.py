"""Reference-image pool for concept transfer.

A "reference" is a normal CXR with annotated masks for all 11 anatomical
concepts. At inference time MedFocus picks the reference whose UOT cost to
the current target image is minimized at a coarse 14x14 grid; the actual
mask transfer then runs at a finer 56x56 grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image

from medfocus.data.io import safe_open_image, resize_pad
from medfocus.ot.mapping import compute_ot_mapping_uot


@dataclass
class ReferenceCXR:
    """One normal CXR with named binary masks (one per concept)."""

    image_path: str
    image: Image.Image
    masks: dict[str, np.ndarray]  # concept -> (H, W) binary

    @classmethod
    def load(cls, image_path: str | Path, masks_dir: str | Path, concepts: Iterable[str]) -> "ReferenceCXR":
        image_path = str(image_path)
        img = safe_open_image(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not open reference image: {image_path}")
        stem = Path(image_path).stem
        masks: dict[str, np.ndarray] = {}
        for c in concepts:
            mp = Path(masks_dir) / stem / f"{_concept_to_filename(c)}.npy"
            if mp.exists():
                masks[c] = np.load(mp).astype(np.uint8)
        return cls(image_path=image_path, image=img, masks=masks)


def _concept_to_filename(c: str) -> str:
    return c.replace(" ", "_").replace("/", "_").lower()


class ReferenceCXRPool:
    """Pool of reference CXRs; picks the lowest-UOT-cost candidate per target."""

    def __init__(
        self,
        references: list[ReferenceCXR],
        *,
        selection_grid: int = 14,
        epsilon: float = 0.05,
        lambda_marginal: float = 0.1,
    ):
        if not references:
            raise ValueError("ReferenceCXRPool requires at least one ReferenceCXR.")
        self.references = references
        self.selection_grid = int(selection_grid)
        self.epsilon = float(epsilon)
        self.lambda_marginal = float(lambda_marginal)

    @classmethod
    def from_directory(
        cls,
        images_dir: str | Path,
        masks_dir: str | Path,
        concepts: Iterable[str],
        *,
        candidates: Optional[Iterable[str]] = None,
        **kwargs,
    ) -> "ReferenceCXRPool":
        images_dir = Path(images_dir)
        if candidates is None:
            paths = sorted(p for p in images_dir.iterdir() if p.is_file())
        else:
            paths = [images_dir / c for c in candidates]
        refs = [ReferenceCXR.load(p, masks_dir, concepts) for p in paths]
        return cls(refs, **kwargs)

    def select_best(self, target: Image.Image, image_size: int = 224) -> tuple[ReferenceCXR, float]:
        """Return the reference whose UOT cost to `target` is lowest at the coarse grid.

        `image_size` should match the resolution at which downstream concept
        transfer runs; we also use it for the selection step so distances
        are commensurable across calls.
        """
        target_arr = np.array(resize_pad(target.convert("L"), image_size))
        best: Optional[ReferenceCXR] = None
        best_cost = float("inf")
        ds = max(image_size // self.selection_grid, 1)

        for ref in self.references:
            ref_arr = np.array(resize_pad(ref.image.convert("L"), image_size))
            res = compute_ot_mapping_uot(
                ref_arr,
                target_arr,
                downsample=ds,
                epsilon=self.epsilon,
                lambda_marginal=self.lambda_marginal,
            )
            if res["cost"] < best_cost:
                best_cost = float(res["cost"])
                best = ref
        assert best is not None
        return best, best_cost
