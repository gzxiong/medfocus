"""Lightweight overlay helpers for figures and notebooks."""

from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


def visualize_heatmap(img, heatmap, alpha: float = 0.45, ax=None):
    """Show `heatmap` overlaid on `img`. `ax` allows composing into a panel."""
    if hasattr(heatmap, "detach"):
        heatmap = heatmap.detach().cpu().numpy()

    img = np.array(img)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.dtype not in (np.float32, np.float64):
        img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    img = np.clip(img, 0.0, 1.0)

    hm = np.asarray(heatmap, dtype=np.float32)
    if hm.ndim != 2:
        raise ValueError(f"heatmap must be (H,W), got {hm.shape}")
    hm = hm - hm.min()
    hm = hm / (hm.max() + 1e-8)

    own_ax = ax is None
    if own_ax:
        plt.figure(figsize=(5, 5))
        ax = plt.gca()
    ax.imshow(img)
    ax.imshow(hm, alpha=alpha)
    ax.axis("off")
    if own_ax:
        plt.tight_layout()
        plt.show()


def overlay_boxes(
    img: Image.Image,
    boxes: Iterable[Sequence[int]],
    *,
    color: str = "yellow",
    width: int = 3,
    label: str | None = None,
) -> Image.Image:
    """Draw `boxes` (xyxy) on a copy of `img`."""
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for b in boxes:
        draw.rectangle(tuple(b), outline=color, width=width)
    if label:
        draw.text((4, 4), label, fill=color)
    return out
