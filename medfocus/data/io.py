"""Image I/O helpers used across the pipeline.

Handles the mix of JPEG, PNG, 16-bit, and DICOM inputs found in MIMIC-CXR-JPG /
VinDR-CXR / PadChest-GR.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pydicom
from PIL import Image, ImageOps


def safe_open_image(path: str | Path) -> Image.Image | None:
    """Open a CXR file as 8-bit grayscale (`mode="L"`).

    Supports JPEG/PNG/16-bit TIFF/DICOM. Returns None on read failure rather
    than raising, since downstream code already filters empty samples.
    """
    p = str(path)
    try:
        if p.lower().endswith((".dcm", ".dicom")):
            return dicom_to_pil(p, force_8bit=True)

        with Image.open(p) as img:
            mode = img.mode
            if mode == "L":
                return img.copy()
            if mode in ("I;16", "I;16B", "I;16L"):
                arr = np.array(img, dtype=np.uint16)
                lo, hi = int(arr.min()), int(arr.max())
                arr8 = (
                    np.zeros(arr.shape, dtype=np.uint8)
                    if hi == lo
                    else ((arr - lo) * 255.0 / (hi - lo)).astype(np.uint8)
                )
                return Image.fromarray(arr8, mode="L")
            if mode in ("I", "F"):
                arr = np.array(img, dtype=np.float32)
                lo, hi = float(arr.min()), float(arr.max())
                arr8 = (
                    np.zeros(arr.shape, dtype=np.uint8)
                    if hi == lo
                    else ((arr - lo) * 255.0 / (hi - lo)).astype(np.uint8)
                )
                return Image.fromarray(arr8, mode="L")
            return img.convert("L").copy()
    except Exception:
        return None


def dicom_to_pil(path: str | Path, force_8bit: bool = True) -> Image.Image:
    """Read a DICOM file and apply windowing + photometric correction."""
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr

    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth", None)
    if wc is not None and ww is not None:
        wc = float(wc[0]) if isinstance(wc, (list, pydicom.multival.MultiValue)) else float(wc)
        ww = float(ww[0]) if isinstance(ww, (list, pydicom.multival.MultiValue)) else float(ww)
        arr = np.clip(arr, wc - ww / 2.0, wc + ww / 2.0)
    else:
        arr = np.clip(arr, arr.min(), arr.max())

    arr -= arr.min()
    denom = arr.max() if arr.max() != 0 else 1.0
    arr = arr / denom

    if force_8bit:
        return Image.fromarray((arr * 255.0).round().astype(np.uint8), mode="L")
    return Image.fromarray((arr * 65535.0).round().astype(np.uint16), mode="I;16")


def mask_image_regions(img: Image.Image, boxes: Iterable[tuple[int, int, int, int]]) -> Image.Image:
    """Zero out the pixels inside each (x1, y1, x2, y2) box."""
    arr = np.array(img.copy())
    for x1, y1, x2, y2 in boxes:
        arr[y1:y2, x1:x2] = 0
    return Image.fromarray(arr)


def keep_image_regions(
    img: Image.Image,
    boxes: Iterable[tuple[int, int, int, int]],
    masked_value: int = 0,
) -> Image.Image:
    """Keep only the pixels inside `boxes`; fill the rest with `masked_value`."""
    arr = np.array(img)
    kept = np.zeros_like(arr) + masked_value
    for x1, y1, x2, y2 in boxes:
        kept[y1:y2, x1:x2] = arr[y1:y2, x1:x2]
    return Image.fromarray(kept)


def resize_pad(img: Image.Image, width: int, fill: int = 0) -> Image.Image:
    """Resize so the longer side == `width`, then pad to a square (width, width)."""
    if width <= 0:
        raise ValueError("width must be a positive integer")
    w, h = img.size
    if w == 0 or h == 0:
        raise ValueError("image has invalid size")

    scale = width / float(max(w, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = img.resize((new_w, new_h), resample=Image.BICUBIC)

    delta_w = width - new_w
    delta_h = width - new_h
    left = delta_w // 2
    top = delta_h // 2
    return ImageOps.expand(resized, border=(left, top, delta_w - left, delta_h - top), fill=fill)
