"""Image I/O round-trip tests (no network, no GPU)."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from medfocus.data.io import (
    keep_image_regions,
    mask_image_regions,
    resize_pad,
    safe_open_image,
)


def test_resize_pad_square_output(tiny_gray_image, tmp_path):
    out = resize_pad(tiny_gray_image, 96)
    assert out.size == (96, 96)


def test_resize_pad_preserves_aspect(tiny_gray_image):
    rect = tiny_gray_image.resize((64, 32))
    out = resize_pad(rect, 64)
    assert out.size == (64, 64)


def test_mask_and_keep_image_regions(tiny_gray_image):
    boxes = [(10, 10, 30, 30)]
    masked = np.array(mask_image_regions(tiny_gray_image, boxes))
    assert masked[15, 15] == 0
    assert masked[5, 5] == np.array(tiny_gray_image)[5, 5]

    kept = np.array(keep_image_regions(tiny_gray_image, boxes))
    assert kept[15, 15] == np.array(tiny_gray_image)[15, 15]
    assert kept[5, 5] == 0


def test_safe_open_image_reads_back(tiny_gray_image, tmp_path):
    p = tmp_path / "tiny.png"
    tiny_gray_image.save(p)
    img = safe_open_image(p)
    assert img is not None
    assert img.mode == "L"
    assert img.size == tiny_gray_image.size


def test_safe_open_image_returns_none_for_missing(tmp_path):
    assert safe_open_image(tmp_path / "missing.png") is None
