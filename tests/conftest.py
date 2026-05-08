"""Shared pytest fixtures.

These tests are CPU-only and never download model weights, so CI can run them.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def tiny_gray_image() -> Image.Image:
    """A 64×64 grayscale CXR-ish blob: bright lung-shaped circle in a darker frame."""
    arr = np.zeros((64, 64), dtype=np.uint8)
    yy, xx = np.mgrid[0:64, 0:64]
    arr[((xx - 22) ** 2 + (yy - 32) ** 2) < 12 ** 2] = 200  # left "lung"
    arr[((xx - 42) ** 2 + (yy - 32) ** 2) < 12 ** 2] = 200  # right "lung"
    return Image.fromarray(arr, mode="L")


@pytest.fixture
def tiny_concept_masks() -> dict:
    """Two binary masks aligned with the tiny_gray_image."""
    left = np.zeros((64, 64), dtype=np.uint8)
    right = np.zeros((64, 64), dtype=np.uint8)
    yy, xx = np.mgrid[0:64, 0:64]
    left[((xx - 22) ** 2 + (yy - 32) ** 2) < 12 ** 2] = 1
    right[((xx - 42) ** 2 + (yy - 32) ** 2) < 12 ** 2] = 1
    return {"left lung": left, "right lung": right}


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]
