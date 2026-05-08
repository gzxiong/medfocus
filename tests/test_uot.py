"""Unit tests for the UOT primitives."""

from __future__ import annotations

import numpy as np

from medfocus.ot.mapping import compute_ot_mapping_uot, map_mask_through_ot
from medfocus.ot.sinkhorn import sinkhorn_uot


def test_sinkhorn_uot_basic_shape():
    a = np.ones(5) / 5
    b = np.ones(7) / 7
    rng = np.random.default_rng(0)
    C = rng.uniform(0, 1, size=(5, 7))
    P = sinkhorn_uot(a, b, C, epsilon=0.1, lambda_marginal=1.0, max_iter=50)
    assert P.shape == (5, 7)
    assert (P >= 0).all()


def test_sinkhorn_uot_converges_toward_marginals():
    a = np.array([0.6, 0.4])
    b = np.array([0.3, 0.7])
    C = np.array([[0.0, 1.0], [1.0, 0.0]])
    # Large lambda_marginal should pull marginals close to (a, b).
    P = sinkhorn_uot(a, b, C, epsilon=0.05, lambda_marginal=20.0, max_iter=400)
    np.testing.assert_allclose(P.sum(axis=1), a, atol=0.05)
    np.testing.assert_allclose(P.sum(axis=0), b, atol=0.05)


def test_compute_ot_mapping_uot_self(tiny_gray_image):
    arr = np.array(tiny_gray_image)
    res = compute_ot_mapping_uot(arr, arr, downsample=2, epsilon=0.05, lambda_marginal=0.1, max_iter=100)
    assert res["transport_plan"].shape[0] == res["source_coords_ds"].shape[0]
    assert res["transport_plan"].shape[1] == res["target_coords_ds"].shape[0]
    assert np.isfinite(res["cost"])


def test_map_mask_through_ot_returns_nonempty(tiny_gray_image, tiny_concept_masks):
    arr = np.array(tiny_gray_image)
    res = compute_ot_mapping_uot(arr, arr, downsample=2, epsilon=0.05, lambda_marginal=0.1, max_iter=100)
    out = map_mask_through_ot(tiny_concept_masks["left lung"], res, output_binary=True, mass_quantile=0.95)
    assert out["mapped_binary_mask"].shape == arr.shape
    assert out["mapped_binary_mask"].sum() > 0
