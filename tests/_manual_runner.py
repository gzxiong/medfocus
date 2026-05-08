"""Standalone runner used in environments without pytest.

Run via:  python tests/_manual_runner.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

# IO ------------------------------------------------------------------------

def test_io():
    from medfocus.data.io import keep_image_regions, mask_image_regions, resize_pad

    arr = np.zeros((64, 64), dtype=np.uint8)
    yy, xx = np.mgrid[0:64, 0:64]
    arr[((xx - 22) ** 2 + (yy - 32) ** 2) < 144] = 200
    arr[((xx - 42) ** 2 + (yy - 32) ** 2) < 144] = 200
    img = Image.fromarray(arr, mode="L")

    assert resize_pad(img, 96).size == (96, 96)
    assert resize_pad(img.resize((64, 32)), 64).size == (64, 64)
    assert np.array(mask_image_regions(img, [(10, 10, 30, 30)]))[15, 15] == 0
    assert np.array(keep_image_regions(img, [(10, 10, 30, 30)]))[5, 5] == 0


# UOT -----------------------------------------------------------------------

def test_uot():
    from medfocus.ot.mapping import compute_ot_mapping_uot, map_mask_through_ot
    from medfocus.ot.sinkhorn import sinkhorn_uot

    rng = np.random.default_rng(0)
    P = sinkhorn_uot(np.ones(5) / 5, np.ones(7) / 7, rng.uniform(0, 1, (5, 7)),
                     epsilon=0.1, lambda_marginal=1.0, max_iter=50)
    assert P.shape == (5, 7) and (P >= 0).all()

    P = sinkhorn_uot(np.array([0.6, 0.4]), np.array([0.3, 0.7]),
                     np.array([[0, 1], [1, 0]], dtype=float),
                     epsilon=0.05, lambda_marginal=20.0, max_iter=400)
    np.testing.assert_allclose(P.sum(1), [0.6, 0.4], atol=0.05)
    np.testing.assert_allclose(P.sum(0), [0.3, 0.7], atol=0.05)

    arr = np.zeros((64, 64), dtype=np.uint8)
    yy, xx = np.mgrid[0:64, 0:64]
    arr[((xx - 22) ** 2 + (yy - 32) ** 2) < 144] = 200

    res = compute_ot_mapping_uot(arr, arr, downsample=2, epsilon=0.05, lambda_marginal=0.1, max_iter=100)
    assert np.isfinite(res["cost"])
    out = map_mask_through_ot((arr > 100).astype(np.uint8), res, output_binary=True, mass_quantile=0.95)
    assert out["mapped_binary_mask"].sum() > 0


# Eval ----------------------------------------------------------------------

def test_eval():
    from medfocus.attribution.eval import bboxes_rescale, eval_union_boxes, heatmap_to_bboxes_quantile

    m = eval_union_boxes([[(0, 0, 10, 10)]], [[(0, 0, 10, 10)]])
    assert m["iou"] == [1.0] and m["f1"] == [1.0]

    m = eval_union_boxes([[(0, 0, 10, 10)]], [[(5, 5, 15, 15)]])
    np.testing.assert_allclose(m["iou"][0], 25 / 175, rtol=1e-6)

    m = eval_union_boxes([[]], [[(0, 0, 10, 10)]])
    assert m["iou"][0] == 0.0

    h = np.zeros((32, 32), dtype=np.float32)
    h[5:15, 5:15] = 1.0
    comps = heatmap_to_bboxes_quantile(h, q=0.5, min_area=10)
    assert len(comps) == 1 and comps[0]["bbox"] == (5, 5, 14, 14)

    assert bboxes_rescale([[0, 0, 10, 10]], orig_size=(20, 20), heatmap_size=(20, 20)) == [[0, 0, 10, 10]]


# Registry ------------------------------------------------------------------

def test_registry():
    from medfocus.lvlm.registry import MODEL_REGISTRY, get_model_spec

    expected = {
        "qwen2_5_vl_3b", "qwen2_5_vl_7b",
        "gemma3_4b", "gemma3_12b",
        "medgemma_4b", "medgemma1_5_4b",
    }
    assert expected.issubset(set(MODEL_REGISTRY))
    for k in expected:
        s = get_model_spec(k)
        assert s.hf_id and s.family in ("qwen", "gemma")


# Configs -------------------------------------------------------------------

def test_configs():
    os.environ["MEDFOCUS_DATA_ROOT"] = "/tmp/fake"
    from medfocus.config import (
        load_datasets, load_medfocus, load_models, load_radedit, load_reference_pool,
    )

    repo = Path(__file__).resolve().parents[1]
    mc = load_models(repo / "configs/models.yaml")
    assert "qwen2_5_vl_3b" in mc.models
    assert mc.models["medgemma1_5_4b"].family == "gemma"

    mfc = load_medfocus(repo / "configs/medfocus.yaml")
    assert len(mfc.concepts) == 11
    assert "cardiac silhouette" in mfc.concepts
    assert mfc.intervention.tau == 0.75
    assert mfc.ot.transfer_grid == 56
    assert set(mfc.composites["bilateral_lungs"]) == {"left lung", "right lung"}

    dc = load_datasets(repo / "configs/datasets.yaml")
    assert dc.data_root == "/tmp/fake"
    assert "imagenome" in dc.datasets

    rc = load_radedit(repo / "configs/radedit.yaml")
    assert rc.radedit and "foreground" in rc.prompts

    rp = load_reference_pool(repo / "configs/reference_pool.yaml")
    assert rp.images_dir.endswith("images")


# MedGround-Bench JSON ------------------------------------------------------

def test_medground_json():
    from medfocus.data.medground_bench import load_medground

    repo = Path(__file__).resolve().parents[1]
    direct = load_medground(repo / "data/medground_bench", split="direct")
    reasoning = load_medground(repo / "data/medground_bench", split="reasoning")
    assert len(direct) == 1880
    assert len(reasoning) == 2060

    fields = {"model", "dataset", "index", "answer", "prediction_direct"}
    assert fields.issubset(direct[0].keys())
    assert {r["model"] for r in direct} == {
        "qwen2_5_vl_3b", "qwen2_5_vl_7b",
        "gemma3_4b", "gemma3_12b",
        "medgemma_4b", "medgemma1_5_4b",
    }


# Main ----------------------------------------------------------------------

if __name__ == "__main__":
    suites = [
        ("io", test_io),
        ("uot", test_uot),
        ("eval", test_eval),
        ("registry", test_registry),
        ("configs", test_configs),
        ("medground_json", test_medground_json),
    ]
    failed = 0
    for name, fn in suites:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
            failed += 1
    if failed:
        sys.exit(f"{failed}/{len(suites)} test suites failed")
    print(f"\nAll {len(suites)} test suites passed.")
