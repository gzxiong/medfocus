"""Tests for heatmap → bbox conversion and union-region IoU."""

from __future__ import annotations

import numpy as np

from medfocus.attribution.eval import (
    bboxes_rescale,
    eval_union_boxes,
    heatmap_to_bboxes_quantile,
)


def test_eval_union_boxes_perfect_match():
    preds = [[(0, 0, 10, 10)]]
    gts = [[(0, 0, 10, 10)]]
    m = eval_union_boxes(preds, gts)
    assert m["iou"] == [1.0]
    assert m["f1"] == [1.0]


def test_eval_union_boxes_partial_overlap():
    preds = [[(0, 0, 10, 10)]]
    gts = [[(5, 5, 15, 15)]]
    # Intersection 25, Union 175
    m = eval_union_boxes(preds, gts)
    np.testing.assert_allclose(m["iou"][0], 25 / 175, rtol=1e-6)


def test_eval_union_boxes_empty_pred():
    preds = [[]]
    gts = [[(0, 0, 10, 10)]]
    m = eval_union_boxes(preds, gts)
    assert m["iou"][0] == 0.0
    assert m["recall"][0] == 0.0


def test_heatmap_to_bboxes_quantile_finds_blob():
    h = np.zeros((32, 32), dtype=np.float32)
    h[5:15, 5:15] = 1.0
    comps = heatmap_to_bboxes_quantile(h, q=0.5, min_area=10)
    assert len(comps) == 1
    x1, y1, x2, y2 = comps[0]["bbox"]
    assert x1 == 5 and y1 == 5
    assert x2 == 14 and y2 == 14


def test_bboxes_rescale_identity():
    boxes = [[0, 0, 10, 10]]
    out = bboxes_rescale(boxes, orig_size=(20, 20), heatmap_size=(20, 20))
    assert out == [[0, 0, 10, 10]]
